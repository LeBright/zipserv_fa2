#include "Fa2.h"
#include "softmax.h"
#include "L_Kernel.cuh"
#include <cutlass/numeric_conversion.h>

__global__ void compute_attn_v2(void* O_ptr, 
                                const void* Q_ptr, const void* K_ptr, const void* V_ptr, 
                                int q_len, int k_len, int v_len,  int o_len,
                                int row_stride, 
                                float sm_scale)
{
    const int m_block = blockIdx.x; // one block process Q_len/kBlockM (AKA N/Br in fa2 paper ) m_blocks 
    const int base_id = blockIdx.y; // it tells us the block process which head(base_id%headnum) of which batch(base_id/headnum) 
    const int tidx = threadIdx.x;  
    
    if(m_block * kBlockM >= q_len) return; // if the block start processing token id exceed actual q_len, return directly

    extern __shared__ __nv_bfloat16 smem[];
    auto Q_smem_ptr = smem;
    auto K_smem_ptr = Q_smem_ptr + cosize(SmemLayoutQ{});
    auto V_smem_ptr = K_smem_ptr + cosize(SmemLayoutK{});

    auto base_offset = base_id * HeadDim;

    auto Q = make_tensor(make_gmem_ptr<__nv_bfloat16>((__nv_bfloat16*)(Q_ptr) + base_offset),
                         make_shape(q_len,Int<HeadDim>{}),
                         make_stride(row_stride, _1{}));
    auto K = make_tensor(make_gmem_ptr<__nv_bfloat16>((__nv_bfloat16*)(K_ptr) + base_offset),
                        make_shape(k_len,Int<HeadDim>{}),
                        make_stride(row_stride, _1{}));
    auto V = make_tensor(make_gmem_ptr<__nv_bfloat16>((__nv_bfloat16*)(V_ptr) + base_offset),
                        make_shape(v_len,Int<HeadDim>{}),
                        make_stride(row_stride, _1{}));
    auto O = make_tensor(make_gmem_ptr<__nv_bfloat16>((__nv_bfloat16*)(O_ptr) + base_offset),
                        make_shape(o_len,Int<HeadDim>{}),
                        make_stride(row_stride, _1{}));

    // global memory
    auto gQ = local_tile(Q, make_tile(Int<kBlockM>{}, Int<HeadDim>{}),
                         make_coord(m_block, _));
    auto gK = local_tile(K,
                         make_tile(Int<kBlockN>{}, Int<HeadDim>{}),
                         make_coord(0, _));
    auto gV = local_tile(V,
                         make_tile(Int<kBlockN>{}, Int<HeadDim>{}),
                         make_coord(0, _));
    
    // shared memory
    auto sQ = make_tensor(make_smem_ptr<__nv_bfloat16>(Q_smem_ptr), SmemLayoutQ{});
    auto sK = make_tensor(make_smem_ptr<__nv_bfloat16>(K_smem_ptr), SmemLayoutK{});
    auto sV = make_tensor(make_smem_ptr<__nv_bfloat16>(V_smem_ptr), SmemLayoutV{});
    auto sVt = make_tensor(make_smem_ptr<__nv_bfloat16>(V_smem_ptr), SmemLayoutVt{});
    auto sVtNoSwizzle = make_tensor(make_smem_ptr<__nv_bfloat16>(V_smem_ptr), SmemLayoutVtNoSwizzle{});

    // global memory to shared memory copy
    GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);
    auto tQgQ = gmem_thr_copy_QKV.partition_S(gQ(_, _, 0));
    auto tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    auto tKgK = gmem_thr_copy_QKV.partition_S(gK(_, _, 0));
    auto tKsK = gmem_thr_copy_QKV.partition_D(sK);
    auto tVgV = gmem_thr_copy_QKV.partition_S(gV(_, _, 0));
    auto tVsV = gmem_thr_copy_QKV.partition_D(sV);

    // register
    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    auto tSrQ = thr_mma.partition_fragment_A(sQ);             // (MMA,MMA_M,MMA_K)
    auto tSrK = thr_mma.partition_fragment_B(sK);             // (MMA,MMA_N,MMA_K)
    auto tOrVt = thr_mma.partition_fragment_B(sVtNoSwizzle);  // (MMA,MMA_K,MMA_N)

    // shared memory to register copy
    auto smem_tiled_copy_Q = make_tiled_copy_A(SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
    auto tSsQ = smem_thr_copy_Q.partition_S(sQ);
    auto tSrQ_view = smem_thr_copy_Q.retile_D(tSrQ);

    auto smem_tiled_copy_K = make_tiled_copy_B(SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    auto tSsK = smem_thr_copy_K.partition_S(sK);
    auto tSrK_view = smem_thr_copy_K.retile_D(tSrK);

    auto smem_tiled_copy_V = make_tiled_copy_B(SmemCopyAtomTransposed{}, tiled_mma);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    auto tOsVt = smem_thr_copy_V.partition_S(sVt);
    auto tOrVt_view = smem_thr_copy_V.retile_D(tOrVt);
    
    // copy Q
    cute::copy(gmem_tiled_copy_QKV, tQgQ, tQsQ);
    cp_async_fence();
    cp_async_wait<0>();
    __syncthreads();

    // multiply sm scale
    __nv_bfloat162 sm_bf162 = make_bfloat162(__float2bfloat16_rn(sm_scale), __float2bfloat16_rn(sm_scale));
    auto tQsQ_int4 = recast<int4>(tQsQ);
    #pragma unroll
    for(int i=0;i<size(tQsQ_int4);i++)
    {
        auto tmp = tQsQ_int4(i);
        auto tmp_bf162 = (__nv_bfloat162*)&tmp;
        #pragma unroll
        for(int j=0;j<4;j++)
        {
            tmp_bf162[j] = __hmul2(sm_bf162, tmp_bf162[j]);
        }
        tQsQ_int4(i) = tmp;
    }

    // copy KV
    cute::copy(gmem_tiled_copy_QKV, tKgK, tKsK);
    cp_async_fence();
    cute::copy(gmem_tiled_copy_QKV, tVgV, tVsV);
    cp_async_fence();   

    auto rAccOut = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<HeadDim>>{});
    auto scores_max = make_tensor<float>(Shape<Int<2 * size<1>(rAccOut)>>{});  // (2*MMA_M)
    auto scores_sum = make_fragment_like(scores_max);
    auto rAccScore = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
    clear(rAccOut);

    #pragma unroll
    for (int i=0;i<size(scores_max);i++) 
    {
        scores_max(i) = -FLT_MAX;
        scores_sum(i) = 0;
    }

    auto ol = logical_divide(rAccOut.layout(), Shape<Int<2>>{});
    auto rAccOut_new_layout = make_layout(make_layout(get<1>(get<0>(ol)),get<1>(ol)),
                                          make_layout(get<0>(get<0>(ol)),get<2>(ol)));
    auto rAccOut_new = make_tensor(rAccOut.data(), rAccOut_new_layout);

    int n_block_min = 0;
    int n_block_max = cute::ceil_div(k_len, kBlockN);

    #pragma unroll 1
    for(int n_block = n_block_min; n_block < n_block_max; n_block++)
    {
        clear(rAccScore);

        // wait K
        cp_async_wait<1>();
        __syncthreads();

        // S=QK^T
        cute::copy(smem_tiled_copy_Q, tSsQ(_, _, Int<0>{}), tSrQ_view(_, _, Int<0>{}));
        cute::copy(smem_tiled_copy_K, tSsK(_, _, Int<0>{}), tSrK_view(_, _, Int<0>{}));

        #pragma unroll
        for(int j=0;j<size<2>(tSrQ);j++)
        {
            if(j<size<2>(tSrQ)-1)
            {
                cute::copy(smem_tiled_copy_Q, tSsQ(_, _, j+1), tSrQ_view(_, _, j+1));
                cute::copy(smem_tiled_copy_K, tSsK(_, _, j+1), tSrK_view(_, _, j+1));
            }
            cute::gemm(tiled_mma, tSrQ(_, _, j), tSrK(_, _, j), rAccScore);
        }

        auto sl = logical_divide(rAccScore.layout(), Shape<Int<2>>{});
        auto rAccScore_new_layout = make_layout(make_layout(get<1>(get<0>(sl)),get<1>(sl)),
                                                make_layout(get<0>(get<0>(sl)),get<2>(sl)));
        auto scores = make_tensor(rAccScore.data(), rAccScore_new_layout);

        // softmax
        auto scores_max_pre = make_fragment_like(scores_max); // m(j-1)
        cute::copy(scores_max, scores_max_pre);

        #pragma unroll
        for(int i=0;i<size<0>(scores);i++)
        {
            float& scores_max_i = scores_max(i);
            float& scores_sum_i = scores_sum(i);

            // m(j) = max(m(j-1), rowmax(S(j)))
            #pragma unroll
            for(int j=0;j<size<1>(scores);j++)
            {
                scores_max_i = max(scores_max_i, scores(i, j));
            }
            scores_max_i = max(scores_max_i, __shfl_xor_sync(0xffffffff, scores_max_i, 0x2));
            scores_max_i = max(scores_max_i, __shfl_xor_sync(0xffffffff, scores_max_i, 0x1));

            // e^(m(j-1) - m(j))
            float scores_scale = exp2f(scores_max_pre(i) - scores_max_i);

            // key value: diag(e^(m(j-1)-m(j))) * O(j-1)
            #pragma unroll
            for(int j=0;j<size<1>(rAccOut_new);j++)
            {
                rAccOut_new(i, j) *= scores_scale;
            }

            float scores_sum_cur_i = 0;
            #pragma unroll
            for(int j=0;j<size<1>(scores);j++)
            {
                // P(j) = e^(S(j) - m(j))
                scores(i, j) = exp2f(scores(i, j) - scores_max_i);

                // rowsum(P(j))
                scores_sum_cur_i += scores(i, j);
            }
            scores_sum_cur_i += __shfl_xor_sync(0xffffffff, scores_sum_cur_i, 0x2);
            scores_sum_cur_i += __shfl_xor_sync(0xffffffff, scores_sum_cur_i, 0x1);
            
            // key value: l(j) = l(j-1) * e^(m(j-1) - m(j)) + rowsum(P(j))
            scores_sum_i = scores_sum_i * scores_scale + scores_sum_cur_i;
        }
        __syncthreads();

        if(n_block<n_block_max-1)
        {
            gK = local_tile(K, 
                            make_tile(Int<kBlockN>{}, Int<HeadDim>{}),
                            make_coord(n_block+1, _));
            tKgK = gmem_thr_copy_QKV.partition_S(gK(_, _, 0));
            cute::copy(gmem_tiled_copy_QKV, tKgK, tKsK);
        }
        cp_async_fence();

        // wait v
        if (n_block < n_block_max - 1) 
        {
            cp_async_wait<1>();
        } 
        else 
        {
            cp_async_wait<0>();
        }
        __syncthreads();

        auto scores_bf16 = make_tensor_like<__nv_bfloat16>(scores);
        auto scores_fp32x2 = recast<float2>(scores);
        auto scores_bf162 = recast<__nv_bfloat162>(scores_bf16);
        #pragma unroll
        for(int j=0;j<size(scores_bf162);j++)
        {
            scores_bf162(j) = __float22bfloat162_rn(scores_fp32x2(j));
        }

        auto l = logical_divide(scores.layout(), Shape<Underscore, Shape<Underscore, Int<2>>>{});
        auto scores_new_layout =make_layout(make_layout(get<0>(get<1>(l)), get<0>(get<0>(l)),
                                                        get<0>(get<1>(get<1>(l)))),
                                                        get<1>(get<0>(l)), get<1>(get<1>(get<1>(l))));
        auto tOrS = make_tensor(scores_bf16.data(), scores_new_layout);

        cute::copy(smem_tiled_copy_V, tOsVt(_, _, Int<0>{}),tOrVt_view(_, _, Int<0>{}));

        #pragma unroll
        for(int j=0;j<size<2>(tOrS);j++)
        {
            if(j<size<2>(tOrS)-1)
            {
                cute::copy(smem_tiled_copy_V, tOsVt(_, _, j+1), tOrVt_view(_, _, j+1));
            }
            cute::gemm(tiled_mma, tOrS(_, _, j), tOrVt(_, _, j), rAccOut);
        }

        __syncthreads();

        if(n_block<n_block_max-1)
        {
            gV = local_tile(V, 
                            make_tile(Int<kBlockN>{}, Int<HeadDim>{}),
                            make_coord(n_block+1, _));
            tVgV = gmem_thr_copy_QKV.partition_S(gV(_, _, 0));
            cute::copy(gmem_tiled_copy_QKV, tVgV, tVsV);
        }
        cp_async_fence();
    }   

    #pragma unroll
    for (int i = 0; i < size(scores_sum); i++) 
    {
        scores_sum(i) = __frcp_rn(scores_sum(i));
    }
    #pragma unroll
    for (int i = 0; i < size<0>(rAccOut_new); i++) 
    {
        #pragma unroll
        for (int j = 0; j < size<1>(rAccOut_new); j++) 
        {
            rAccOut_new(i, j) *= scores_sum(i);
        }
    }

    auto rAccOut_bf16 = make_tensor_like<__nv_bfloat16>(rAccOut);
    auto rAccOut_fp32x2 = recast<float2>(rAccOut);
    auto rAccOut_bf162 = recast<__nv_bfloat162>(rAccOut_bf16);
    #pragma unroll
    for(int i=0;i<size(rAccOut_bf162);i++)
    {
        rAccOut_bf162(i) = __float22bfloat162_rn(rAccOut_fp32x2(i));
    }

    auto sO = make_tensor(sQ.data(), SmemLayoutO{});
    auto smem_tiled_copy_O = make_tiled_copy_C(SmemCopyAtomO{}, tiled_mma);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
    auto taccOrO = smem_thr_copy_O.retile_S(rAccOut_bf16);
    auto taccOsO = smem_thr_copy_O.partition_D(sO);
    cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);
    
    auto gO = local_tile(O, 
                         make_tile(Int<kBlockM>{}, Int<HeadDim>{}),
                         make_coord(m_block, _));
    GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
    auto tOsO = gmem_thr_copy_O.partition_S(sO);
    auto tOgO = gmem_thr_copy_O.partition_D(gO(_, _, 0));
    
    __syncthreads();
    cute::copy(gmem_tiled_copy_O, tOsO, tOgO);
}

// A*B=C
// B in global, transposed
__device__ void BF16TripleBitmap_MM_Kernel_prapareQKV(
    __nv_bfloat16* smem_C,
    const uint8_t* SignMantissa,
    const __nv_bfloat16* CompressedFull,
    const uint64_t* Bitmap1,
    const uint64_t* Bitmap2,
    const uint64_t* Bitmap3,
    const int* TileOffsets_Median,
    const int* TileOffsets_Global,
    const int max_high_freq_count,
    const int max_full_count,
    const uint8_t start_exp,
    const __nv_bfloat16* B,
    const int M_Global,
    const int N_Global,
    const int K_Global,
    const int head_id,
    const int n_block)
{
    // Tile_M in zipserv-> HeadDim in fa2
    // Tile_N in zipserv-> kBlockM in fa2

    const int x = n_block;                             
    const int y = head_id;
    
    const int NumKBlock = K_Global / TILE_K;

    int NumIter = NumKBlock;

    const int BlockOffset = K_Global / TILE_K * y;
    
    // Calculate shared memory size - add double buffering support
    // extern __shared__ __nv_bfloat16 smem1[];
    
    // B matrix double buffering
    __nv_bfloat16* smem_B = smem_C;
    // A matrix related data double buffering
    const int bitmap_size = 64;
    
    // Bitmap double buffering
    uint64_t* smem_Bitmap1 = reinterpret_cast<uint64_t*>(smem_B + (TILE_K * kBlockM * 2));
    uint64_t* smem_Bitmap2 = smem_Bitmap1 + bitmap_size * 2; // Double buffering
    uint64_t* smem_Bitmap3 = smem_Bitmap2 + bitmap_size * 2; // Double buffering
    
    // Compressed data double buffering
    __nv_bfloat16* smem_FullValues = reinterpret_cast<__nv_bfloat16*>(smem_Bitmap3 + bitmap_size * 2);
    uint8_t* smem_SignMantissa = reinterpret_cast<uint8_t*>(smem_FullValues + max_full_count * 2); // Double buffering
    
    const unsigned int warpId = threadIdx.x / WARP_SIZE;
    const int Tile_Start_M = y * HeadDim;
    const int Tile_Start_Bitmap = y;
    const int Tile_Start_N = x * kBlockM;
    
    int Warp_i = warpId;
    int Warp_j = 0;
    int warp_start_row = 16 * Warp_i;
    int warp_start_col = 0;
    
    // Register allocation
    uint32_t __restrict__ a[2][4]; // double buffering
    uint32_t __restrict__ b[4 * 2][4]; // double buffering
    
    const int WarpOffset = BlockOffset * 4 + Warp_i;
    int global_tile_idx = BlockOffset;
    
    const int* high_freq_start_ptr = TileOffsets_Global + global_tile_idx * 2;
    const int* full_start_ptr = TileOffsets_Global + global_tile_idx * 2 + 1;
    
    int high_freq_start = high_freq_start_ptr[0];
    int full_start = full_start_ptr[0];
    int high_freq_count = high_freq_start_ptr[2] - high_freq_start;
    int full_count = full_start_ptr[2] - full_start;
    
    const __nv_bfloat16* BTileGlobalPTR = B + Tile_Start_N * K_Global;
    
    const uint64_t* Bitmap1TileGlobalPTR = Bitmap1 + Tile_Start_Bitmap * K_Global;
    const uint64_t* Bitmap2TileGlobalPTR = Bitmap2 + Tile_Start_Bitmap * K_Global;
    const uint64_t* Bitmap3TileGlobalPTR = Bitmap3 + Tile_Start_Bitmap * K_Global;
    
    // Initial load into the first double buffer
    CopyTripleBitmapToShared<1>(
        smem_Bitmap1, smem_Bitmap2, smem_Bitmap3,
        Bitmap1TileGlobalPTR, Bitmap2TileGlobalPTR, Bitmap3TileGlobalPTR);
    cp_async_group_commit();
    
    CopyCompressedDataToShared(
        smem_SignMantissa, smem_FullValues,
        SignMantissa + high_freq_start, CompressedFull + full_start,
        high_freq_count, full_count);
    cp_async_group_commit();
    
    CopyTileFromGlobalToShared_X_64_BF16<kBlockM>(
        smem_B, BTileGlobalPTR, K_Global);
    cp_async_group_commit();
    
    // Initialize accumulators
    float c[WARP_ROW_TENSORS_BITMAP_V3 * 4][REG_PER_C_TENSOR_16_16];
    for (int i = 0; i < WARP_ROW_TENSORS_BITMAP_V3 * 4; i++)
        for (int j = 0; j < REG_PER_C_TENSOR_16_16; j++)
            c[i][j] = 0.0f;
    
    cp_async_wait_group<1>();
    
    const int* median_offset_high_warp_ptr = TileOffsets_Median + WarpOffset * 2;
    const int* median_offset_full_warp_ptr = TileOffsets_Median + WarpOffset * 2 + 1;
    
    int next_high_freq_start = high_freq_start_ptr[2];
    int next_full_start = full_start_ptr[2];
    int next_high_freq_count = high_freq_start_ptr[4] - next_high_freq_start;
    int next_full_count = full_start_ptr[4] - next_full_start;
    
    cp_async_wait_group<0>();
    __syncthreads();
    
    // ====== Preload tile 0 ======
    // Fetch current tile offset
    int current_high_freq_start = median_offset_high_warp_ptr[0];
    int current_full_start = median_offset_full_warp_ptr[0];
    
    // Current warp read pointer
    uint64_t* smem_Bitmap1_Warp = smem_Bitmap1 + Warp_i * 16;
    uint64_t* smem_Bitmap2_Warp = smem_Bitmap2 + Warp_i * 16;
    uint64_t* smem_Bitmap3_Warp = smem_Bitmap3 + Warp_i * 16;
    
    // Preload K=0 data into the first buffer
    LoadNextSlice(
        a, b, smem_SignMantissa, smem_FullValues,
        smem_Bitmap1_Warp, smem_Bitmap2_Warp, smem_Bitmap3_Warp,
        start_exp, current_high_freq_start, current_full_start,
        smem_B, warp_start_row, warp_start_col, 0);
    
    #pragma unroll(1)
    for (int tile_id_k = 0; tile_id_k < NumIter; tile_id_k++) {
        high_freq_start = next_high_freq_start;
        full_start = next_full_start;
        high_freq_count = next_high_freq_count;
        full_count = next_full_count;
        
        next_high_freq_start = high_freq_start_ptr[(tile_id_k+2)*2];
        next_full_start = full_start_ptr[(tile_id_k+2)*2];
        next_high_freq_count = high_freq_start_ptr[(tile_id_k+3)*2] - next_high_freq_start;
        next_full_count = full_start_ptr[(tile_id_k+3)*2] - next_full_start;
        
        BTileGlobalPTR = BTileGlobalPTR + TILE_K;
        Bitmap1TileGlobalPTR = Bitmap1TileGlobalPTR + 64;
        Bitmap2TileGlobalPTR = Bitmap2TileGlobalPTR + 64;
        Bitmap3TileGlobalPTR = Bitmap3TileGlobalPTR + 64;
        
        // Compute double-buffer pointers
        __nv_bfloat16* __restrict__ smem_write_B_PTR = smem_B + ((tile_id_k + 1) % 2) * (TILE_K * kBlockM);
        __nv_bfloat16* __restrict__ smem_read_B_PTR = smem_B + ((tile_id_k) % 2) * (TILE_K * kBlockM);
        
        // Double-buffer pointers for A matrix data
        uint64_t* smem_write_Bitmap1 = smem_Bitmap1 + ((tile_id_k + 1) % 2) * bitmap_size;
        uint64_t* smem_write_Bitmap2 = smem_Bitmap2 + ((tile_id_k + 1) % 2) * bitmap_size;
        uint64_t* smem_write_Bitmap3 = smem_Bitmap3 + ((tile_id_k + 1) % 2) * bitmap_size;
        
        uint64_t* smem_read_Bitmap1 = smem_Bitmap1 + ((tile_id_k) % 2) * bitmap_size;
        uint64_t* smem_read_Bitmap2 = smem_Bitmap2 + ((tile_id_k) % 2) * bitmap_size;
        uint64_t* smem_read_Bitmap3 = smem_Bitmap3 + ((tile_id_k) % 2) * bitmap_size;
        
        __nv_bfloat16* smem_write_FullValues = smem_FullValues + ((tile_id_k + 1) % 2) * max_full_count;
        __nv_bfloat16* smem_read_FullValues = smem_FullValues + ((tile_id_k) % 2) * max_full_count;
        
        uint8_t* smem_write_SignMantissa = smem_SignMantissa + ((tile_id_k + 1) % 2) * max_high_freq_count;
        uint8_t* smem_read_SignMantissa = smem_SignMantissa + ((tile_id_k) % 2) * max_high_freq_count;
        
        // Current warp read pointer
        uint64_t* smem_read_Bitmap1_Warp = smem_read_Bitmap1 + Warp_i * 16;
        uint64_t* smem_read_Bitmap2_Warp = smem_read_Bitmap2 + Warp_i * 16;
        uint64_t* smem_read_Bitmap3_Warp = smem_read_Bitmap3 + Warp_i * 16;
        
        bool GlobalCopy = (tile_id_k + 1) < NumIter;
        
        // ====== Launch async load for the next tile =======
        // Load the next tile data into the write buffer
        CopyTripleBitmapToShared<1>(
            smem_write_Bitmap1, smem_write_Bitmap2, smem_write_Bitmap3,
            Bitmap1TileGlobalPTR, Bitmap2TileGlobalPTR, Bitmap3TileGlobalPTR, GlobalCopy);
        cp_async_group_commit();
        
        CopyCompressedDataToShared(
            smem_write_SignMantissa, smem_write_FullValues,
            SignMantissa + high_freq_start, CompressedFull + full_start,
            high_freq_count, full_count, GlobalCopy);
        cp_async_group_commit();
        
        // CopyTileFromGlobalToShared_X_64_BF16<TilingConfig::TILE_N2, TilingConfig>(
        //     smem_write_B_PTR, BTileGlobalPTR, K_Global, N_Global, GlobalCopy);
        CopyTileFromGlobalToShared_X_64_BF16<kBlockM>(
            smem_write_B_PTR, BTileGlobalPTR, K_Global, GlobalCopy);
        cp_async_group_commit();
        
        // ====== Key fix: improved interleaving =======
        
        // // 1. Load tile 1
        // current_high_freq_start = median_offset_high_warp_ptr[tile_id_k * 8];
        // current_full_start = median_offset_full_warp_ptr[tile_id_k * 8];
        
        LoadNextSlice(
            a, b, smem_read_SignMantissa, smem_read_FullValues,
            smem_read_Bitmap1_Warp, smem_read_Bitmap2_Warp, smem_read_Bitmap3_Warp, 
            start_exp, current_high_freq_start, current_full_start,
            smem_read_B_PTR, warp_start_row, warp_start_col, 1);
            
        // 2. Compute tile 0
        SingleMMASlice(c, a, b, 0);
        
        
        LoadNextSlice(
            a, b, smem_read_SignMantissa, smem_read_FullValues,
            smem_read_Bitmap1_Warp, smem_read_Bitmap2_Warp, smem_read_Bitmap3_Warp, 
            start_exp, current_high_freq_start, current_full_start,
            smem_read_B_PTR, warp_start_row, warp_start_col, 2);
        
        // 4. Compute tile 1
        SingleMMASlice(c, a, b, 1);
        
        
        LoadNextSlice(
            a, b, smem_read_SignMantissa, smem_read_FullValues,
            smem_read_Bitmap1_Warp, smem_read_Bitmap2_Warp, smem_read_Bitmap3_Warp, 
            start_exp, current_high_freq_start, current_full_start,
            smem_read_B_PTR, warp_start_row, warp_start_col, 3);
        
        // 6. Compute tile 2
        SingleMMASlice(c, a, b, 2);
        
        // 7. Wait for the next tile load to complete
        cp_async_wait_group<0>();
        __syncthreads();
        
        // 8. Compute tile 3
        SingleMMASlice(c, a, b, 3);
        
        // 9. If another tile exists, load its tile 0 data
        if (GlobalCopy) {
            current_high_freq_start = median_offset_high_warp_ptr[(tile_id_k+1) * 8];
            current_full_start = median_offset_full_warp_ptr[(tile_id_k+1) * 8];
            
            uint64_t* smem_write_Bitmap1_Warp = smem_write_Bitmap1 + Warp_i * 16;
            uint64_t* smem_write_Bitmap2_Warp = smem_write_Bitmap2 + Warp_i * 16;
            uint64_t* smem_write_Bitmap3_Warp = smem_write_Bitmap3 + Warp_i * 16;
            
            LoadNextSlice(
                a, b, smem_write_SignMantissa, smem_write_FullValues,
                smem_write_Bitmap1_Warp, smem_write_Bitmap2_Warp, smem_write_Bitmap3_Warp, 
                start_exp, current_high_freq_start, current_full_start,
                smem_write_B_PTR, warp_start_row, warp_start_col, 0);
        }
    }

    StoreToSharedMemoryFromRegisterBitmapV3_Swizzle(smem_C, c);

    __syncthreads();
}


__global__ void compute_attn_v2_zipserv(
                                void* O_ptr, 
                                const void* K_ptr, const void* V_ptr, 
                                int k_len, int v_len,  int o_len,
                                int row_stride, 
                                float sm_scale,
                                const uint8_t* Q_SignMantissa,
                                const __nv_bfloat16* Q_CompressedFull,
                                const uint64_t* Q_Bitmap1,
                                const uint64_t* Q_Bitmap2,
                                const uint64_t* Q_Bitmap3,
                                const int* Q_TileOffsets_Median,
                                const int* Q_TileOffsets_Global,
                                const int Q_max_high_freq_count,
                                const int Q_max_full_count,
                                const uint8_t Q_start_exp,
                                const int Q_M_Global,
                                const int Q_N_Global,
                                const int Q_K_Global,
                                const __nv_bfloat16* X)
{
    const int m_block = blockIdx.x; // one block process Q_len/kBlockM (AKA N/Br in fa2 paper ) m_blocks 
    const int base_id = blockIdx.y; // it tells us the block process which head(base_id%headnum) of which batch(base_id/headnum) 
    const int tidx = threadIdx.x;    

    if (m_block * kBlockM >= o_len) return;

    extern __shared__ __nv_bfloat16 smem[];
    auto Q_smem_ptr = smem;
    auto K_smem_ptr = Q_smem_ptr + cosize(SmemLayoutQ{});
    auto V_smem_ptr = K_smem_ptr + cosize(SmemLayoutK{});

    auto base_offset = base_id * HeadDim;

    auto K = make_tensor(make_gmem_ptr<__nv_bfloat16>((__nv_bfloat16*)(K_ptr) + base_offset),
                        make_shape(k_len,Int<HeadDim>{}),
                        make_stride(row_stride, _1{}));
    auto V = make_tensor(make_gmem_ptr<__nv_bfloat16>((__nv_bfloat16*)(V_ptr) + base_offset),
                        make_shape(v_len,Int<HeadDim>{}),
                        make_stride(row_stride, _1{}));
    auto O = make_tensor(make_gmem_ptr<__nv_bfloat16>((__nv_bfloat16*)(O_ptr) + base_offset),
                        make_shape(o_len,Int<HeadDim>{}),
                        make_stride(row_stride, _1{}));

    // global memory
    auto gK = local_tile(K,
                         make_tile(Int<kBlockN>{}, Int<HeadDim>{}),
                         make_coord(0, _));
    auto gV = local_tile(V,
                         make_tile(Int<kBlockN>{}, Int<HeadDim>{}),
                         make_coord(0, _));
    
    // shared memory
    auto sQ = make_tensor(make_smem_ptr<__nv_bfloat16>(Q_smem_ptr), SmemLayoutQ{});
    auto sK = make_tensor(make_smem_ptr<__nv_bfloat16>(K_smem_ptr), SmemLayoutK{});
    auto sV = make_tensor(make_smem_ptr<__nv_bfloat16>(V_smem_ptr), SmemLayoutV{});
    auto sVt = make_tensor(make_smem_ptr<__nv_bfloat16>(V_smem_ptr), SmemLayoutVt{});
    auto sVtNoSwizzle = make_tensor(make_smem_ptr<__nv_bfloat16>(V_smem_ptr), SmemLayoutVtNoSwizzle{});

    // global memory to shared memory copy
    GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);
    auto tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    auto tKgK = gmem_thr_copy_QKV.partition_S(gK(_, _, 0));
    auto tKsK = gmem_thr_copy_QKV.partition_D(sK);
    auto tVgV = gmem_thr_copy_QKV.partition_S(gV(_, _, 0));
    auto tVsV = gmem_thr_copy_QKV.partition_D(sV);

    // register
    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    auto tSrQ = thr_mma.partition_fragment_A(sQ);             // (MMA,MMA_M,MMA_K)
    auto tSrK = thr_mma.partition_fragment_B(sK);             // (MMA,MMA_N,MMA_K)
    auto tOrVt = thr_mma.partition_fragment_B(sVtNoSwizzle);  // (MMA,MMA_K,MMA_N)

    // shared memory to register copy
    auto smem_tiled_copy_Q = make_tiled_copy_A(SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
    auto tSsQ = smem_thr_copy_Q.partition_S(sQ);
    auto tSrQ_view = smem_thr_copy_Q.retile_D(tSrQ);

    auto smem_tiled_copy_K = make_tiled_copy_B(SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    auto tSsK = smem_thr_copy_K.partition_S(sK);
    auto tSrK_view = smem_thr_copy_K.retile_D(tSrK);

    auto smem_tiled_copy_V = make_tiled_copy_B(SmemCopyAtomTransposed{}, tiled_mma);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    auto tOsVt = smem_thr_copy_V.partition_S(sVt);
    auto tOrVt_view = smem_thr_copy_V.retile_D(tOrVt);
    
    // copy Q
    // cute::copy(gmem_tiled_copy_QKV, tQgQ, tQsQ);
    BF16TripleBitmap_MM_Kernel_prapareQKV(
        smem, 
        Q_SignMantissa, Q_CompressedFull, Q_Bitmap1, Q_Bitmap2, Q_Bitmap3, 
        Q_TileOffsets_Median, Q_TileOffsets_Global, Q_max_high_freq_count, Q_max_full_count, 
        Q_start_exp,
        X,
        Q_M_Global, Q_N_Global, Q_K_Global,
        base_id,
        m_block
    );
    cp_async_fence();
    cp_async_wait<0>();
    __syncthreads();

    // multiply sm scale
    __nv_bfloat162 sm_bf162 = make_bfloat162(__float2bfloat16_rn(sm_scale), __float2bfloat16_rn(sm_scale));
    auto tQsQ_int4 = recast<int4>(tQsQ);
    #pragma unroll
    for(int i=0;i<size(tQsQ_int4);i++)
    {
        auto tmp = tQsQ_int4(i);
        auto tmp_bf162 = (__nv_bfloat162*)&tmp;
        #pragma unroll
        for(int j=0;j<4;j++)
        {
            tmp_bf162[j] = __hmul2(sm_bf162, tmp_bf162[j]);
        }
        tQsQ_int4(i) = tmp;
    }

    // copy KV
    cute::copy(gmem_tiled_copy_QKV, tKgK, tKsK);
    cp_async_fence();
    cute::copy(gmem_tiled_copy_QKV, tVgV, tVsV);
    cp_async_fence();   

    auto rAccOut = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<HeadDim>>{});
    auto scores_max = make_tensor<float>(Shape<Int<2 * size<1>(rAccOut)>>{});  // (2*MMA_M)
    auto scores_sum = make_fragment_like(scores_max);
    auto rAccScore = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
    clear(rAccOut);

    #pragma unroll
    for (int i=0;i<size(scores_max);i++) 
    {
        scores_max(i) = -FLT_MAX;
        scores_sum(i) = 0;
    }

    auto ol = logical_divide(rAccOut.layout(), Shape<Int<2>>{});
    auto rAccOut_new_layout = make_layout(make_layout(get<1>(get<0>(ol)),get<1>(ol)),
                                          make_layout(get<0>(get<0>(ol)),get<2>(ol)));
    auto rAccOut_new = make_tensor(rAccOut.data(), rAccOut_new_layout);

    int n_block_min = 0;
    int n_block_max = cute::ceil_div(k_len, kBlockN);

    #pragma unroll 1
    for(int n_block = n_block_min; n_block < n_block_max; n_block++)
    {
        clear(rAccScore);

        // wait K
        cp_async_wait<1>();
        __syncthreads();

        // S=QK^T
        cute::copy(smem_tiled_copy_Q, tSsQ(_, _, Int<0>{}), tSrQ_view(_, _, Int<0>{}));
        cute::copy(smem_tiled_copy_K, tSsK(_, _, Int<0>{}), tSrK_view(_, _, Int<0>{}));

        #pragma unroll
        for(int j=0;j<size<2>(tSrQ);j++)
        {
            if(j<size<2>(tSrQ)-1)
            {
                cute::copy(smem_tiled_copy_Q, tSsQ(_, _, j+1), tSrQ_view(_, _, j+1));
                cute::copy(smem_tiled_copy_K, tSsK(_, _, j+1), tSrK_view(_, _, j+1));
            }
            cute::gemm(tiled_mma, tSrQ(_, _, j), tSrK(_, _, j), rAccScore);
        }

        auto sl = logical_divide(rAccScore.layout(), Shape<Int<2>>{});
        auto rAccScore_new_layout = make_layout(make_layout(get<1>(get<0>(sl)),get<1>(sl)),
                                                make_layout(get<0>(get<0>(sl)),get<2>(sl)));
        auto scores = make_tensor(rAccScore.data(), rAccScore_new_layout);

        // softmax
        auto scores_max_pre = make_fragment_like(scores_max); // m(j-1)
        cute::copy(scores_max, scores_max_pre);

        #pragma unroll
        for(int i=0;i<size<0>(scores);i++)
        {
            float& scores_max_i = scores_max(i);
            float& scores_sum_i = scores_sum(i);

            // m(j) = max(m(j-1), rowmax(S(j)))
            #pragma unroll
            for(int j=0;j<size<1>(scores);j++)
            {
                scores_max_i = max(scores_max_i, scores(i, j));
            }
            scores_max_i = max(scores_max_i, __shfl_xor_sync(0xffffffff, scores_max_i, 0x2));
            scores_max_i = max(scores_max_i, __shfl_xor_sync(0xffffffff, scores_max_i, 0x1));

            // e^(m(j-1) - m(j))
            float scores_scale = exp2f(scores_max_pre(i) - scores_max_i);

            // key value: diag(e^(m(j-1)-m(j))) * O(j-1)
            #pragma unroll
            for(int j=0;j<size<1>(rAccOut_new);j++)
            {
                rAccOut_new(i, j) *= scores_scale;
            }

            float scores_sum_cur_i = 0;
            #pragma unroll
            for(int j=0;j<size<1>(scores);j++)
            {
                // P(j) = e^(S(j) - m(j))
                scores(i, j) = exp2f(scores(i, j) - scores_max_i);

                // rowsum(P(j))
                scores_sum_cur_i += scores(i, j);
            }
            scores_sum_cur_i += __shfl_xor_sync(0xffffffff, scores_sum_cur_i, 0x2);
            scores_sum_cur_i += __shfl_xor_sync(0xffffffff, scores_sum_cur_i, 0x1);
            
            // key value: l(j) = l(j-1) * e^(m(j-1) - m(j)) + rowsum(P(j))
            scores_sum_i = scores_sum_i * scores_scale + scores_sum_cur_i;
        }
        __syncthreads();

        if(n_block<n_block_max-1)
        {
            gK = local_tile(K, 
                            make_tile(Int<kBlockN>{}, Int<HeadDim>{}),
                            make_coord(n_block+1, _));
            tKgK = gmem_thr_copy_QKV.partition_S(gK(_, _, 0));
            cute::copy(gmem_tiled_copy_QKV, tKgK, tKsK);
        }
        cp_async_fence();

        // wait v
        if (n_block < n_block_max - 1) 
        {
            cp_async_wait<1>();
        } 
        else 
        {
            cp_async_wait<0>();
        }
        __syncthreads();

        auto scores_bf16 = make_tensor_like<__nv_bfloat16>(scores);
        auto scores_fp32x2 = recast<float2>(scores);
        auto scores_bf162 = recast<__nv_bfloat162>(scores_bf16);
        #pragma unroll
        for(int j=0;j<size(scores_bf162);j++)
        {
            scores_bf162(j) = __float22bfloat162_rn(scores_fp32x2(j));
        }

        auto l = logical_divide(scores.layout(), Shape<Underscore, Shape<Underscore, Int<2>>>{});
        auto scores_new_layout =make_layout(make_layout(get<0>(get<1>(l)), get<0>(get<0>(l)),
                                                        get<0>(get<1>(get<1>(l)))),
                                                        get<1>(get<0>(l)), get<1>(get<1>(get<1>(l))));
        auto tOrS = make_tensor(scores_bf16.data(), scores_new_layout);

        cute::copy(smem_tiled_copy_V, tOsVt(_, _, Int<0>{}),tOrVt_view(_, _, Int<0>{}));

        #pragma unroll
        for(int j=0;j<size<2>(tOrS);j++)
        {
            if(j<size<2>(tOrS)-1)
            {
                cute::copy(smem_tiled_copy_V, tOsVt(_, _, j+1), tOrVt_view(_, _, j+1));
            }
            cute::gemm(tiled_mma, tOrS(_, _, j), tOrVt(_, _, j), rAccOut);
        }

        __syncthreads();

        if(n_block<n_block_max-1)
        {
            gV = local_tile(V, 
                            make_tile(Int<kBlockN>{}, Int<HeadDim>{}),
                            make_coord(n_block+1, _));
            tVgV = gmem_thr_copy_QKV.partition_S(gV(_, _, 0));
            cute::copy(gmem_tiled_copy_QKV, tVgV, tVsV);
        }
        cp_async_fence();
    }   

    #pragma unroll
    for (int i = 0; i < size(scores_sum); i++) 
    {
        scores_sum(i) = __frcp_rn(scores_sum(i));
    }
    #pragma unroll
    for (int i = 0; i < size<0>(rAccOut_new); i++) 
    {
        #pragma unroll
        for (int j = 0; j < size<1>(rAccOut_new); j++) 
        {
            rAccOut_new(i, j) *= scores_sum(i);
        }
    }

    auto rAccOut_bf16 = make_tensor_like<__nv_bfloat16>(rAccOut);
    auto rAccOut_fp32x2 = recast<float2>(rAccOut);
    auto rAccOut_bf162 = recast<__nv_bfloat162>(rAccOut_bf16);
    #pragma unroll
    for(int i=0;i<size(rAccOut_bf162);i++)
    {
        rAccOut_bf162(i) = __float22bfloat162_rn(rAccOut_fp32x2(i));
    }

    auto sO = make_tensor(sQ.data(), SmemLayoutO{});
    auto smem_tiled_copy_O = make_tiled_copy_C(SmemCopyAtomO{}, tiled_mma);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
    auto taccOrO = smem_thr_copy_O.retile_S(rAccOut_bf16);
    auto taccOsO = smem_thr_copy_O.partition_D(sO);
    cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);
    
    auto gO = local_tile(O, 
                         make_tile(Int<kBlockM>{}, Int<HeadDim>{}),
                         make_coord(m_block, _));
    GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
    auto tOsO = gmem_thr_copy_O.partition_S(sO);
    auto tOgO = gmem_thr_copy_O.partition_D(gO(_, _, 0));
    
    __syncthreads();
    cute::copy(gmem_tiled_copy_O, tOsO, tOgO);
}
