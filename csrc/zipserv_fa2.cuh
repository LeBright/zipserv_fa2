#include "Fa2.h"
#include "softmax.h"
#include "L_Kernel.cuh"
#include <cutlass/numeric_conversion.h>


// __forceinline__ __device__ int64_t offset( const int& batch_id, const int64_t& batch_stride)
// {
//     return batch_id * batch_stride;
// }
// template<typename tilecopy, typename engine0, typename layout0, typename engine1, typename layout1>
// __forceinline__ __device__ void copy(const tilecopy& tiled_copy, const Tensor<engine0, layout0>& src,  Tensor<engine1, layout1>& dst) {
//     CUTE_STATIC_ASSERT_V(rank(src) == Int<3>{});
//     CUTE_STATIC_ASSERT_V(rank(dst) == Int<3>{});
//     CUTE_STATIC_ASSERT_V(size<0>(src) == size<0>(dst));                     
//     CUTE_STATIC_ASSERT_V(size<1>(src) == size<1>(dst));                     
//     CUTE_STATIC_ASSERT_V(size<2>(src) == size<2>(dst));                     
//     #pragma unroll
//     for (int m = 0; m < size<1>(src); ++m) 
//     {
//         #pragma unroll
//         for (int k = 0; k < size<2>(src); ++k) 
//         {
//             cute::copy(tiled_copy, src(_, m, k), dst(_, m, k));
//         }
//     }
// }
// template <typename To_type, typename Engine, typename Layout>
// __forceinline__ __device__ auto convert_type(Tensor<Engine, Layout> const &tensor) {
//     using From_type = typename Engine::value_type;
//     constexpr int numel = decltype(size(tensor))::value;
//     cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
//     // HACK: this requires tensor to be "contiguous"
//     auto frag = convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel> *>(tensor.data()));
//     return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
// }
// template<typename MMA_traits, typename Layout>
// __forceinline__ __device__ auto convert_layout_acc_Aregs(Layout acc_layout) {
//     using X = Underscore;
//     static_assert(decltype(size<0>(acc_layout))::value == 4);
//     static_assert(decltype(rank(acc_layout))::value == 3);
//     constexpr int mma_shape_K = get<2>(typename MMA_traits::Shape_MNK{});
//     static_assert(mma_shape_K == 8 || mma_shape_K == 16);
//     if constexpr (mma_shape_K == 8) {
//         return acc_layout;
//     } else {
//         auto l = logical_divide(acc_layout, Shape<X, X, _2>{});  // (4, MMA_M, (2, MMA_N / 2)))
//         return make_layout(make_layout(get<0>(l), get<2, 0>(l)), get<1>(l), get<2, 1>(l));
//     }
// };
// template<typename Tensor0, typename Tensor1,
//          typename Tensor2, typename Tensor3, typename Tensor4,
//          typename TiledMma, typename TiledCopyA, typename TiledCopyB,
//          typename ThrCopyA, typename ThrCopyB>
// __forceinline__ __device__ void gemm(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsA,
//                             Tensor4 const& tCsB, TiledMma tiled_mma,
//                             TiledCopyA smem_tiled_copy_A, TiledCopyB smem_tiled_copy_B,
//                             ThrCopyA smem_thr_copy_A, ThrCopyB smem_thr_copy_B) {
//     CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                     // MMA_M
//     CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));                     // MMA_N
//     CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                     // MMA_K
//     Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
//     CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));            // M
//     Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
//     CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // N
//     cute::copy(smem_tiled_copy_A, tCsA(_, _, _0{}), tCrA_copy_view(_, _, _0{})); 
//     cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{})); 
//     #pragma unroll
//     for (int i = 0; i < size<2>(tCrA); ++i) {
//         if (i < size<2>(tCrA) - 1) {
//             cute::copy(smem_tiled_copy_A, tCsA(_, _, i + 1), tCrA_copy_view(_, _, i + 1)); 
//             cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1)); 
//         }
//         cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
//     }
// }
// template<typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3,
//          typename TiledMma, typename TiledCopy, typename ThrCopy>
// __forceinline__ __device__ void gemm_rs(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsB,
//                                TiledMma tiled_mma, TiledCopy smem_tiled_copy_B,
//                                ThrCopy smem_thr_copy_B) {
//     if (threadIdx.x == 0) 
//     {
//         printf("[DBG] n_block=%d tCsB_after(T0)[0~7]: ", 0);
//         for (int i = 0; i < min(8, (int)size(tCsB)); i++)
//             printf("%.4f ", tCsB(i));
//         printf("\n");
//     }
//     CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                     // MMA_M
//     CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));                     // MMA_N
//     CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                     // MMA_K
//     Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
//     CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // N
//     cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
//     #pragma unroll
//     for (int i = 0; i < size<2>(tCrA); ++i) {
//         if (i < size<2>(tCrA) - 1) {
//             cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
//         }
//         if (threadIdx.x == 0) 
//         {
//             printf("[DBG] n_block=%d tCrB(T0)[0~7]: ", 0);
//             for (int i = 0; i < min(8, (int)size(tCrB_copy_view)); i++)
//                 printf("%.4f ", tCrB_copy_view(i));
//             printf("\n");
//         }
//         cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
//     }
// }


// __global__ void compute_attn(int seqlen_q, int seqlen_kv, int seqlen_o,
//                              int actual_seqlen_q, int actual_seqlen_kv,
//                              __nv_bfloat16* Q_ptr, __nv_bfloat16* K_ptr, __nv_bfloat16* V_ptr, __nv_bfloat16* O_ptr,
//                              float softmax_scale_log2, float scale_softmax) 
// {
//     const int block_id = blockIdx.x;
//     const int batch_id = blockIdx.y;
//     const int head_id = blockIdx.z;

//     // Shared memory.
//     extern __shared__ char smem[];

//     // The thread index.
//     const int tidx = threadIdx.x;

//     // printf("block_id: %d, kBlockM: %d, actual_seqlen_q: %d\n", block_id, kBlockM, actual_seqlen_q);
//     if (block_id * kBlockM >= actual_seqlen_q) return;
//     if (threadIdx.x == 0) {
//         printf("block_id: %d, batch_id: %d, head_id: %d\n", block_id, batch_id, head_id);
//         printf("actual_seqlen_q: %d, actual_seqlen_kv: %d\n", actual_seqlen_q, actual_seqlen_kv);
//         printf("kBlockM: %d, kBlockN: %d\n", kBlockM, kBlockN);
//     }
//     const int n_block_min =  0 ;
//     int n_block_max = cute::ceil_div(actual_seqlen_kv, kBlockN);


//     // ───────────────────────────── QKV batch init ────────────────────────────
//     //
//     // ┌──────────────────────────────────────────────────────────────┐
//     // │  token 0    │ head0[d0..d63] │ head1[d0..d63] │ ... │ head31 │
//     // ├──────────────────────────────────────────────────────────────┤
//     // │  token 1    │ head0[d0..d63] │ head1[d0..d63] │ ... │ head31 │
//     // ├──────────────────────────────────────────────────────────────┤
//     // │  ...                                                         │
//     // ├──────────────────────────────────────────────────────────────┤
//     // │    token actual_seqlen_q-1  ← last valid token               │
//     // ╠══════════════════════════════════════════════════════════════╣  ← actual_seqlen_q
//     // ║  token actual_seqlen_q    ║                                  ║
//     // ║  token ...                ║        padding area              ║  ← garbage data, not accessed
//     // ║  token seqlen_q - 1       ║                                  ║
//     // ╚══════════════════════════════════════════════════════════════╝  ← seqlen_q（padded）
//     auto Q_batch_ptr=make_gmem_ptr(reinterpret_cast<__nv_bfloat16*>(Q_ptr)+offset(batch_id, seqlen_q*HeadNum*HeadDim));
//     Tensor Q_batch = make_tensor(Q_batch_ptr,
//                             make_shape(actual_seqlen_q, HeadNum, HeadDim),
//                             make_stride(HeadNum*HeadDim, HeadDim, _1{}));
//     auto K_batch_ptr = make_gmem_ptr(reinterpret_cast<__nv_bfloat16*>(K_ptr)+offset(batch_id, seqlen_kv*HeadNum*HeadDim));
//     Tensor K_batch = make_tensor(K_batch_ptr,
//                             make_shape(actual_seqlen_kv, HeadNum,HeadDim),
//                             make_stride(HeadDim*HeadNum,HeadDim,_1{}));

//     auto V_batch_ptr = make_gmem_ptr(reinterpret_cast<__nv_bfloat16*>(V_ptr)+offset(batch_id, seqlen_kv*HeadNum*HeadDim));
//     Tensor V_batch = make_tensor(V_batch_ptr,
//                             make_shape(actual_seqlen_kv, HeadNum,HeadDim),
//                             make_stride(HeadDim*HeadNum,HeadDim,_1{}));
//     // ───────────────────────────── QKV batch init ────────────────────────────



//     // ───────────────────────────── QKV global memory init ────────────────────────────
//     //
//     // Q:
//     //                     d0            d63
//     //                     ┌──────────────┐
//     //          token0     │              │  ← block_id=0 
//     //          ...        │▓▓▓▓▓▓▓▓▓▓▓▓▓▓│  ← Q_globalmem (kBlockM=64 , HeadDim=64 )
//     //          token63    │              │
//     //                     ├──────────────┤  ← block_id=1 
//     //          token64    │              │  
//     //          ...        │              │
//     //                     └──────────────┘
//     //
//     //          Q_globalmem shape: (kBlockM=64, HeadDim=64) 
//     //
//     // KV:
//     //                 d0            d63
//     //                 ┌──────────────┐
//     //          kv0    │▓▓▓▓▓▓▓▓▓▓▓▓▓▓│  ← n_block=0  ┐
//     //          ...    │▓▓▓▓▓▓▓▓▓▓▓▓▓▓│               │ kBlockN=64 
//     //          kv63   │▓▓▓▓▓▓▓▓▓▓▓▓▓▓│               ┘
//     //                 ├──────────────┤
//     //          kv64   │▓▓▓▓▓▓▓▓▓▓▓▓▓▓│  ← n_block=1
//     //          ...    │▓▓▓▓▓▓▓▓▓▓▓▓▓▓│
//     //          kv127  │▓▓▓▓▓▓▓▓▓▓▓▓▓▓│
//     //                 ├──────────────┤
//     //          ...    │▓▓▓▓▓▓▓▓▓▓▓▓▓▓│  ← n_block=2
//     //                 └──────────────┘
//     //
//     //          K_globalmem shape: (kBlockN=64, HeadDim=64, nblocksN)
//     //                                                          ↑
//     //                                use K_globalmem(_, _, n_block) get one block in iteration
//     //                                                          ↑ shape: (kBlockN=64, HeadDim=64)
//     Tensor Q_globalmem = local_tile(Q_batch(_, head_id, _),
//                                     Shape<Int<kBlockM>,Int<HeadDim>>{},
//                                     make_coord(block_id,0));
//     Tensor K_globalmem = local_tile(K_batch(_,head_id,_),
//                                     Shape<Int<kBlockN>,Int<HeadDim>>{},
//                                     make_coord(_,0));
//     Tensor V_globalmem = local_tile(V_batch(_,head_id,_),
//                                     Shape<Int<kBlockN>,Int<HeadDim>>{},
//                                     make_coord(_,0));
//     // ───────────────────────────── QKV global memory init ────────────────────────────

    

//     // ─────────────────────────────────── QKV Shared Memory init ──────────────────────────────────
//     //
//     //  ┌─────────────────────────┬──────────────────────────┬──────────────────────────────┐
//     //  │                         │                          │         V_sharedmem          │
//     //  │       Q_sharedmem       │       K_sharedmem        │      V_sharedmem_trans       │
//     //  │                         │                          │  V_sharedmem_trans_noswizzle │
//     //  │                         │                          │                              │
//     //  │                         │                          │        SmemLayoutKV          │
//     //  │       SmemLayoutQ       │       SmemLayoutKV       │     SmemLayoutVtransposed    │
//     //  │                         │                          │SmemLayoutVtransposedNoSwizzle│
//     //  │                         │                          │                              │
//     //  │ (kBlockM=64,HeadDim=64) │  (kBlockN=64,HeadDim=64) │   (kBlockN=64,HeadDim=64)    │
//     //  │                         │                          │   (HeadDim=64,kBlockN=64)    │
//     //  │                         │                          │                              │
//     //  └─────────────────────────┴──────────────────────────┴──────────────────────────────┘
//     //  ↑                         ↑                          ↑
//     //  smem                   +size(Q)                 +size(Q)+size(K)
//     //
//     //  V_sharedmem_trans        → use same memory as V_sharedmem, interpreted with transposed layout (HeadDim,kBlockN)
//     //  V_sharedmem_trans_noswizzle → use same memory as V_sharedmem, interpreted with transposed layout without swizzle
//     Tensor Q_sharedmem = make_tensor(make_smem_ptr(reinterpret_cast<__nv_bfloat16*>(smem)), SmemLayoutQ{});
//     Tensor K_sharedmem = make_tensor(Q_sharedmem.data()+size(Q_sharedmem),SmemLayoutKV{});
//     Tensor V_sharedmem = make_tensor(K_sharedmem.data()+size(K_sharedmem),SmemLayoutKV{});
//     Tensor V_sharedmem_trans = make_tensor(V_sharedmem.data(),SmemLayoutVtransposed{});
//     Tensor V_sharedmem_trans_noswizzle = make_tensor(V_sharedmem.data(), SmemLayoutVtransposedNoSwizzle{});
//     // ─────────────────────────────────── QKV Shared Memory init ──────────────────────────────────



//     // ─────────────────────────────────── tiled copy gmem to smem ──────────────────────────────────
//     //  partition_S / partition_D
//     //
//     //  globalmem_tiled_copy_QKV describe how 512 threads collaborate to move a tile
//     //  get_thread_slice(tidx) cut the slice for the current thread
//     //
//     //  Q:
//     //  gmem Q_globalmem                    smem Q_sharedmem
//     //  ┌──────────────┐                    ┌──────────────┐
//     //  │  .  T0  .    │                    │  .  T0  .    │
//     //  │  .  T1  .    │      ──────►       │  .  T1  .    │
//     //  │  . ...  .    │      cp.async      │  . ...  .    │
//     //  │  . T511 .    │                    │  . T511 .    │
//     //  └──────────────┘                    └──────────────┘
//     //    tiled_Q_src                          tiled_Q_dst
//     //  (which part will be read)           (which part will be written)
//     //
//     //  K/V the same, but K_src/V_src contain the 3rd dimension nblocksN:
//     //  tiled_K_src shape: (CPY, CPY_N, CPY_K, nblocksN)
//     //  In the loop, use tiled_K_src(_, _, _, n_block) to get the current block
//     GmemTiledCopyQKV tiled_cpy_g2s;
//     auto tiled_cpy_g2s_thread = tiled_cpy_g2s.get_thread_slice(tidx); 
//     Tensor tiled_cpy_g2s_thread_Qsrc = tiled_cpy_g2s_thread.partition_S(Q_globalmem);
//     Tensor tiled_cpy_g2s_thread_Qdst = tiled_cpy_g2s_thread.partition_D(Q_sharedmem);
//     Tensor tiled_cpy_g2s_thread_Ksrc = tiled_cpy_g2s_thread.partition_S(K_globalmem);
//     Tensor tiled_cpy_g2s_thread_Kdst = tiled_cpy_g2s_thread.partition_D(K_sharedmem);
//     Tensor tiled_cpy_g2s_thread_Vsrc = tiled_cpy_g2s_thread.partition_S(V_globalmem);
//     Tensor tiled_cpy_g2s_thread_Vdst = tiled_cpy_g2s_thread.partition_D(V_sharedmem);
//     // ─────────────────────────────────── tiled copy gmem to smem ──────────────────────────────────



//     // ─────────────────────────────────── tiled mma ──────────────────────────────────
//     //  partition_fragment
//     //
//     //  TiledMma: 4 warp × 1 × 1，each warp calculate 16 rows
//     //
//     //  Q_reg  shape: (MMA, MMA_M, MMA_K)  ← GEMM1 A operand registers
//     //  K_reg  shape: (MMA, MMA_N, MMA_K)  ← GEMM1 B operand registers
//     //  V_reg  shape: (MMA, MMA_K, MMA_N)  ← GEMM2 B operand registers
//     //  acc_O  shape: (MMA, MMA_M, MMA_K)  ← GEMM2 C accumulators (float)
//     //
//     //  ┌─────────────────────────────────────┐
//     //  │ warp0: acc_O  0~15                  │
//     //  ├─────────────────────────────────────┤
//     //  │ warp1: acc_O  16~31                 │
//     //  ├─────────────────────────────────────┤
//     //  │ warp2: acc_O  32~47                 │
//     //  ├─────────────────────────────────────┤
//     //  │ warp3: acc_O  48~63                 │
//     //  └─────────────────────────────────────┘
//     //
//     TiledMma tiled_mma;
//     auto tiled_mma_thread = tiled_mma.get_thread_slice(tidx);
//     Tensor tiled_mma_thread_Q = tiled_mma_thread.partition_fragment_A(Q_sharedmem);
//     Tensor tiled_mma_thread_K = tiled_mma_thread.partition_fragment_B(K_sharedmem);
//     Tensor tiled_mma_thread_V = tiled_mma_thread.partition_fragment_B(V_sharedmem_trans_noswizzle);
//     Tensor tiled_mma_acc_O = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<HeadDim>>{});
//     // ─────────────────────────────────── tiled mma ──────────────────────────────────



//     // ─────────────────────────────────── tiled copy smem to reg ──────────────────────────────────
//     //  partition_S for smem→reg copy
//     //
//     //  tiled_smem_reg_Q_src: thread read Q_sharedmem use ldmatrix 
//     //  tiled_smem_reg_K_src: thread read K_sharedmem use ldmatrix 
//     //  tiled_smem_reg_V_src: thread read V_sharedmem_trans use ldmatrix.T 
//     //
//     //     Q_sharedmem             reg_Q
//     //  ┌──────────────┐          ┌──────┐
//     //  │▓▓ T0  ▓▓▓    │ ldmatrix │ T0   │
//     //  │▓▓ T1  ▓▓▓    │ ───────► │ T1   │
//     //  │     ...      │          │ ...  │
//     //  └──────────────┘          └──────┘
//     //  tiled_smem_reg_Q_src       reg_Q（use cute::copy）
//     auto tiled_copy_s2r_Q = make_tiled_copy_A(SmemCopyAtom{}, tiled_mma);
//     auto tiled_copy_s2r_thread_Q = tiled_copy_s2r_Q.get_thread_slice(tidx);
//     Tensor tiled_cpy_s2r_Qsrc = tiled_copy_s2r_thread_Q.partition_S(Q_sharedmem);

//     auto tiled_copy_s2r_K = make_tiled_copy_B(SmemCopyAtom{}, tiled_mma);
//     auto tiled_copy_s2r_thread_K = tiled_copy_s2r_K.get_thread_slice(tidx);
//     Tensor tiled_cpy_s2r_Ksrc = tiled_copy_s2r_thread_K.partition_S(K_sharedmem);

//     auto tiled_copy_s2r_V = make_tiled_copy_B(SmemCopyAtomTransposed{}, tiled_mma);
//     auto tiled_copy_s2r_thread_V = tiled_copy_s2r_V.get_thread_slice(tidx);
//     Tensor tiled_cpy_s2r_Vsrc = tiled_copy_s2r_thread_V.partition_S(V_sharedmem_trans);
//     // ─────────────────────────────────── tiled copy smem to reg ──────────────────────────────────

//     // prologue
//     // load QK
//     copy(tiled_cpy_g2s,tiled_cpy_g2s_thread_Qsrc, tiled_cpy_g2s_thread_Qdst);
//     int n_block = n_block_max - 1;
//     copy(tiled_cpy_g2s, tiled_cpy_g2s_thread_Ksrc(_, _, _, n_block), tiled_cpy_g2s_thread_Kdst);
//     cute::cp_async_fence();
//     cp_async_wait<0>();
//     __syncthreads();


//     clear(tiled_mma_acc_O);

//     // Softmax<2 * size<1>(tiled_mma_acc_O)> softmax;

//     for (; n_block >= n_block_min; --n_block) 
//     {

//         Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
//         clear(acc_s);
//         cp_async_wait<0>();
//         __syncthreads();

//         // load V
//         copy(tiled_cpy_g2s, tiled_cpy_g2s_thread_Vsrc(_, _, _, n_block), tiled_cpy_g2s_thread_Vdst);
//         cute::cp_async_fence();

//         // S = Q * K^T
//         gemm(acc_s,                                             // acc
//              tiled_mma_thread_Q, tiled_mma_thread_K,            // reg Q and K
//              tiled_cpy_s2r_Qsrc, tiled_cpy_s2r_Ksrc,            // smem Q and K
//              tiled_mma,                                         // tiled mma
//              tiled_copy_s2r_Q, tiled_copy_s2r_K,                // tiled copy smem to reg for Q and K
//              tiled_copy_s2r_thread_Q, tiled_copy_s2r_thread_K); // thread slice for tiled copy smem to reg for Q and K

    
//         cp_async_wait<0>();
//         __syncthreads();

//         // load next K
//         if (n_block > n_block_min) {
//             copy(tiled_cpy_g2s, tiled_cpy_g2s_thread_Ksrc(_, _, _, n_block - 1), tiled_cpy_g2s_thread_Kdst);
//             cute::cp_async_fence();
//         }

//         // // softmax
//         // if (n_block == n_block_max - 1)
//         // {
//         //     softmax.template softmax_rescale_o<true>(acc_s, tiled_mma_acc_O, softmax_scale_log2);
//         // }
//         // else
//         // {
//         //     softmax.template softmax_rescale_o<false>(acc_s, tiled_mma_acc_O, softmax_scale_log2);
//         // }

//         // if (threadIdx.x == 0) {
//         //     printf("[DBG] n_block=%d acc_s after softmax(T0)[0~7]: ", n_block);
//         //     for (int i = 0; i < min(8, (int)size(acc_s)); i++)
//         //         printf("%.4f ", acc_s(i));
//         //     printf("\n");
//         //     printf("[DBG] n_block=%d row_max(T0)[0~3]: ");
//         //     for (int i = 0; i < min(4, (int)size(softmax.row_max)); i++)
//         //         printf("%.4f ", softmax.row_max(i));
//         //     printf("\n");
//         //     printf("[DBG] n_block=%d row_sum(T0)[0~3]: ");
//         //     for (int i = 0; i < min(4, (int)size(softmax.row_sum)); i++)
//         //         printf("%.4f ", softmax.row_sum(i));
//         //     printf("\n");
//         // }

//         // PV
//         Tensor rP = convert_type<__nv_bfloat16>(acc_s);
//         auto tOrP = make_tensor(rP.data(), convert_layout_acc_Aregs<TiledMma>(rP.layout()));
//         if (threadIdx.x == 0) {
//             printf("[DBG] n_block=%d acc_s after softmax(T0)[0~7]: ", n_block);
//             for (int i = 0; i < min(8, (int)size(acc_s)); i++)
//                 printf("%.4f ", acc_s(i));
//             printf("\n");
//         }
//         cp_async_wait<0>();
//         __syncthreads();
//         gemm_rs(tiled_mma_acc_O,          
//                 tOrP,                     
//                 tiled_mma_thread_V,       
//                 tiled_cpy_s2r_Vsrc,       
//                 tiled_mma,                
//                 tiled_copy_s2r_V,         
//                 tiled_copy_s2r_thread_V); 



//         if (threadIdx.x == 0) {
//             printf("[DBG] n_block=%d acc_O(T0)[0~7]: ", n_block);
//             for (int i = 0; i < min(8, (int)size(tiled_mma_acc_O)); i++)
//                 printf("%.4f ", tiled_mma_acc_O(i));
//             printf("\n");
//         }
//     }

//     // Epilogue

//     // softmax.template normalize_softmax_lse(tiled_mma_acc_O, scale_softmax);


//     if (threadIdx.x == 0) {
//         printf("[DBG] acc_O after normalize(T0)[0~7]: ");
//         for (int i = 0; i < min(8, (int)size(tiled_mma_acc_O)); i++)
//             printf("%.4f ", tiled_mma_acc_O(i));
//         printf("\n");
//     }

//     // Convert acc_o from fp32 to fp16/bf16
//     Tensor Oreg = convert_type<__nv_bfloat16>(tiled_mma_acc_O);

//     if (threadIdx.x == 0) {
//         printf("[DBG] Oreg after convert_type(T0)[0~7]: ");
//         for (int i = 0; i < min(8, (int)size(Oreg)); i++)
//             printf("%.4f ", __bfloat162float(Oreg(i)));
//         printf("\n");
//     }
//     Tensor Osmem = make_tensor(Q_sharedmem.data(), SmemLayoutO{});    // (SMEM_M,SMEM_N)

//     auto smem_tiled_copy_O = make_tiled_copy_C(SmemCopyAtomO{}, tiled_mma);
//     auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
//     Tensor taccOrO = smem_thr_copy_O.retile_S(Oreg);        // ((Atom,AtomNum), MMA_M, MMA_N)
//     Tensor taccOsO = smem_thr_copy_O.partition_D(Osmem);     // ((Atom,AtomNum),PIPE_M,PIPE_N)


//     cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

//     Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<__nv_bfloat16*>(O_ptr)
//                                           + offset(batch_id, seqlen_o*HeadNum*HeadDim)),
//                             make_shape(actual_seqlen_q, HeadNum, HeadDim),
//                             make_stride(HeadNum*HeadDim, HeadDim, _1{}));
//     Tensor gO = local_tile(mO(_, head_id, _), Shape<Int<kBlockM>, Int<HeadDim>>{},
//                            make_coord(block_id, 0));  // (kBlockM, kHeadDim)

//     GmemTiledCopyO gmem_tiled_copy_O;
//     auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
//     Tensor tOsO = gmem_thr_copy_O.partition_S(Osmem);        // ((Atom,AtomNum),ATOM_M,ATOM_N)
//     Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

//     __syncthreads();

//     Tensor tOrO = make_tensor<__nv_bfloat16>(shape(tOgO));
//     cute::copy(gmem_tiled_copy_O, tOsO, tOrO);  // smem → reg

//     copy(gmem_tiled_copy_O, tOrO, tOgO);  // reg → gmem
// }

__global__ void compute_attn_v2(void* O_ptr, 
                                const void* Q_ptr, const void* K_ptr, const void* V_ptr, 
                                int q_len, int k_len, int v_len,  int o_len,
                                int head_stride, 
                                float sm_scale)
{
    const int m_block = blockIdx.x; // one block process Q_len/kBlockM (AKA N/Br in fa2 paper ) m_blocks 
    const int base_id = blockIdx.y; // it tells us the block process which head(base_id%headnum) of which batch(base_id/headnum) 
    const int tidx = threadIdx.x;    

    extern __shared__ __nv_bfloat16 smem[];
    auto Q_smem_ptr = smem;
    auto K_smem_ptr = Q_smem_ptr + cosize(SmemLayoutQ{});
    auto V_smem_ptr = K_smem_ptr + cosize(SmemLayoutK{});

    auto base_offset = base_id * head_stride;

    auto Q = make_tensor(make_gmem_ptr<__nv_bfloat16>(Q_ptr + base_offset),
                         make_shape(q_len,Int<HeadDim>{}),
                         make_stride(Int<HeadDim>{}, _1{}));
    auto K = make_tensor(make_gmem_ptr<__nv_bfloat16>(K_ptr + base_offset),
                        make_shape(k_len,Int<HeadDim>{}),
                        make_stride(Int<HeadDim>{}, _1{}));
    auto V = make_tensor(make_gmem_ptr<__nv_bfloat16>(V_ptr + base_offset),
                        make_shape(v_len,Int<HeadDim>{}),
                        make_stride(Int<HeadDim>{}, _1{}));
    auto O = make_tensor(make_gmem_ptr<__nv_bfloat16>(O_ptr + base_offset),
                        make_shape(o_len,Int<HeadDim>{}),
                        make_stride(Int<HeadDim>{}, _1{}));

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
    for (int i=0;i<size(rAccScore);i++) 
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
            float scores_max_i = scores_max(i);
            float scores_sum_i = scores_sum(i);

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
        cp_async_wait<1>();
        __syncthreads();

        auto scores_bf16 = make_tensor_like<__nv_bfloat16>(scores);
        auto scores_fp32x2 = recast<float2>(scores);
        auto scores_bf162 = make_tensor_like<__nv_bfloat162>(scores);
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
    auto taccOrO = smem_thr_copy_O.partition_S(rAccOut_bf16);
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
    __nv_bfloat16* smem_CFrag,
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
    const int y = head_id % (M_Global / HeadDim);
    
    const int NumKBlock = K_Global / TILE_K;

    int NumIter = 0;

    NumIter = NumKBlock;

    const int BlockOffset = K_Global / TILE_K * y;
    
    // Calculate shared memory size - add double buffering support
    // extern __shared__ __nv_bfloat16 smem1[];
    
    // B matrix double buffering
    __nv_bfloat16* smem_B = smem_CFrag;
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

    StoreToSharedMemoryFromRegisterBitmapV3_Swizzle(smem_CFrag, c);

    __syncthreads();
}

// __global__ void compute_atten_zipserv(int seqlen_q, int seqlen_kv, int seqlen_o,
//                              int actual_seqlen_q, int actual_seqlen_kv,
//                              __nv_bfloat16* Q_ptr, __nv_bfloat16* K_ptr, __nv_bfloat16* V_ptr, __nv_bfloat16* O_ptr,
//                              float softmax_scale_log2, float scale_softmax,
//                             const __nv_bfloat16* X,
//                             const uint8_t* SignMantissa_Wq,
//                             const __nv_bfloat16* CompressedFull_Wq,
//                             const uint64_t* Bitmap1_Wq,
//                             const uint64_t* Bitmap2_Wq,
//                             const uint64_t* Bitmap3_Wq,
//                             const int* TileOffsets_Median_Wq,
//                             const int* TileOffsets_Global_Wq,
//                             const int max_high_freq_count_Wq,
//                             const int max_full_count_Wq,
//                             const uint8_t start_exp_Wq,
//                             const int M_Global_Wq,
//                             const int N_Global_Wq,
//                             const int K_Global_Wq,
//                             const uint8_t* SignMantissa_Wk,
//                             const __nv_bfloat16* CompressedFull_Wk,
//                             const uint64_t* Bitmap1_Wk,
//                             const uint64_t* Bitmap2_Wk,
//                             const uint64_t* Bitmap3_Wk,
//                             const int* TileOffsets_Median_Wk,
//                             const int* TileOffsets_Global_Wk,
//                             const int max_high_freq_count_Wk,
//                             const int max_full_count_Wk,
//                             const uint8_t start_exp_Wk,
//                             const int M_Global_Wk,
//                             const int N_Global_Wk,
//                             const int K_Global_Wk)
// {
//     const int block_id = blockIdx.x;
//     const int batch_id = blockIdx.y;
//     const int head_id = blockIdx.z;

//     // Shared memory.
//     extern __shared__ char smem[];

//     // The thread index.
//     const int tidx = threadIdx.x;

//     // printf("block_id: %d, kBlockM: %d, actual_seqlen_q: %d\n", block_id, kBlockM, actual_seqlen_q);
//     if (block_id * kBlockM >= actual_seqlen_q) return;

//     const int n_block_min =  0 ;
//     int n_block_max = cute::ceil_div(actual_seqlen_kv, kBlockN);



//     // auto Q_batch_ptr=make_gmem_ptr(reinterpret_cast<__nv_bfloat16*>(Q_ptr)+offset(batch_id, seqlen_q*HeadNum*HeadDim));
//     // Tensor Q_batch = make_tensor(Q_batch_ptr,
//     //                         make_shape(actual_seqlen_q, HeadNum, HeadDim),
//     //                         make_stride(HeadNum*HeadDim, HeadDim, _1{}));
//     // auto K_batch_ptr = make_gmem_ptr(reinterpret_cast<__nv_bfloat16*>(K_ptr)+offset(batch_id, seqlen_kv*HeadNum*HeadDim));
//     // Tensor K_batch = make_tensor(K_batch_ptr,
//     //                         make_shape(actual_seqlen_kv, HeadNum,HeadDim),
//     //                         make_stride(HeadDim*HeadNum,HeadDim,_1{}));

//     auto V_batch_ptr = make_gmem_ptr(reinterpret_cast<__nv_bfloat16*>(V_ptr)+offset(batch_id, seqlen_kv*HeadNum*HeadDim));
//     Tensor V_batch = make_tensor(V_batch_ptr,
//                             make_shape(actual_seqlen_kv, HeadNum,HeadDim),
//                             make_stride(HeadDim*HeadNum,HeadDim,_1{}));

//     // Tensor Q_globalmem = local_tile(Q_batch(_, head_id, _),
//     //                                 Shape<Int<kBlockM>,Int<HeadDim>>{},
//     //                                 make_coord(block_id,0));
//     // Tensor K_globalmem = local_tile(K_batch(_,head_id,_),
//     //                                 Shape<Int<kBlockN>,Int<HeadDim>>{},
//     //                                 make_coord(_,0));
//     Tensor V_globalmem = local_tile(V_batch(_,head_id,_),
//                                     Shape<Int<kBlockN>,Int<HeadDim>>{},
//                                     make_coord(_,0));
//     int n_block = n_block_max - 1;

//     // prepare Q
//     BF16TripleBitmap_MM_Kernel_prapareQKV(reinterpret_cast<__nv_bfloat16*>(smem),
//                                     SignMantissa_Wq,
//                                     CompressedFull_Wq,
//                                     Bitmap1_Wq,
//                                     Bitmap2_Wq,
//                                     Bitmap3_Wq,
//                                     TileOffsets_Median_Wq,
//                                     TileOffsets_Global_Wq,
//                                     max_high_freq_count_Wq,
//                                     max_full_count_Wq,
//                                     start_exp_Wq,
//                                     X,
//                                     M_Global_Wq,
//                                     N_Global_Wq,
//                                     K_Global_Wq,
//                                     head_id,
//                                     n_block);

//     Tensor Q_sharedmem = make_tensor(make_smem_ptr(reinterpret_cast<__nv_bfloat16*>(smem)), SmemLayoutQ{});
//     Tensor K_sharedmem = make_tensor(Q_sharedmem.data()+size(Q_sharedmem),SmemLayoutKV{});
//     Tensor V_sharedmem = make_tensor(K_sharedmem.data()+size(K_sharedmem),SmemLayoutKV{});
//     Tensor V_sharedmem_trans = make_tensor(V_sharedmem.data(),SmemLayoutVtransposed{});
//     Tensor V_sharedmem_trans_noswizzle = make_tensor(V_sharedmem.data(), SmemLayoutVtransposedNoSwizzle{});
//     // prepare K
//     BF16TripleBitmap_MM_Kernel_prapareQKV(reinterpret_cast<__nv_bfloat16*>(smem+size(Q_sharedmem)),
//                                     SignMantissa_Wk,
//                                     CompressedFull_Wk,
//                                     Bitmap1_Wk,
//                                     Bitmap2_Wk,
//                                     Bitmap3_Wk,
//                                     TileOffsets_Median_Wk,
//                                     TileOffsets_Global_Wk,
//                                     max_high_freq_count_Wk,
//                                     max_full_count_Wk,
//                                     start_exp_Wk,
//                                     X,
//                                     M_Global_Wk,
//                                     N_Global_Wk,
//                                     K_Global_Wk,
//                                     head_id,
//                                     n_block);
//     GmemTiledCopyQKV tiled_cpy_g2s;
//     auto tiled_cpy_g2s_thread = tiled_cpy_g2s.get_thread_slice(tidx); 
//     // Tensor tiled_cpy_g2s_thread_Qsrc = tiled_cpy_g2s_thread.partition_S(Q_globalmem);
//     // Tensor tiled_cpy_g2s_thread_Qdst = tiled_cpy_g2s_thread.partition_D(Q_sharedmem);
//     // Tensor tiled_cpy_g2s_thread_Ksrc = tiled_cpy_g2s_thread.partition_S(K_globalmem);
//     // Tensor tiled_cpy_g2s_thread_Kdst = tiled_cpy_g2s_thread.partition_D(K_sharedmem);
//     Tensor tiled_cpy_g2s_thread_Vsrc = tiled_cpy_g2s_thread.partition_S(V_globalmem);
//     Tensor tiled_cpy_g2s_thread_Vdst = tiled_cpy_g2s_thread.partition_D(V_sharedmem);

//     TiledMma tiled_mma;
//     auto tiled_mma_thread = tiled_mma.get_thread_slice(tidx);
//     Tensor tiled_mma_thread_Q = tiled_mma_thread.partition_fragment_A(Q_sharedmem);
//     Tensor tiled_mma_thread_K = tiled_mma_thread.partition_fragment_B(K_sharedmem);
//     Tensor tiled_mma_thread_V = tiled_mma_thread.partition_fragment_B(V_sharedmem_trans_noswizzle);
//     Tensor tiled_mma_acc_O = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<HeadDim>>{});

//     auto tiled_copy_s2r_Q = make_tiled_copy_A(SmemCopyAtom{}, tiled_mma);
//     auto tiled_copy_s2r_thread_Q = tiled_copy_s2r_Q.get_thread_slice(tidx);
//     Tensor tiled_cpy_s2r_Qsrc = tiled_copy_s2r_thread_Q.partition_S(Q_sharedmem);

//     auto tiled_copy_s2r_K = make_tiled_copy_B(SmemCopyAtom{}, tiled_mma);
//     auto tiled_copy_s2r_thread_K = tiled_copy_s2r_K.get_thread_slice(tidx);
//     Tensor tiled_cpy_s2r_Ksrc = tiled_copy_s2r_thread_K.partition_S(K_sharedmem);

//     auto tiled_copy_s2r_V = make_tiled_copy_B(SmemCopyAtomTransposed{}, tiled_mma);
//     auto tiled_copy_s2r_thread_V = tiled_copy_s2r_V.get_thread_slice(tidx);
//     Tensor tiled_cpy_s2r_Vsrc = tiled_copy_s2r_thread_V.partition_S(V_sharedmem_trans);


//     cute::cp_async_fence();

//     clear(tiled_mma_acc_O);

//     Softmax<2 * size<1>(tiled_mma_acc_O)> softmax;

//     for (; n_block >= n_block_min; --n_block) 
//     {

//         Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
//         clear(acc_s);
//         cp_async_wait<0>();
//         __syncthreads();

//         // load V
//         copy(tiled_cpy_g2s, tiled_cpy_g2s_thread_Vsrc(_, _, _, n_block), tiled_cpy_g2s_thread_Vdst);
//         cute::cp_async_fence();

//         // S = Q * K^T
//         gemm(acc_s,                                             // acc
//              tiled_mma_thread_Q, tiled_mma_thread_K,            // reg Q and K
//              tiled_cpy_s2r_Qsrc, tiled_cpy_s2r_Ksrc,            // smem Q and K
//              tiled_mma,                                         // tiled mma
//              tiled_copy_s2r_Q, tiled_copy_s2r_K,                // tiled copy smem to reg for Q and K
//              tiled_copy_s2r_thread_Q, tiled_copy_s2r_thread_K); // thread slice for tiled copy smem to reg for Q and K

//         cp_async_wait<0>();
//         __syncthreads();

//         // load next K
//         if (n_block > n_block_min) {
//             // copy(tiled_cpy_g2s, tiled_cpy_g2s_thread_Ksrc(_, _, _, n_block - 1), tiled_cpy_g2s_thread_Kdst);
//             BF16TripleBitmap_MM_Kernel_prapareQKV(reinterpret_cast<__nv_bfloat16*>(smem+size(Q_sharedmem)),
//                                 SignMantissa_Wk,
//                                 CompressedFull_Wk,
//                                 Bitmap1_Wk,
//                                 Bitmap2_Wk,
//                                 Bitmap3_Wk,
//                                 TileOffsets_Median_Wk,
//                                 TileOffsets_Global_Wk,
//                                 max_high_freq_count_Wk,
//                                 max_full_count_Wk,
//                                 start_exp_Wk,
//                                 X,
//                                 M_Global_Wk,
//                                 N_Global_Wk,
//                                 K_Global_Wk,
//                                 head_id,
//                                 n_block-1);
//             cute::cp_async_fence();
//         }

//         // softmax
//         if (n_block == n_block_max - 1)
//         {
//             softmax.template softmax_rescale_o<true>(acc_s, tiled_mma_acc_O, softmax_scale_log2);
//         }
//         else
//         {
//             softmax.template softmax_rescale_o<false>(acc_s, tiled_mma_acc_O, softmax_scale_log2);
//         }

//         // PV
//         Tensor rP = convert_type<__nv_bfloat16>(acc_s);
//         auto tOrP = make_tensor(rP.data(), convert_layout_acc_Aregs<TiledMma>(rP.layout()));
//         gemm_rs(tiled_mma_acc_O,          // acc
//                 tOrP,                     // reg P
//                 tiled_mma_thread_V,       // reg V
//                 tiled_cpy_s2r_Vsrc,       // smem V
//                 tiled_mma,                // tiled mma
//                 tiled_copy_s2r_V,         // tiled copy smem to reg for V
//                 tiled_copy_s2r_thread_V); // thread slice for tiled copy smem to reg for V
//     }

//     // Epilogue

//     softmax.template normalize_softmax_lse(tiled_mma_acc_O, scale_softmax);

//     // Convert acc_o from fp32 to fp16/bf16
//     Tensor Oreg = convert_type<__nv_bfloat16>(tiled_mma_acc_O);
//     Tensor Osmem = make_tensor(Q_sharedmem.data(), SmemLayoutO{});    // (SMEM_M,SMEM_N)

//     auto smem_tiled_copy_O = make_tiled_copy_C(SmemCopyAtomO{}, tiled_mma);
//     auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
//     Tensor taccOrO = smem_thr_copy_O.retile_S(Oreg);        // ((Atom,AtomNum), MMA_M, MMA_N)
//     Tensor taccOsO = smem_thr_copy_O.partition_D(Osmem);     // ((Atom,AtomNum),PIPE_M,PIPE_N)


//     cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

//     Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<__nv_bfloat16*>(O_ptr)
//                                           + offset(batch_id, seqlen_o*HeadNum*HeadDim)),
//                             make_shape(actual_seqlen_q, HeadNum, HeadDim),
//                             make_stride(HeadNum*HeadDim, HeadDim, _1{}));
//     Tensor gO = local_tile(mO(_, head_id, _), Shape<Int<kBlockM>, Int<HeadDim>>{},
//                            make_coord(block_id, 0));  // (kBlockM, kHeadDim)

//     GmemTiledCopyO gmem_tiled_copy_O;
//     auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
//     Tensor tOsO = gmem_thr_copy_O.partition_S(Osmem);        // ((Atom,AtomNum),ATOM_M,ATOM_N)
//     Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

//     __syncthreads();

//     Tensor tOrO = make_tensor<__nv_bfloat16>(shape(tOgO));
//     cute::copy(gmem_tiled_copy_O, tOsO, tOrO);

//     copy(gmem_tiled_copy_O, tOrO, tOgO);
// }
