#include "Fa2.h"
#include "softmax.h"
#include "L_Kernel.cuh"
#include <cutlass/numeric_conversion.h>


__forceinline__ __device__ int64_t offset( const int& batch_id, const int64_t& batch_stride)
{
    return batch_id * batch_stride;
}
template<typename tilecopy, typename engine0, typename layout0, typename engine1, typename layout1>
__forceinline__ __device__ void copy(const tilecopy& tiled_copy, const Tensor<engine0, layout0>& src,  Tensor<engine1, layout1>& dst) {
    CUTE_STATIC_ASSERT_V(rank(src) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(dst) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(src) == size<0>(dst));                     
    CUTE_STATIC_ASSERT_V(size<1>(src) == size<1>(dst));                     
    CUTE_STATIC_ASSERT_V(size<2>(src) == size<2>(dst));                     
    #pragma unroll
    for (int m = 0; m < size<1>(src); ++m) 
    {
        #pragma unroll
        for (int k = 0; k < size<2>(src); ++k) 
        {
            cute::copy(tiled_copy, src(_, m, k), dst(_, m, k));
        }
    }
}
template <typename To_type, typename Engine, typename Layout>
__forceinline__ __device__ auto convert_type(Tensor<Engine, Layout> const &tensor) {
    using From_type = typename Engine::value_type;
    constexpr int numel = decltype(size(tensor))::value;
    cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
    // HACK: this requires tensor to be "contiguous"
    auto frag = convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel> *>(tensor.data()));
    return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
}
template<typename MMA_traits, typename Layout>
__forceinline__ __device__ auto convert_layout_acc_Aregs(Layout acc_layout) {
    using X = Underscore;
    static_assert(decltype(size<0>(acc_layout))::value == 4);
    static_assert(decltype(rank(acc_layout))::value == 3);
    constexpr int mma_shape_K = get<2>(typename MMA_traits::Shape_MNK{});
    static_assert(mma_shape_K == 8 || mma_shape_K == 16);
    if constexpr (mma_shape_K == 8) {
        return acc_layout;
    } else {
        auto l = logical_divide(acc_layout, Shape<X, X, _2>{});  // (4, MMA_M, (2, MMA_N / 2)))
        return make_layout(make_layout(get<0>(l), get<2, 0>(l)), get<1>(l), get<2, 1>(l));
    }
};
template<typename Tensor0, typename Tensor1,
         typename Tensor2, typename Tensor3, typename Tensor4,
         typename TiledMma, typename TiledCopyA, typename TiledCopyB,
         typename ThrCopyA, typename ThrCopyB>
__forceinline__ __device__ void gemm(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsA,
                            Tensor4 const& tCsB, TiledMma tiled_mma,
                            TiledCopyA smem_tiled_copy_A, TiledCopyB smem_tiled_copy_B,
                            ThrCopyA smem_thr_copy_A, ThrCopyB smem_thr_copy_B) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                     // MMA_K
    Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));            // M
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // N
    cute::copy(smem_tiled_copy_A, tCsA(_, _, _0{}), tCrA_copy_view(_, _, _0{})); 
    cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{})); 
    #pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            cute::copy(smem_tiled_copy_A, tCsA(_, _, i + 1), tCrA_copy_view(_, _, i + 1)); 
            cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1)); 
        }
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
    }
}
template<typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3,
         typename TiledMma, typename TiledCopy, typename ThrCopy>
__forceinline__ __device__ void gemm_rs(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsB,
                               TiledMma tiled_mma, TiledCopy smem_tiled_copy_B,
                               ThrCopy smem_thr_copy_B) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                     // MMA_K
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // N
    cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
    #pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
        }
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
    }
}


__global__ void compute_attn(int seqlen_q, int seqlen_kv, int seqlen_o,
                             int actual_seqlen_q, int actual_seqlen_kv,
                             __nv_bfloat16* Q_ptr, __nv_bfloat16* K_ptr, __nv_bfloat16* V_ptr, __nv_bfloat16* O_ptr,
                             float softmax_scale_log2, float scale_softmax) 
{
    const int block_id = blockIdx.x;
    const int batch_id = blockIdx.y;
    const int head_id = blockIdx.z;

    // Shared memory.
    extern __shared__ char smem[];

    // The thread index.
    const int tidx = threadIdx.x;

    // printf("block_id: %d, kBlockM: %d, actual_seqlen_q: %d\n", block_id, kBlockM, actual_seqlen_q);
    if (block_id * kBlockM >= actual_seqlen_q) return;
    if (threadIdx.x == 0) {
        printf("block_id: %d, batch_id: %d, head_id: %d\n", block_id, batch_id, head_id);
        printf("actual_seqlen_q: %d, actual_seqlen_kv: %d\n", actual_seqlen_q, actual_seqlen_kv);
        printf("kBlockM: %d, kBlockN: %d\n", kBlockM, kBlockN);
    }
    const int n_block_min =  0 ;
    int n_block_max = cute::ceil_div(actual_seqlen_kv, kBlockN);


    // ───────────────────────────── QKV batch init ────────────────────────────
    //
    // ┌──────────────────────────────────────────────────────────────┐
    // │  token 0    │ head0[d0..d63] │ head1[d0..d63] │ ... │ head31 │
    // ├──────────────────────────────────────────────────────────────┤
    // │  token 1    │ head0[d0..d63] │ head1[d0..d63] │ ... │ head31 │
    // ├──────────────────────────────────────────────────────────────┤
    // │  ...                                                         │
    // ├──────────────────────────────────────────────────────────────┤
    // │    token actual_seqlen_q-1  ← last valid token               │
    // ╠══════════════════════════════════════════════════════════════╣  ← actual_seqlen_q
    // ║  token actual_seqlen_q    ║                                  ║
    // ║  token ...                ║        padding area              ║  ← garbage data, not accessed
    // ║  token seqlen_q - 1       ║                                  ║
    // ╚══════════════════════════════════════════════════════════════╝  ← seqlen_q（padded）
    auto Q_batch_ptr=make_gmem_ptr(reinterpret_cast<__nv_bfloat16*>(Q_ptr)+offset(batch_id, seqlen_q*HeadNum*HeadDim));
    Tensor Q_batch = make_tensor(Q_batch_ptr,
                            make_shape(actual_seqlen_q, HeadNum, HeadDim),
                            make_stride(HeadNum*HeadDim, HeadDim, _1{}));
    auto K_batch_ptr = make_gmem_ptr(reinterpret_cast<__nv_bfloat16*>(K_ptr)+offset(batch_id, seqlen_kv*HeadNum*HeadDim));
    Tensor K_batch = make_tensor(K_batch_ptr,
                            make_shape(actual_seqlen_kv, HeadNum,HeadDim),
                            make_stride(HeadDim*HeadNum,HeadDim,_1{}));

    auto V_batch_ptr = make_gmem_ptr(reinterpret_cast<__nv_bfloat16*>(V_ptr)+offset(batch_id, seqlen_kv*HeadNum*HeadDim));
    Tensor V_batch = make_tensor(V_batch_ptr,
                            make_shape(actual_seqlen_kv, HeadNum,HeadDim),
                            make_stride(HeadDim*HeadNum,HeadDim,_1{}));
    // ───────────────────────────── QKV batch init ────────────────────────────



    // ───────────────────────────── QKV global memory init ────────────────────────────
    //
    // Q:
    //                     d0            d63
    //                     ┌──────────────┐
    //          token0     │              │  ← block_id=0 
    //          ...        │▓▓▓▓▓▓▓▓▓▓▓▓▓▓│  ← Q_globalmem (kBlockM=64 , HeadDim=64 )
    //          token63    │              │
    //                     ├──────────────┤  ← block_id=1 
    //          token64    │              │  
    //          ...        │              │
    //                     └──────────────┘
    //
    //          Q_globalmem shape: (kBlockM=64, HeadDim=64) 
    //
    // KV:
    //                 d0            d63
    //                 ┌──────────────┐
    //          kv0    │▓▓▓▓▓▓▓▓▓▓▓▓▓▓│  ← n_block=0  ┐
    //          ...    │▓▓▓▓▓▓▓▓▓▓▓▓▓▓│               │ kBlockN=64 
    //          kv63   │▓▓▓▓▓▓▓▓▓▓▓▓▓▓│               ┘
    //                 ├──────────────┤
    //          kv64   │▓▓▓▓▓▓▓▓▓▓▓▓▓▓│  ← n_block=1
    //          ...    │▓▓▓▓▓▓▓▓▓▓▓▓▓▓│
    //          kv127  │▓▓▓▓▓▓▓▓▓▓▓▓▓▓│
    //                 ├──────────────┤
    //          ...    │▓▓▓▓▓▓▓▓▓▓▓▓▓▓│  ← n_block=2
    //                 └──────────────┘
    //
    //          K_globalmem shape: (kBlockN=64, HeadDim=64, nblocksN)
    //                                                          ↑
    //                                use K_globalmem(_, _, n_block) get one block in iteration
    //                                                          ↑ shape: (kBlockN=64, HeadDim=64)
    Tensor Q_globalmem = local_tile(Q_batch(_, head_id, _),
                                    Shape<Int<kBlockM>,Int<HeadDim>>{},
                                    make_coord(block_id,0));
    Tensor K_globalmem = local_tile(K_batch(_,head_id,_),
                                    Shape<Int<kBlockN>,Int<HeadDim>>{},
                                    make_coord(_,0));
    Tensor V_globalmem = local_tile(V_batch(_,head_id,_),
                                    Shape<Int<kBlockN>,Int<HeadDim>>{},
                                    make_coord(_,0));
    // ───────────────────────────── QKV global memory init ────────────────────────────

    

    // ─────────────────────────────────── QKV Shared Memory init ──────────────────────────────────
    //
    //  ┌─────────────────────────┬──────────────────────────┬──────────────────────────────┐
    //  │                         │                          │         V_sharedmem          │
    //  │       Q_sharedmem       │       K_sharedmem        │      V_sharedmem_trans       │
    //  │                         │                          │  V_sharedmem_trans_noswizzle │
    //  │                         │                          │                              │
    //  │                         │                          │        SmemLayoutKV          │
    //  │       SmemLayoutQ       │       SmemLayoutKV       │     SmemLayoutVtransposed    │
    //  │                         │                          │SmemLayoutVtransposedNoSwizzle│
    //  │                         │                          │                              │
    //  │ (kBlockM=64,HeadDim=64) │  (kBlockN=64,HeadDim=64) │   (kBlockN=64,HeadDim=64)    │
    //  │                         │                          │   (HeadDim=64,kBlockN=64)    │
    //  │                         │                          │                              │
    //  └─────────────────────────┴──────────────────────────┴──────────────────────────────┘
    //  ↑                         ↑                          ↑
    //  smem                   +size(Q)                 +size(Q)+size(K)
    //
    //  V_sharedmem_trans        → use same memory as V_sharedmem, interpreted with transposed layout (HeadDim,kBlockN)
    //  V_sharedmem_trans_noswizzle → use same memory as V_sharedmem, interpreted with transposed layout without swizzle
    Tensor Q_sharedmem = make_tensor(make_smem_ptr(reinterpret_cast<__nv_bfloat16*>(smem)), SmemLayoutQ{});
    Tensor K_sharedmem = make_tensor(Q_sharedmem.data()+size(Q_sharedmem),SmemLayoutKV{});
    Tensor V_sharedmem = make_tensor(K_sharedmem.data()+size(K_sharedmem),SmemLayoutKV{});
    Tensor V_sharedmem_trans = make_tensor(V_sharedmem.data(),SmemLayoutVtransposed{});
    Tensor V_sharedmem_trans_noswizzle = make_tensor(V_sharedmem.data(), SmemLayoutVtransposedNoSwizzle{});
    // ─────────────────────────────────── QKV Shared Memory init ──────────────────────────────────



    // ─────────────────────────────────── tiled copy gmem to smem ──────────────────────────────────
    //  partition_S / partition_D
    //
    //  globalmem_tiled_copy_QKV describe how 512 threads collaborate to move a tile
    //  get_thread_slice(tidx) cut the slice for the current thread
    //
    //  Q:
    //  gmem Q_globalmem                    smem Q_sharedmem
    //  ┌──────────────┐                    ┌──────────────┐
    //  │  .  T0  .    │                    │  .  T0  .    │
    //  │  .  T1  .    │      ──────►       │  .  T1  .    │
    //  │  . ...  .    │      cp.async      │  . ...  .    │
    //  │  . T511 .    │                    │  . T511 .    │
    //  └──────────────┘                    └──────────────┘
    //    tiled_Q_src                          tiled_Q_dst
    //  (which part will be read)           (which part will be written)
    //
    //  K/V the same, but K_src/V_src contain the 3rd dimension nblocksN:
    //  tiled_K_src shape: (CPY, CPY_N, CPY_K, nblocksN)
    //  In the loop, use tiled_K_src(_, _, _, n_block) to get the current block
    GmemTiledCopyQKV tiled_cpy_g2s;
    auto tiled_cpy_g2s_thread = tiled_cpy_g2s.get_thread_slice(tidx); 
    Tensor tiled_cpy_g2s_thread_Qsrc = tiled_cpy_g2s_thread.partition_S(Q_globalmem);
    Tensor tiled_cpy_g2s_thread_Qdst = tiled_cpy_g2s_thread.partition_D(Q_sharedmem);
    Tensor tiled_cpy_g2s_thread_Ksrc = tiled_cpy_g2s_thread.partition_S(K_globalmem);
    Tensor tiled_cpy_g2s_thread_Kdst = tiled_cpy_g2s_thread.partition_D(K_sharedmem);
    Tensor tiled_cpy_g2s_thread_Vsrc = tiled_cpy_g2s_thread.partition_S(V_globalmem);
    Tensor tiled_cpy_g2s_thread_Vdst = tiled_cpy_g2s_thread.partition_D(V_sharedmem);
    // ─────────────────────────────────── tiled copy gmem to smem ──────────────────────────────────



    // ─────────────────────────────────── tiled mma ──────────────────────────────────
    //  partition_fragment
    //
    //  TiledMma: 4 warp × 1 × 1，each warp calculate 16 rows
    //
    //  Q_reg  shape: (MMA, MMA_M, MMA_K)  ← GEMM1 A operand registers
    //  K_reg  shape: (MMA, MMA_N, MMA_K)  ← GEMM1 B operand registers
    //  V_reg  shape: (MMA, MMA_K, MMA_N)  ← GEMM2 B operand registers
    //  acc_O  shape: (MMA, MMA_M, MMA_K)  ← GEMM2 C accumulators (float)
    //
    //  ┌─────────────────────────────────────┐
    //  │ warp0: acc_O  0~15                  │
    //  ├─────────────────────────────────────┤
    //  │ warp1: acc_O  16~31                 │
    //  ├─────────────────────────────────────┤
    //  │ warp2: acc_O  32~47                 │
    //  ├─────────────────────────────────────┤
    //  │ warp3: acc_O  48~63                 │
    //  └─────────────────────────────────────┘
    //
    TiledMma tiled_mma;
    auto tiled_mma_thread = tiled_mma.get_thread_slice(tidx);
    Tensor tiled_mma_thread_Q = tiled_mma_thread.partition_fragment_A(Q_sharedmem);
    Tensor tiled_mma_thread_K = tiled_mma_thread.partition_fragment_B(K_sharedmem);
    Tensor tiled_mma_thread_V = tiled_mma_thread.partition_fragment_B(V_sharedmem_trans_noswizzle);
    Tensor tiled_mma_acc_O = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<HeadDim>>{});
    // ─────────────────────────────────── tiled mma ──────────────────────────────────



    // ─────────────────────────────────── tiled copy smem to reg ──────────────────────────────────
    //  partition_S for smem→reg copy
    //
    //  tiled_smem_reg_Q_src: thread read Q_sharedmem use ldmatrix 
    //  tiled_smem_reg_K_src: thread read K_sharedmem use ldmatrix 
    //  tiled_smem_reg_V_src: thread read V_sharedmem_trans use ldmatrix.T 
    //
    //     Q_sharedmem             reg_Q
    //  ┌──────────────┐          ┌──────┐
    //  │▓▓ T0  ▓▓▓    │ ldmatrix │ T0   │
    //  │▓▓ T1  ▓▓▓    │ ───────► │ T1   │
    //  │     ...      │          │ ...  │
    //  └──────────────┘          └──────┘
    //  tiled_smem_reg_Q_src       reg_Q（use cute::copy）
    auto tiled_copy_s2r_Q = make_tiled_copy_A(SmemCopyAtom{}, tiled_mma);
    auto tiled_copy_s2r_thread_Q = tiled_copy_s2r_Q.get_thread_slice(tidx);
    Tensor tiled_cpy_s2r_Qsrc = tiled_copy_s2r_thread_Q.partition_S(Q_sharedmem);

    auto tiled_copy_s2r_K = make_tiled_copy_B(SmemCopyAtom{}, tiled_mma);
    auto tiled_copy_s2r_thread_K = tiled_copy_s2r_K.get_thread_slice(tidx);
    Tensor tiled_cpy_s2r_Ksrc = tiled_copy_s2r_thread_K.partition_S(K_sharedmem);

    auto tiled_copy_s2r_V = make_tiled_copy_B(SmemCopyAtomTransposed{}, tiled_mma);
    auto tiled_copy_s2r_thread_V = tiled_copy_s2r_V.get_thread_slice(tidx);
    Tensor tiled_cpy_s2r_Vsrc = tiled_copy_s2r_thread_V.partition_S(V_sharedmem_trans);
    // ─────────────────────────────────── tiled copy smem to reg ──────────────────────────────────

    // prologue
    // load QK
    copy(tiled_cpy_g2s,tiled_cpy_g2s_thread_Qsrc, tiled_cpy_g2s_thread_Qdst);
    int n_block = n_block_max - 1;
    copy(tiled_cpy_g2s, tiled_cpy_g2s_thread_Ksrc(_, _, _, n_block), tiled_cpy_g2s_thread_Kdst);
    cute::cp_async_fence();

    clear(tiled_mma_acc_O);

    Softmax<2 * size<1>(tiled_mma_acc_O)> softmax;

    for (; n_block >= n_block_min; --n_block) 
    {

        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        clear(acc_s);
        cp_async_wait<0>();
        __syncthreads();

        // load V
        copy(tiled_cpy_g2s, tiled_cpy_g2s_thread_Vsrc(_, _, _, n_block), tiled_cpy_g2s_thread_Vdst);
        cute::cp_async_fence();

        // S = Q * K^T
        gemm(acc_s,                                             // acc
             tiled_mma_thread_Q, tiled_mma_thread_K,            // reg Q and K
             tiled_cpy_s2r_Qsrc, tiled_cpy_s2r_Ksrc,            // smem Q and K
             tiled_mma,                                         // tiled mma
             tiled_copy_s2r_Q, tiled_copy_s2r_K,                // tiled copy smem to reg for Q and K
             tiled_copy_s2r_thread_Q, tiled_copy_s2r_thread_K); // thread slice for tiled copy smem to reg for Q and K

        cp_async_wait<0>();
        __syncthreads();

        // load next K
        if (n_block > n_block_min) {
            copy(tiled_cpy_g2s, tiled_cpy_g2s_thread_Ksrc(_, _, _, n_block - 1), tiled_cpy_g2s_thread_Kdst);
            cute::cp_async_fence();
        }

        // softmax
        if (n_block == n_block_max - 1)
        {
            softmax.template softmax_rescale_o<true>(acc_s, tiled_mma_acc_O, softmax_scale_log2);
        }
        else
        {
            softmax.template softmax_rescale_o<false>(acc_s, tiled_mma_acc_O, softmax_scale_log2);
        }

        // PV
        Tensor rP = convert_type<__nv_bfloat16>(acc_s);
        auto tOrP = make_tensor(rP.data(), convert_layout_acc_Aregs<TiledMma>(rP.layout()));
        gemm_rs(tiled_mma_acc_O,          // acc
                tOrP,                     // reg P
                tiled_mma_thread_V,       // reg V
                tiled_cpy_s2r_Vsrc,       // smem V
                tiled_mma,                // tiled mma
                tiled_copy_s2r_V,         // tiled copy smem to reg for V
                tiled_copy_s2r_thread_V); // thread slice for tiled copy smem to reg for V
    }

    // Epilogue

    softmax.template normalize_softmax_lse(tiled_mma_acc_O, scale_softmax);

    // Convert acc_o from fp32 to fp16/bf16
    Tensor Oreg = convert_type<__nv_bfloat16>(tiled_mma_acc_O);
    Tensor Osmem = make_tensor(Q_sharedmem.data(), SmemLayoutO{});    // (SMEM_M,SMEM_N)

    auto smem_tiled_copy_O = make_tiled_copy_C(SmemCopyAtomO{}, tiled_mma);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
    Tensor taccOrO = smem_thr_copy_O.retile_S(Oreg);        // ((Atom,AtomNum), MMA_M, MMA_N)
    Tensor taccOsO = smem_thr_copy_O.partition_D(Osmem);     // ((Atom,AtomNum),PIPE_M,PIPE_N)


    cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

    Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<__nv_bfloat16*>(O_ptr)
                                          + offset(batch_id, seqlen_o*HeadNum*HeadDim)),
                            make_shape(actual_seqlen_q, HeadNum, HeadDim),
                            make_stride(seqlen_o*HeadNum*HeadDim, HeadDim, _1{}));
    Tensor gO = local_tile(mO(_, head_id, _), Shape<Int<kBlockM>, Int<HeadDim>>{},
                           make_coord(block_id, 0));  // (kBlockM, kHeadDim)

    GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
    Tensor tOsO = gmem_thr_copy_O.partition_S(Osmem);        // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

    __syncthreads();

    Tensor tOrO = make_tensor<__nv_bfloat16>(shape(tOgO));
    cute::copy(gmem_tiled_copy_O, tOsO, tOrO);

    copy(gmem_tiled_copy_O, tOrO, tOgO);
        if (threadIdx.x == 0) {
        printf("@@@block_id: %d, batch_id: %d, head_id: %d\n", block_id, batch_id, head_id);
        printf("@@@actual_seqlen_q: %d, actual_seqlen_kv: %d\n", actual_seqlen_q, actual_seqlen_kv);
        printf("@@@kBlockM: %d, kBlockN: %d\n", kBlockM, kBlockN);
    }
}
