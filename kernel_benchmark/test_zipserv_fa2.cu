#include <cuda_runtime.h>
#include <cstdio>
#include "zipserv_fa2.cuh"

int main() {
    // 假设每个参数都用最小合法值
    int seqlen_q = 1, seqlen_kv = 1, seqlen_o = 1, actual_seqlen_q = 1, actual_seqlen_kv = 1;
    __nv_bfloat16 *Q = nullptr, *K = nullptr, *V = nullptr, *O = nullptr;
    float softmax_scale_log2 = 1.0f, scale_softmax = 1.0f;

    // 你可以根据 kernel 内部需求分配 device memory
    // cudaMalloc(&Q, ...); cudaMalloc(&K, ...); cudaMalloc(&V, ...); cudaMalloc(&O, ...);

    compute_attn<<<1, 32, 48*1024>>>(
        seqlen_q, seqlen_kv, seqlen_o, actual_seqlen_q, actual_seqlen_kv,
        Q, K, V, O, softmax_scale_log2, scale_softmax
    );

    cudaDeviceSynchronize();

    printf("kernel launch done\\n");
    // cudaFree(Q); cudaFree(K); cudaFree(V); cudaFree(O);
    return 0;
}