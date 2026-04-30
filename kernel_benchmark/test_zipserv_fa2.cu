#include <cuda_runtime.h>
#include <cstdio>
#include "zipserv_fa2.cuh"
#include "utils.h"

int main_empty_kernel() {
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
int main()
{
    int Wq_M_GLOBAL = 64;
    int Wq_N_GLOBAL = 64;
    int Wk_M_GLOBAL = 64;
    int Wk_N_GLOBAL = 64;
    int Wv_M_GLOBAL = 64;
    int Wv_N_GLOBAL = 64;
    int X_M_GLOBAL = 64;
    int X_N_GLOBAL = 64;

    int SPLIT_K = 1;

    // Host memory
    __nv_bfloat16* Wq_host            = NULL;  // row major    
    __nv_bfloat16* Wk_host            = NULL;  // row major
    __nv_bfloat16* Wv_host            = NULL;  // row major
    __nv_bfloat16* X_host            = NULL;  // col major
    __nv_bfloat16* X_Transposed_host = NULL;  // row major

    __nv_bfloat16* Q_host            = NULL;  // row major
    __nv_bfloat16* K_host            = NULL;  // row major
    __nv_bfloat16* V_host            = NULL;  // row major

    // Device memory
    __nv_bfloat16* Wq_device            = NULL;  // row major    
    __nv_bfloat16* Wk_device            = NULL;  // row major
    __nv_bfloat16* Wv_device            = NULL;  // row major
    __nv_bfloat16* X_device            = NULL;  // col major
    __nv_bfloat16* X_Transposed_device = NULL;  // row major

    __nv_bfloat16* Q_device            = NULL;  // row major
    __nv_bfloat16* K_device            = NULL;  // row major
    __nv_bfloat16* V_device            = NULL;  // row major
    
    printf("Allocating host memory...\n");
    Wq_host            = (__nv_bfloat16*)malloc(sizeof(__nv_bfloat16) * Wq_M_GLOBAL * Wq_N_GLOBAL);
    Wk_host            = (__nv_bfloat16*)malloc(sizeof(__nv_bfloat16) * Wk_M_GLOBAL * Wk_N_GLOBAL);
    Wv_host            = (__nv_bfloat16*)malloc(sizeof(__nv_bfloat16) * Wv_M_GLOBAL * Wv_N_GLOBAL);
    X_host            = (__nv_bfloat16*)malloc(sizeof(__nv_bfloat16) * X_M_GLOBAL * X_N_GLOBAL);
    X_Transposed_host = (__nv_bfloat16*)malloc(sizeof(__nv_bfloat16) * X_N_GLOBAL * X_M_GLOBAL);

    Q_host            = (__nv_bfloat16*)malloc(sizeof(__nv_bfloat16) * Wq_M_GLOBAL * X_N_GLOBAL);
    K_host            = (__nv_bfloat16*)malloc(sizeof(__nv_bfloat16) * Wk_M_GLOBAL * X_N_GLOBAL);
    V_host            = (__nv_bfloat16*)malloc(sizeof(__nv_bfloat16) * Wv_M_GLOBAL * X_N_GLOBAL);

    cudaMalloc(reinterpret_cast<void**>(&Wq_device), sizeof(__nv_bfloat16) * Wq_M_GLOBAL * Wq_N_GLOBAL);
    cudaMalloc(reinterpret_cast<void**>(&Wk_device), sizeof(__nv_bfloat16) * Wk_M_GLOBAL * Wk_N_GLOBAL);
    cudaMalloc(reinterpret_cast<void**>(&Wv_device), sizeof(__nv_bfloat16) * Wv_M_GLOBAL * Wv_N_GLOBAL);
    cudaMalloc(reinterpret_cast<void**>(&X_device), sizeof(__nv_bfloat16) * X_M_GLOBAL * X_N_GLOBAL);
    cudaMalloc(reinterpret_cast<void**>(&X_Transposed_device), sizeof(__nv_bfloat16) * X_N_GLOBAL * X_M_GLOBAL);
    cudaMalloc(reinterpret_cast<void**>(&Q_device), sizeof(__nv_bfloat16) * Wq_M_GLOBAL * X_N_GLOBAL);
    cudaMalloc(reinterpret_cast<void**>(&K_device), sizeof(__nv_bfloat16) * Wk_M_GLOBAL * X_N_GLOBAL);
    cudaMalloc(reinterpret_cast<void**>(&V_device), sizeof(__nv_bfloat16) * Wv_M_GLOBAL * X_N_GLOBAL);

    init_host_matrices_bf16(Wq_host, X_host, Wq_M_GLOBAL, Wq_N_GLOBAL, X_N_GLOBAL);
    init_host_matrices_bf16_one(Wk_host, Wk_M_GLOBAL, Wk_N_GLOBAL);
    init_host_matrices_bf16_one(Wv_host, Wv_M_GLOBAL, Wv_N_GLOBAL);

    for (int i = 0; i < X_M_GLOBAL; i++)
        for (int j = 0; j < X_N_GLOBAL; j++)
            X_Transposed_device[i * X_N_GLOBAL + j] = X_host[i + j * X_M_GLOBAL];

    cudaMemcpy(Wq_device, Wq_host, sizeof(__nv_bfloat16) * Wq_M_GLOBAL * Wq_N_GLOBAL, cudaMemcpyHostToDevice);
    cudaMemcpy(Wk_device, Wk_host, sizeof(__nv_bfloat16) * Wk_M_GLOBAL * Wk_N_GLOBAL, cudaMemcpyHostToDevice);
    cudaMemcpy(Wv_device, Wv_host, sizeof(__nv_bfloat16) * Wv_M_GLOBAL * Wv_N_GLOBAL, cudaMemcpyHostToDevice);
    cudaMemcpy(X_device, X_host, sizeof(__nv_bfloat16) * X_M_GLOBAL * X_N_GLOBAL, cudaMemcpyHostToDevice);
    cudaMemcpy(X_Transposed_device, X_Transposed_host, sizeof(__nv_bfloat16) * X_N_GLOBAL * X_M_GLOBAL, cudaMemcpyHostToDevice);
        
    // Wq
    __nv_bfloat16* Wq_top_exponents_cpu = nullptr;
    __nv_bfloat16* Wq_compressed_full_cpu = nullptr;
    uint8_t* Wq_sign_mantissa_cpu = nullptr;
    uint64_t* Wq_bitmap1_cpu = nullptr;
    uint64_t* Wq_bitmap2_cpu = nullptr;
    uint64_t* Wq_bitmap3_cpu = nullptr;
    int* Wq_TileOffsets_cpu = nullptr;
    int* Wq_TileOffsets_median_cpu = nullptr;
    int* Wq_TileOffsets_global_cpu = nullptr;
    int Wq_max_high_freq_count = 0;
    int Wq_max_full_count = 0;
    uint8_t Wq_start_exp = 0;
    int num_global_tiles = InitBF16MatrixTripleBitmap_Host(
        A_host, M_GLOBAL, K_GLOBAL, 
        8, 16, 64, 8, 64, 64,
        &Wq_top_exponents_cpu, &Wq_compressed_full_cpu, &Wq_sign_mantissa_cpu,
        &Wq_bitmap1_cpu, &Wq_bitmap2_cpu, &Wq_bitmap3_cpu,
        &Wq_TileOffsets_cpu, &Wq_TileOffsets_median_cpu, &Wq_TileOffsets_global_cpu,
        Wq_max_high_freq_count, Wq_max_full_count, Wq_start_exp);
    int Wq_tile_m = 8;
    int Wq_tile_k = 8;
    int Wq_tile_m_global = 64;
    int Wq_tile_k_global = 64;
    int Wq_num_tiles_m = Wq_M_GLOBAL / Wq_tile_m;
    int Wq_num_tiles_k = Wq_N_GLOBAL / Wq_tile_k;
    int Wq_num_tiles = Wq_num_tiles_m * Wq_num_tiles_k;
    int Wq_num_median_tiles_m = Wq_M_GLOBAL / 16;
    int Wq_num_median_tiles_k = Wq_N_GLOBAL / 64;
    int Wq_num_median_tiles = Wq_num_median_tiles_m * Wq_num_median_tiles_k;
    int Wq_high_freq_count = Wq_TileOffsets_global_cpu[num_global_tiles * 2];
    int Wq_full_count = Wq_TileOffsets_global_cpu[num_global_tiles * 2 + 1];
    size_t Wq_original_size = Wq_M_GLOBAL * Wq_N_GLOBAL * sizeof(__nv_bfloat16);
    size_t Wq_compressed_size = 
        // (7 * sizeof(__nv_bfloat16)) +                    // High-freq exponents
        (Wq_high_freq_count * sizeof(uint8_t)) +            // High-freq elements (sign+mantissa)
        (Wq_full_count * sizeof(__nv_bfloat16)) +           // Non-high-freq elements
        (Wq_num_tiles * sizeof(uint64_t) * 3) +             // Three bitmaps
        // (Wq_num_tiles * 2 * sizeof(int)) +                  // Small tile offsets
        (Wq_num_median_tiles * 2 * sizeof(int)) +           // Medium tile offsets
        ((num_global_tiles + 1) * 2 * sizeof(int));      // Global tile offsets
    float Wq_compression_ratio = (float)Wq_original_size / Wq_compressed_size;
    __nv_bfloat16* Wq_top_exponents_gpu = nullptr;
    __nv_bfloat16* Wq_compressed_full_gpu = nullptr;
    uint8_t* Wq_sign_mantissa_gpu = nullptr;
    uint64_t* Wq_bitmap1_gpu = nullptr;
    uint64_t* Wq_bitmap2_gpu = nullptr;
    uint64_t* Wq_bitmap3_gpu = nullptr;
    int* Wq_TileOffsets_gpu = nullptr;
    int* Wq_TileOffsets_median_gpu = nullptr;
    int* Wq_TileOffsets_global_gpu = nullptr;
    cudaMalloc(&Wq_top_exponents_gpu, 7 * sizeof(__nv_bfloat16)); // 7 high-freq exponents
    cudaMalloc(&Wq_compressed_full_gpu, full_count * sizeof(__nv_bfloat16));
    cudaMalloc(&Wq_sign_mantissa_gpu, high_freq_count * sizeof(uint8_t));
    cudaMalloc(&Wq_bitmap1_gpu, num_tiles * sizeof(uint64_t));
    cudaMalloc(&Wq_bitmap2_gpu, num_tiles * sizeof(uint64_t));
    cudaMalloc(&Wq_bitmap3_gpu, num_tiles * sizeof(uint64_t));
    cudaMalloc(&Wq_TileOffsets_gpu, num_tiles * 2 * sizeof(int));
    cudaMalloc(&Wq_TileOffsets_median_gpu, num_median_tiles * 2 * sizeof(int));
    cudaMalloc(&Wq_TileOffsets_global_gpu, (num_global_tiles + 1) * 2 * sizeof(int));
    int* Wq_max_high_freq_gpu = nullptr;
    int* Wq_max_full_gpu = nullptr;
    cudaMalloc(&Wq_max_high_freq_gpu, sizeof(int));
    cudaMalloc(&Wq_max_full_gpu, sizeof(int));
    printf("Copying compressed data to GPU...\n");
    // Copy compressed data to device
    cudaMemcpy(Wq_top_exponents_gpu, Wq_top_exponents_cpu, 7 * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(Wq_compressed_full_gpu, Wq_compressed_full_cpu, full_count * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(Wq_sign_mantissa_gpu, Wq_sign_mantissa_cpu, high_freq_count * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(Wq_bitmap1_gpu, Wq_bitmap1_cpu, num_tiles * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(Wq_bitmap2_gpu, Wq_bitmap2_cpu, num_tiles * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(Wq_bitmap3_gpu, Wq_bitmap3_cpu, num_tiles * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(Wq_TileOffsets_gpu, Wq_TileOffsets_cpu, num_tiles * 2 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(Wq_TileOffsets_median_gpu, Wq_TileOffsets_median_cpu, num_median_tiles * 2 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(Wq_TileOffsets_global_gpu, Wq_TileOffsets_global_cpu, (num_global_tiles + 1) * 2 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(Wq_max_high_freq_gpu, &Wq_max_high_freq_count, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(Wq_max_full_gpu, &Wq_max_full_count, sizeof(int), cudaMemcpyHostToDevice);
    // Free host compressed data memory
    free(Wq_top_exponents_cpu);
    free(Wq_compressed_full_cpu);
    free(Wq_sign_mantissa_cpu);
    free(Wq_bitmap1_cpu);
    free(Wq_bitmap2_cpu);
    free(Wq_bitmap3_cpu);
    free(Wq_TileOffsets_cpu);
    free(Wq_TileOffsets_median_cpu);
    free(Wq_TileOffsets_global_cpu);

    // Wk
    __nv_bfloat16* Wk_top_exponents_cpu = nullptr;
    __nv_bfloat16* Wk_compressed_full_cpu = nullptr;
    uint8_t* Wk_sign_mantissa_cpu = nullptr;
    uint64_t* Wk_bitmap1_cpu = nullptr;
    uint64_t* Wk_bitmap2_cpu = nullptr;
    uint64_t* Wk_bitmap3_cpu = nullptr;
    int* Wk_TileOffsets_cpu = nullptr;
    int* Wk_TileOffsets_median_cpu = nullptr;
    int* Wk_TileOffsets_global_cpu = nullptr;
    int Wk_max_high_freq_count = 0;
    int Wk_max_full_count = 0;
    uint8_t Wk_start_exp = 0;
    int num_global_tiles = InitBF16MatrixTripleBitmap_Host(
        A_host, M_GLOBAL, K_GLOBAL, 
        8, 16, 64, 8, 64, 64,
        &Wk_top_exponents_cpu, &Wk_compressed_full_cpu, &Wk_sign_mantissa_cpu,
        &Wk_bitmap1_cpu, &Wk_bitmap2_cpu, &Wk_bitmap3_cpu,
        &Wk_TileOffsets_cpu, &Wk_TileOffsets_median_cpu, &Wk_TileOffsets_global_cpu,
        Wk_max_high_freq_count, Wk_max_full_count, Wk_start_exp);
    int Wk_tile_m = 8;
    int Wk_tile_k = 8;
    int Wk_tile_m_global = 64;
    int Wk_tile_k_global = 64;
    int Wk_num_tiles_m = Wk_M_GLOBAL / Wk_tile_m;
    int Wk_num_tiles_k = Wk_N_GLOBAL / Wk_tile_k;
    int Wk_num_tiles = Wk_num_tiles_m * Wk_num_tiles_k;
    int Wk_num_median_tiles_m = Wk_M_GLOBAL / 16;
    int Wk_num_median_tiles_k = Wk_N_GLOBAL / 64;
    int Wk_num_median_tiles = Wk_num_median_tiles_m * Wk_num_median_tiles_k;
    int Wk_high_freq_count = Wk_TileOffsets_global_cpu[num_global_tiles * 2];
    int Wk_full_count = Wk_TileOffsets_global_cpu[num_global_tiles * 2 + 1];
    size_t Wk_original_size = Wk_M_GLOBAL * Wk_N_GLOBAL * sizeof(__nv_bfloat16);
    size_t Wk_compressed_size = 
        // (7 * sizeof(__nv_bfloat16)) +                    // High-freq exponents
        (Wk_high_freq_count * sizeof(uint8_t)) +            // High-freq elements (sign+mantissa)
        (Wk_full_count * sizeof(__nv_bfloat16)) +           // Non-high-freq elements
        (Wk_num_tiles * sizeof(uint64_t) * 3) +             // Three bitmaps
        // (Wk_num_tiles * 2 * sizeof(int)) +                  // Small tile offsets
        (Wk_num_median_tiles * 2 * sizeof(int)) +           // Medium tile offsets
        ((num_global_tiles + 1) * 2 * sizeof(int));      // Global tile offsets
    float Wk_compression_ratio = (float)Wk_original_size / Wk_compressed_size;
    __nv_bfloat16* Wk_top_exponents_gpu = nullptr;
    __nv_bfloat16* Wk_compressed_full_gpu = nullptr;
    uint8_t* Wk_sign_mantissa_gpu = nullptr;
    uint64_t* Wk_bitmap1_gpu = nullptr;
    uint64_t* Wk_bitmap2_gpu = nullptr;
    uint64_t* Wk_bitmap3_gpu = nullptr;
    int* Wk_TileOffsets_gpu = nullptr;
    int* Wk_TileOffsets_median_gpu = nullptr;
    int* Wk_TileOffsets_global_gpu = nullptr;
    cudaMalloc(&Wk_top_exponents_gpu, 7 * sizeof(__nv_bfloat16)); // 7 high-freq exponents
    cudaMalloc(&Wk_compressed_full_gpu, full_count * sizeof(__nv_bfloat16));
    cudaMalloc(&Wk_sign_mantissa_gpu, high_freq_count * sizeof(uint8_t));
    cudaMalloc(&Wk_bitmap1_gpu, num_tiles * sizeof(uint64_t));
    cudaMalloc(&Wk_bitmap2_gpu, num_tiles * sizeof(uint64_t));
    cudaMalloc(&Wk_bitmap3_gpu, num_tiles * sizeof(uint64_t));
    cudaMalloc(&Wk_TileOffsets_gpu, num_tiles * 2 * sizeof(int));
    cudaMalloc(&Wk_TileOffsets_median_gpu, num_median_tiles * 2 * sizeof(int));
    cudaMalloc(&Wk_TileOffsets_global_gpu, (num_global_tiles + 1) * 2 * sizeof(int));
    int* Wk_max_high_freq_gpu = nullptr;
    int* Wk_max_full_gpu = nullptr;
    cudaMalloc(&Wk_max_high_freq_gpu, sizeof(int));
    cudaMalloc(&Wk_max_full_gpu, sizeof(int));
    printf("Copying compressed data to GPU...\n");
    // Copy compressed data to device
    cudaMemcpy(Wk_top_exponents_gpu, Wk_top_exponents_cpu, 7 * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(Wk_compressed_full_gpu, Wk_compressed_full_cpu, full_count * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(Wk_sign_mantissa_gpu, Wk_sign_mantissa_cpu, high_freq_count * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(Wk_bitmap1_gpu, Wk_bitmap1_cpu, num_tiles * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(Wk_bitmap2_gpu, Wk_bitmap2_cpu, num_tiles * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(Wk_bitmap3_gpu, Wk_bitmap3_cpu, num_tiles * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(Wk_TileOffsets_gpu, Wk_TileOffsets_cpu, num_tiles * 2 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(Wk_TileOffsets_median_gpu, Wk_TileOffsets_median_cpu, num_median_tiles * 2 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(Wk_TileOffsets_global_gpu, Wk_TileOffsets_global_cpu, (num_global_tiles + 1) * 2 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(Wk_max_high_freq_gpu, &Wk_max_high_freq_count, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(Wk_max_full_gpu, &Wk_max_full_count, sizeof(int), cudaMemcpyHostToDevice);
    // Free host compressed data memory
    free(Wk_top_exponents_cpu);
    free(Wk_compressed_full_cpu);
    free(Wk_sign_mantissa_cpu);
    free(Wk_bitmap1_cpu);
    free(Wk_bitmap2_cpu);
    free(Wk_bitmap3_cpu);
    free(Wk_TileOffsets_cpu);
    free(Wk_TileOffsets_median_cpu);
    free(Wk_TileOffsets_global_cpu);

    // Q=Wq*X
    BF16TripleBitmap_MM_API(
        0,
        Wq_sign_mantissa_gpu,Wq_compressed_full_gpu,
        Wq_bitmap1_gpu,Wq_bitmap2_gpu,Wq_bitmap3_gpu,
        Wq_TileOffsets_median_gpu,Wq_TileOffsets_global_gpu,
        Wq_max_high_freq_gpu,Wq_max_full_gpu,
        Wq_start_exp,
        X_Transposed_device,
        Q_device,
        Wq_M_GLOBAL, X_N_GLOBAL, Wq_N_GLOBAL,
        nullptr,
        1
    );
    Q_host = (__nv_bfloat16*)malloc(sizeof(__nv_bfloat16) * Wq_M_GLOBAL * X_N_GLOBAL);
    cudaMemcpy(Q_host, Q_device, sizeof(__nv_bfloat16) * Wq_M_GLOBAL * X_N_GLOBAL, cudaMemcpyDeviceToHost);  // Col Major
    print_bf16_matrix("Matrix Q ", Q_host, Wq_M_GLOBAL, X_N_GLOBAL);
    // K=Wk*X
        BF16TripleBitmap_MM_API(
        0,
        Wk_sign_mantissa_gpu,Wk_compressed_full_gpu,
        Wk_bitmap1_gpu,Wk_bitmap2_gpu,Wk_bitmap3_gpu,
        Wk_TileOffsets_median_gpu,Wk_TileOffsets_global_gpu,
        Wk_max_high_freq_gpu,Wk_max_full_gpu,
        Wk_start_exp,
        X_device,
        K_device,
        Wk_M_GLOBAL, X_N_GLOBAL, Wk_N_GLOBAL,
        nullptr,
        1
    );
    // V=Wv*X
        BF16TripleBitmap_MM_API(
        0,
        Wv_sign_mantissa_gpu,Wv_compressed_full_gpu,
        Wv_bitmap1_gpu,Wv_bitmap2_gpu,Wv_bitmap3_gpu,
        Wv_TileOffsets_median_gpu,Wv_TileOffsets_global_gpu,
        Wv_max_high_freq_gpu,Wv_max_full_gpu,
        Wv_start_exp,
        X_Transposed_device,
        V_device,
        Wv_M_GLOBAL, X_N_GLOBAL, Wv_N_GLOBAL,
        nullptr,
        1
    );

    return 0;
}