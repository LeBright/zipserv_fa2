#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include "zipserv_fa2.cuh"
#include "L_API.cuh"
#include "utils.h"

#define CUDA_CHECK(call)                                                        \
    do                                                                          \
    {                                                                           \
        cudaError_t _err = (call);                                              \
        if (_err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s (%d)\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(_err), (int)_err);   \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

void print_bf16_matrix(const char* name, __nv_bfloat16* matrix, int rows, int cols, int max_rows = 128, int max_cols = 32) {
    printf("\n===== %s [%dx%d] =====\n", name, rows, cols);
    
    int display_rows = std::min(rows, max_rows);
    int display_cols = std::min(cols, max_cols);
    
    for (int i = 0; i < display_rows; i++) {
        for (int j = 0; j < display_cols; j++) {
            printf("%8.4f ", __bfloat162float(matrix[i * cols + j]));
        }
        if (cols > max_cols) printf("...");
        printf("\n");
    }
    if (rows > max_rows) printf("...\n");
    
    // Print some BF16 internal details
    printf("BF16 details example (first 4 elements):\n");
    for (int i = 0; i < std::min(4, rows * cols); i++) {
        uint16_t bits = __bfloat16_as_ushort(matrix[i]);
        uint8_t sign = (bits >> 15) & 0x1;
        uint8_t exponent = (bits >> 7) & 0xFF;
        uint8_t mantissa = bits & 0x7F;
        
        printf("Element[%d] = %8.4f (sign=%d, exp=%3d, mantissa=%3d, raw=0x%04X)\n", 
               i, __bfloat162float(matrix[i]), sign, exponent, mantissa, bits);
    }
    printf("\n");
}

void flush_l2_cache() {
    static void* d_flush_buffer = nullptr;
    static size_t flush_buffer_size = 0;
    static bool initialized = false;
    static bool disabled = false;

    if (disabled) {
        return;
    }

    (void)cudaGetLastError();
    
    if (!initialized) {
        // Get L2 cache size dynamically
        int device = 0;
        int l2_size = 0;
        cudaError_t err = cudaGetDevice(&device);
        if (err != cudaSuccess) {
            fprintf(stderr, "WARN: flush_l2_cache disabled (cudaGetDevice failed): %s (%d)\n",
                    cudaGetErrorString(err), (int)err);
            disabled = true;
            return;
        }

        err = cudaDeviceGetAttribute(&l2_size, cudaDevAttrL2CacheSize, device);
        if (err != cudaSuccess || l2_size <= 0) {
            fprintf(stderr, "WARN: flush_l2_cache disabled (query L2 size failed): %s (%d), l2_size=%d\n",
                    cudaGetErrorString(err), (int)err, l2_size);
            disabled = true;
            return;
        }

        // Try 2x L2 first, then degrade to 1x and 0.5x to avoid hard failures.
        size_t candidates[3] = {
            static_cast<size_t>(l2_size) * 2,
            static_cast<size_t>(l2_size),
            static_cast<size_t>(l2_size) / 2
        };

        for (size_t candidate_size : candidates) {
            if (candidate_size == 0) {
                continue;
            }

            err = cudaMalloc(&d_flush_buffer, candidate_size);
            if (err == cudaSuccess && d_flush_buffer != nullptr) {
                flush_buffer_size = candidate_size;
                initialized = true;
                break;
            }

            fprintf(stderr, "WARN: flush_l2_cache alloc %zu bytes failed: %s (%d)\n",
                    candidate_size, cudaGetErrorString(err), (int)err);
            d_flush_buffer = nullptr;
            (void)cudaGetLastError();
        }

        if (!initialized) {
            fprintf(stderr, "WARN: flush_l2_cache disabled (all allocation attempts failed).\n");
            disabled = true;
            return;
        }

        printf("L2 Cache size: %d bytes, Flush buffer size: %zu bytes\n", l2_size, flush_buffer_size);
    }
    
    // Flush L2 cache by writing to buffer larger than L2 cache
    cudaError_t err = cudaMemsetAsync(d_flush_buffer, 0, flush_buffer_size);
    if (err != cudaSuccess) {
        fprintf(stderr, "WARN: flush_l2_cache memset failed, disabling flush: %s (%d)\n",
                cudaGetErrorString(err), (int)err);
        if (d_flush_buffer) {
            (void)cudaFree(d_flush_buffer);
            d_flush_buffer = nullptr;
        }
        flush_buffer_size = 0;
        initialized = false;
        disabled = true;
        return;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "WARN: flush_l2_cache sync failed, disabling flush: %s (%d)\n",
                cudaGetErrorString(err), (int)err);
        if (d_flush_buffer) {
            (void)cudaFree(d_flush_buffer);
            d_flush_buffer = nullptr;
        }
        flush_buffer_size = 0;
        initialized = false;
        disabled = true;
    }
}

int main(int argc, char** argv)
{

    int d_model = 64;
    int seq_len = 64;
    int mode = 0;  // prepare_Q:0, prepare_QK:1 
    if (argc == 3) 
    {
        d_model = std::atoi(argv[1]);
        seq_len = std::atoi(argv[2]);
        if (d_model <= 0 || seq_len <= 0 || d_model % 64 != 0 || seq_len % 64 != 0) 
        {
            printf("Invalid args. d_model and seq_len must be positive multiples of 64.\n");
            printf("Usage: %s [d_model seq_len]\n", argv[0]);
            return -1;
        }
    } 
    else if(argc == 4)
    {
        d_model = std::atoi(argv[1]);
        seq_len = std::atoi(argv[2]);
        if (d_model <= 0 || seq_len <= 0 || d_model % 64 != 0 || seq_len % 64 != 0) 
        {
            printf("Invalid args. d_model and seq_len must be positive multiples of 64.\n");
            printf("Usage: %s [d_model seq_len]\n", argv[0]);
            return -1;
        }
        mode = std::atoi(argv[3]);
    }
    else if (argc != 1) 
    {
        printf("Usage: %s [d_model seq_len]\n", argv[0]);
        return -1;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int Wq_M_GLOBAL = d_model;
    int Wq_N_GLOBAL = d_model;
    int Wk_M_GLOBAL = d_model;
    int Wk_N_GLOBAL = d_model;
    int Wv_M_GLOBAL = d_model;
    int Wv_N_GLOBAL = d_model;
    int X_M_GLOBAL  = d_model;
    int X_N_GLOBAL  = seq_len;

    printf("Running config: d_model=%d, seq_len=%d\n", d_model, seq_len);

    // Host memory
    __nv_bfloat16* Wq_host             = NULL;   
    __nv_bfloat16* Wk_host             = NULL;  
    __nv_bfloat16* Wv_host             = NULL;  
    __nv_bfloat16* X_host              = NULL;   
    __nv_bfloat16* X_Transposed_host   = NULL;   

    __nv_bfloat16* Q_host              = NULL;  
    __nv_bfloat16* Q_host_cublas       = NULL;  
    __nv_bfloat16* K_host              = NULL;  
    __nv_bfloat16* K_host_cublas       = NULL;  
    __nv_bfloat16* V_host              = NULL;  
    __nv_bfloat16* V_host_cublas       = NULL;  
    __nv_bfloat16* O_host              = NULL;  
    __nv_bfloat16* O_host_baseline     = NULL;  

    // Device memory
    __nv_bfloat16* Wq_device           = NULL;
    __nv_bfloat16* Wk_device           = NULL;
    __nv_bfloat16* Wv_device           = NULL;
    __nv_bfloat16* X_device            = NULL; 
    __nv_bfloat16* X_Transposed_device = NULL; 

    __nv_bfloat16* Q_device            = NULL;  
    __nv_bfloat16* Q_device_cublas     = NULL;  
    __nv_bfloat16* K_device            = NULL;  
    __nv_bfloat16* K_device_cublas     = NULL;  
    __nv_bfloat16* V_device            = NULL;  
    __nv_bfloat16* V_device_cublas     = NULL;  
    __nv_bfloat16* Q_host_cpu          = NULL;
    __nv_bfloat16* K_host_cpu          = NULL;
    __nv_bfloat16* V_host_cpu          = NULL;
    __nv_bfloat16* O_device            = NULL;  
    
    printf("Allocating host memory...\n");
    Wq_host            = (__nv_bfloat16*)malloc(sizeof(__nv_bfloat16) * Wq_M_GLOBAL * Wq_N_GLOBAL);
    Wk_host            = (__nv_bfloat16*)malloc(sizeof(__nv_bfloat16) * Wk_M_GLOBAL * Wk_N_GLOBAL);
    Wv_host            = (__nv_bfloat16*)malloc(sizeof(__nv_bfloat16) * Wv_M_GLOBAL * Wv_N_GLOBAL);
    X_host             = (__nv_bfloat16*)malloc(sizeof(__nv_bfloat16) * X_M_GLOBAL * X_N_GLOBAL);
    X_Transposed_host  = (__nv_bfloat16*)malloc(sizeof(__nv_bfloat16) * X_N_GLOBAL * X_M_GLOBAL);

    Q_host             = (__nv_bfloat16*)malloc(sizeof(__nv_bfloat16) * Wq_M_GLOBAL * X_N_GLOBAL);
    Q_host_cublas      = (__nv_bfloat16*)malloc(sizeof(__nv_bfloat16) * Wq_M_GLOBAL * X_N_GLOBAL);
    K_host             = (__nv_bfloat16*)malloc(sizeof(__nv_bfloat16) * Wk_M_GLOBAL * X_N_GLOBAL);
    K_host_cublas      = (__nv_bfloat16*)malloc(sizeof(__nv_bfloat16) * Wk_M_GLOBAL * X_N_GLOBAL);
    V_host             = (__nv_bfloat16*)malloc(sizeof(__nv_bfloat16) * Wv_M_GLOBAL * X_N_GLOBAL);
    V_host_cublas      = (__nv_bfloat16*)malloc(sizeof(__nv_bfloat16) * Wv_M_GLOBAL * X_N_GLOBAL);
    O_host             = (__nv_bfloat16*)malloc(sizeof(__nv_bfloat16) * Wq_M_GLOBAL * X_N_GLOBAL);
    O_host_baseline    = (__nv_bfloat16*)malloc(sizeof(__nv_bfloat16) * Wq_M_GLOBAL * X_N_GLOBAL);

    Q_host_cpu         = (__nv_bfloat16*)malloc(sizeof(__nv_bfloat16) * Wq_M_GLOBAL * X_N_GLOBAL);
    K_host_cpu         = (__nv_bfloat16*)malloc(sizeof(__nv_bfloat16) * Wk_M_GLOBAL * X_N_GLOBAL);
    V_host_cpu         = (__nv_bfloat16*)malloc(sizeof(__nv_bfloat16) * Wv_M_GLOBAL * X_N_GLOBAL);

    cudaMalloc(reinterpret_cast<void**>(&Wq_device), sizeof(__nv_bfloat16) * Wq_M_GLOBAL * Wq_N_GLOBAL);
    cudaMalloc(reinterpret_cast<void**>(&Wk_device), sizeof(__nv_bfloat16) * Wk_M_GLOBAL * Wk_N_GLOBAL);
    cudaMalloc(reinterpret_cast<void**>(&Wv_device), sizeof(__nv_bfloat16) * Wv_M_GLOBAL * Wv_N_GLOBAL);
    cudaMalloc(reinterpret_cast<void**>(&X_device), sizeof(__nv_bfloat16) * X_M_GLOBAL * X_N_GLOBAL);
    cudaMalloc(reinterpret_cast<void**>(&X_Transposed_device), sizeof(__nv_bfloat16) * X_N_GLOBAL * X_M_GLOBAL);
    cudaMalloc(reinterpret_cast<void**>(&Q_device), sizeof(__nv_bfloat16) * Wq_M_GLOBAL * X_N_GLOBAL);
    cudaMalloc(reinterpret_cast<void**>(&Q_device_cublas), sizeof(__nv_bfloat16) * Wq_M_GLOBAL * X_N_GLOBAL);
    cudaMalloc(reinterpret_cast<void**>(&K_device_cublas), sizeof(__nv_bfloat16) * Wk_M_GLOBAL * X_N_GLOBAL);
    cudaMalloc(reinterpret_cast<void**>(&V_device_cublas), sizeof(__nv_bfloat16) * Wv_M_GLOBAL * X_N_GLOBAL);
    cudaMalloc(reinterpret_cast<void**>(&K_device), sizeof(__nv_bfloat16) * Wk_M_GLOBAL * X_N_GLOBAL);
    cudaMalloc(reinterpret_cast<void**>(&V_device), sizeof(__nv_bfloat16) * Wv_M_GLOBAL * X_N_GLOBAL);
    cudaMalloc(reinterpret_cast<void**>(&O_device), sizeof(__nv_bfloat16) * Wq_M_GLOBAL * X_N_GLOBAL);
    init_host_matrices_bf16(Wq_host, X_host, Wq_M_GLOBAL, Wq_N_GLOBAL, X_N_GLOBAL);
    init_host_matrices_bf16_one(Wk_host, Wk_M_GLOBAL, Wk_N_GLOBAL);
    init_host_matrices_bf16_one(Wv_host, Wv_M_GLOBAL, Wv_N_GLOBAL);

    for (int i = 0; i < X_M_GLOBAL; i++)
        for (int j = 0; j < X_N_GLOBAL; j++)
            X_Transposed_host[i * X_N_GLOBAL + j] = X_host[i + j * X_M_GLOBAL];

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

    int Wq_num_global_tiles = InitBF16MatrixTripleBitmap_Host(
        Wq_host, Wq_M_GLOBAL, Wq_N_GLOBAL, 
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
    int Wq_high_freq_count = Wq_TileOffsets_global_cpu[Wq_num_global_tiles * 2];
    int Wq_full_count = Wq_TileOffsets_global_cpu[Wq_num_global_tiles * 2 + 1];
    size_t Wq_original_size = Wq_M_GLOBAL * Wq_N_GLOBAL * sizeof(__nv_bfloat16);
    size_t Wq_compressed_size = 
        // (7 * sizeof(__nv_bfloat16)) +                    // High-freq exponents
        (Wq_high_freq_count * sizeof(uint8_t)) +            // High-freq elements (sign+mantissa)
        (Wq_full_count * sizeof(__nv_bfloat16)) +           // Non-high-freq elements
        (Wq_num_tiles * sizeof(uint64_t) * 3) +             // Three bitmaps
        // (Wq_num_tiles * 2 * sizeof(int)) +                  // Small tile offsets
        (Wq_num_median_tiles * 2 * sizeof(int)) +           // Medium tile offsets
        ((Wq_num_global_tiles + 1) * 2 * sizeof(int));      // Global tile offsets
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
    cudaMalloc(&Wq_compressed_full_gpu, Wq_full_count * sizeof(__nv_bfloat16));
    cudaMalloc(&Wq_sign_mantissa_gpu, Wq_high_freq_count * sizeof(uint8_t));
    cudaMalloc(&Wq_bitmap1_gpu, Wq_num_tiles * sizeof(uint64_t));
    cudaMalloc(&Wq_bitmap2_gpu, Wq_num_tiles * sizeof(uint64_t));
    cudaMalloc(&Wq_bitmap3_gpu, Wq_num_tiles * sizeof(uint64_t));
    cudaMalloc(&Wq_TileOffsets_gpu, Wq_num_tiles * 2 * sizeof(int));
    cudaMalloc(&Wq_TileOffsets_median_gpu, Wq_num_median_tiles * 2 * sizeof(int));
    cudaMalloc(&Wq_TileOffsets_global_gpu, (Wq_num_global_tiles + 1) * 2 * sizeof(int));
    int* Wq_max_high_freq_gpu = nullptr;
    int* Wq_max_full_gpu = nullptr;
    cudaMalloc(&Wq_max_high_freq_gpu, sizeof(int));
    cudaMalloc(&Wq_max_full_gpu, sizeof(int));
    printf("Copying compressed data to GPU...\n");
    // Copy compressed data to device
    cudaMemcpy(Wq_top_exponents_gpu, Wq_top_exponents_cpu, 7 * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(Wq_compressed_full_gpu, Wq_compressed_full_cpu, Wq_full_count * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(Wq_sign_mantissa_gpu, Wq_sign_mantissa_cpu, Wq_high_freq_count * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(Wq_bitmap1_gpu, Wq_bitmap1_cpu, Wq_num_tiles * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(Wq_bitmap2_gpu, Wq_bitmap2_cpu, Wq_num_tiles * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(Wq_bitmap3_gpu, Wq_bitmap3_cpu, Wq_num_tiles * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(Wq_TileOffsets_gpu, Wq_TileOffsets_cpu, Wq_num_tiles * 2 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(Wq_TileOffsets_median_gpu, Wq_TileOffsets_median_cpu, Wq_num_median_tiles * 2 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(Wq_TileOffsets_global_gpu, Wq_TileOffsets_global_cpu, (Wq_num_global_tiles + 1) * 2 * sizeof(int), cudaMemcpyHostToDevice);
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
    int Wk_num_global_tiles = InitBF16MatrixTripleBitmap_Host(
        Wk_host, Wk_M_GLOBAL, Wk_N_GLOBAL, 
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
    int Wk_high_freq_count = Wk_TileOffsets_global_cpu[Wk_num_global_tiles * 2];
    int Wk_full_count = Wk_TileOffsets_global_cpu[Wk_num_global_tiles * 2 + 1];
    size_t Wk_original_size = Wk_M_GLOBAL * Wk_N_GLOBAL * sizeof(__nv_bfloat16);
    size_t Wk_compressed_size = 
        // (7 * sizeof(__nv_bfloat16)) +                    // High-freq exponents
        (Wk_high_freq_count * sizeof(uint8_t)) +            // High-freq elements (sign+mantissa)
        (Wk_full_count * sizeof(__nv_bfloat16)) +           // Non-high-freq elements
        (Wk_num_tiles * sizeof(uint64_t) * 3) +             // Three bitmaps
        // (Wk_num_tiles * 2 * sizeof(int)) +                  // Small tile offsets
        (Wk_num_median_tiles * 2 * sizeof(int)) +           // Medium tile offsets
        ((Wk_num_global_tiles + 1) * 2 * sizeof(int));      // Global tile offsets
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
    cudaMalloc(&Wk_compressed_full_gpu, Wk_full_count * sizeof(__nv_bfloat16));
    cudaMalloc(&Wk_sign_mantissa_gpu, Wk_high_freq_count * sizeof(uint8_t));
    cudaMalloc(&Wk_bitmap1_gpu, Wk_num_tiles * sizeof(uint64_t));
    cudaMalloc(&Wk_bitmap2_gpu, Wk_num_tiles * sizeof(uint64_t));
    cudaMalloc(&Wk_bitmap3_gpu, Wk_num_tiles * sizeof(uint64_t));
    cudaMalloc(&Wk_TileOffsets_gpu, Wk_num_tiles * 2 * sizeof(int));
    cudaMalloc(&Wk_TileOffsets_median_gpu, Wk_num_median_tiles * 2 * sizeof(int));
    cudaMalloc(&Wk_TileOffsets_global_gpu, (Wk_num_global_tiles + 1) * 2 * sizeof(int));
    int* Wk_high_freq_gpu = nullptr;
    int* Wk_full_gpu = nullptr;
    cudaMalloc(&Wk_high_freq_gpu, sizeof(int));
    cudaMalloc(&Wk_full_gpu, sizeof(int));
    printf("Copying compressed data to GPU...\n");
    // Copy compressed data to device
    cudaMemcpy(Wk_top_exponents_gpu, Wk_top_exponents_cpu, 7 * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(Wk_compressed_full_gpu, Wk_compressed_full_cpu, Wk_full_count * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(Wk_sign_mantissa_gpu, Wk_sign_mantissa_cpu, Wk_high_freq_count * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(Wk_bitmap1_gpu, Wk_bitmap1_cpu, Wk_num_tiles * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(Wk_bitmap2_gpu, Wk_bitmap2_cpu, Wk_num_tiles * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(Wk_bitmap3_gpu, Wk_bitmap3_cpu, Wk_num_tiles * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(Wk_TileOffsets_gpu, Wk_TileOffsets_cpu, Wk_num_tiles * 2 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(Wk_TileOffsets_median_gpu, Wk_TileOffsets_median_cpu, Wk_num_median_tiles * 2 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(Wk_TileOffsets_global_gpu, Wk_TileOffsets_global_cpu, (Wk_num_global_tiles + 1) * 2 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(Wk_high_freq_gpu, &Wk_high_freq_count, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(Wk_full_gpu, &Wk_full_count, sizeof(int), cudaMemcpyHostToDevice);
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

    // Wv
    __nv_bfloat16* Wv_top_exponents_cpu = nullptr;
    __nv_bfloat16* Wv_compressed_full_cpu = nullptr;
    uint8_t* Wv_sign_mantissa_cpu = nullptr;
    uint64_t* Wv_bitmap1_cpu = nullptr;
    uint64_t* Wv_bitmap2_cpu = nullptr;
    uint64_t* Wv_bitmap3_cpu = nullptr;
    int* Wv_TileOffsets_cpu = nullptr;
    int* Wv_TileOffsets_median_cpu = nullptr;
    int* Wv_TileOffsets_global_cpu = nullptr;
    int Wv_max_high_freq_count = 0;
    int Wv_max_full_count = 0;
    uint8_t Wv_start_exp = 0;
    int Wv_num_global_tiles = InitBF16MatrixTripleBitmap_Host(
        Wv_host, Wv_M_GLOBAL, Wv_N_GLOBAL, 
        8, 16, 64, 8, 64, 64,
        &Wv_top_exponents_cpu, &Wv_compressed_full_cpu, &Wv_sign_mantissa_cpu,
        &Wv_bitmap1_cpu, &Wv_bitmap2_cpu, &Wv_bitmap3_cpu,
        &Wv_TileOffsets_cpu, &Wv_TileOffsets_median_cpu, &Wv_TileOffsets_global_cpu,
        Wv_max_high_freq_count, Wv_max_full_count, Wv_start_exp);
    int Wv_tile_m = 8;
    int Wv_tile_k = 8;
    int Wv_tile_m_global = 64;
    int Wv_tile_k_global = 64;
    int Wv_num_tiles_m = Wv_M_GLOBAL / Wv_tile_m;
    int Wv_num_tiles_k = Wv_N_GLOBAL / Wv_tile_k;
    int Wv_num_tiles = Wv_num_tiles_m * Wv_num_tiles_k;
    int Wv_num_median_tiles_m = Wv_M_GLOBAL / 16;
    int Wv_num_median_tiles_k = Wv_N_GLOBAL / 64;
    int Wv_num_median_tiles = Wv_num_median_tiles_m * Wv_num_median_tiles_k;
    int Wv_high_freq_count = Wv_TileOffsets_global_cpu[Wv_num_global_tiles * 2];
    int Wv_full_count = Wv_TileOffsets_global_cpu[Wv_num_global_tiles * 2 + 1];
    size_t Wv_original_size = Wv_M_GLOBAL * Wv_N_GLOBAL * sizeof(__nv_bfloat16);
    size_t Wv_compressed_size = 
        // (7 * sizeof(__nv_bfloat16)) +                    // High-freq exponents
        (Wv_high_freq_count * sizeof(uint8_t)) +            // High-freq elements (sign+mantissa)
        (Wv_full_count * sizeof(__nv_bfloat16)) +           // Non-high-freq elements
        (Wv_num_tiles * sizeof(uint64_t) * 3) +             // Three bitmaps
        // (Wv_num_tiles * 2 * sizeof(int)) +                  // Small tile offsets
        (Wv_num_median_tiles * 2 * sizeof(int)) +           // Medium tile offsets
        ((Wv_num_global_tiles + 1) * 2 * sizeof(int));      // Global tile offsets
    float Wv_compression_ratio = (float)Wv_original_size / Wv_compressed_size;
    __nv_bfloat16* Wv_top_exponents_gpu = nullptr;
    __nv_bfloat16* Wv_compressed_full_gpu = nullptr;
    uint8_t* Wv_sign_mantissa_gpu = nullptr;
    uint64_t* Wv_bitmap1_gpu = nullptr;
    uint64_t* Wv_bitmap2_gpu = nullptr;
    uint64_t* Wv_bitmap3_gpu = nullptr;
    int* Wv_TileOffsets_gpu = nullptr;
    int* Wv_TileOffsets_median_gpu = nullptr;
    int* Wv_TileOffsets_global_gpu = nullptr;
    cudaMalloc(&Wv_top_exponents_gpu, 7 * sizeof(__nv_bfloat16)); // 7 high-freq exponents
    cudaMalloc(&Wv_compressed_full_gpu, Wv_full_count * sizeof(__nv_bfloat16));
    cudaMalloc(&Wv_sign_mantissa_gpu, Wv_high_freq_count * sizeof(uint8_t));
    cudaMalloc(&Wv_bitmap1_gpu, Wv_num_tiles * sizeof(uint64_t));
    cudaMalloc(&Wv_bitmap2_gpu, Wv_num_tiles * sizeof(uint64_t));
    cudaMalloc(&Wv_bitmap3_gpu, Wv_num_tiles * sizeof(uint64_t));
    cudaMalloc(&Wv_TileOffsets_gpu, Wv_num_tiles * 2 * sizeof(int));
    cudaMalloc(&Wv_TileOffsets_median_gpu, Wv_num_median_tiles * 2 * sizeof(int));
    cudaMalloc(&Wv_TileOffsets_global_gpu, (Wv_num_global_tiles + 1) * 2 * sizeof(int));
    int* Wv_high_freq_gpu = nullptr;
    int* Wv_full_gpu = nullptr;
    cudaMalloc(&Wv_high_freq_gpu, sizeof(int));
    cudaMalloc(&Wv_full_gpu, sizeof(int));
    printf("Copying compressed data to GPU...\n");
    // Copy compressed data to device
    cudaMemcpy(Wv_top_exponents_gpu, Wv_top_exponents_cpu, 7 * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(Wv_compressed_full_gpu, Wv_compressed_full_cpu, Wv_full_count * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(Wv_sign_mantissa_gpu, Wv_sign_mantissa_cpu, Wv_high_freq_count * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(Wv_bitmap1_gpu, Wv_bitmap1_cpu, Wv_num_tiles * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(Wv_bitmap2_gpu, Wv_bitmap2_cpu, Wv_num_tiles * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(Wv_bitmap3_gpu, Wv_bitmap3_cpu, Wv_num_tiles * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(Wv_TileOffsets_gpu, Wv_TileOffsets_cpu, Wv_num_tiles * 2 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(Wv_TileOffsets_median_gpu, Wv_TileOffsets_median_cpu, Wv_num_median_tiles * 2 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(Wv_TileOffsets_global_gpu, Wv_TileOffsets_global_cpu, (Wv_num_global_tiles + 1) * 2 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(Wv_high_freq_gpu, &Wv_high_freq_count, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(Wv_full_gpu, &Wv_full_count, sizeof(int), cudaMemcpyHostToDevice);
    // Free host compressed data memory
    free(Wv_top_exponents_cpu);
    free(Wv_compressed_full_cpu);
    free(Wv_sign_mantissa_cpu);
    free(Wv_bitmap1_cpu);
    free(Wv_bitmap2_cpu);
    free(Wv_bitmap3_cpu);
    free(Wv_TileOffsets_cpu);
    free(Wv_TileOffsets_median_cpu);
    free(Wv_TileOffsets_global_cpu);

    // // Q=Wq*X
    // // cpu
    // for(int i=0; i<Wq_M_GLOBAL;i++)
    // {
    //     for(int j=0;j<X_N_GLOBAL;j++)
    //     {
    //         Q_host_cpu[i*X_N_GLOBAL+j] = __float2bfloat16(0.0f);
    //         for(int k=0;k<Wq_N_GLOBAL;k++)
    //         {
    //             Q_host_cpu[i*X_N_GLOBAL+j] += Wq_host[i*Wq_N_GLOBAL+k] * X_host[j*X_M_GLOBAL+k];
    //         }
    //     }
    // }

    // zipserv
    BF16TripleBitmap_MM_API(
        0,
        Wq_sign_mantissa_gpu,Wq_compressed_full_gpu,
        Wq_bitmap1_gpu,Wq_bitmap2_gpu,Wq_bitmap3_gpu,
        Wq_TileOffsets_median_gpu,Wq_TileOffsets_global_gpu,
        Wq_max_high_freq_count,Wq_max_full_count,
        Wq_start_exp,
        X_device,
        Q_device,
        Wq_M_GLOBAL, X_N_GLOBAL, Wq_N_GLOBAL,
        nullptr,
        1);

    // Q_host = (__nv_bfloat16*)malloc(sizeof(__nv_bfloat16) * Wq_M_GLOBAL * X_N_GLOBAL);
    // cudaMemcpy(Q_host, Q_device, sizeof(__nv_bfloat16) * Wq_M_GLOBAL * X_N_GLOBAL, cudaMemcpyDeviceToHost); 
    // // cublas
    // cublasHandle_t handle_Q;
    // cublasCreate(&handle_Q);
    // cublasSetStream(handle_Q, 0);
    // cublasSetMathMode(handle_Q, CUBLAS_PEDANTIC_MATH);
    // cudaDeviceSynchronize();
    // int              m = Wq_M_GLOBAL, n = X_N_GLOBAL, k = Wq_N_GLOBAL;
    // const float      alpha     = 1.0;
    // const float      beta      = 0.0;
    // cublasGemmAlgo_t CuBlasALG_Q = static_cast<cublasGemmAlgo_t>(0);
    // cublasGemmEx(
    //     handle_Q,
    //     CUBLAS_OP_T,
    //     CUBLAS_OP_N,
    //     m,
    //     n,
    //     k,
    //     &alpha,
    //     Wq_device,
    //     CUDA_R_16BF,
    //     k,
    //     X_device,
    //     CUDA_R_16BF,
    //     k,
    //     &beta,
    //     Q_device_cublas,
    //     CUDA_R_16BF,
    //     m,
    //     CUDA_R_32F,
    //     CuBlasALG_Q);
    // Q_host_cublas = (__nv_bfloat16*)malloc(sizeof(__nv_bfloat16) * Wq_M_GLOBAL * X_N_GLOBAL);
    // cudaMemcpy(Q_host_cublas, Q_device_cublas, sizeof(__nv_bfloat16) * Wq_M_GLOBAL * X_N_GLOBAL, cudaMemcpyDeviceToHost); 
    // compare results
    // double max_abs_error_Q=ComputeTotalError_BF16(Q_host_cublas, Q_host, Wq_M_GLOBAL, X_N_GLOBAL);
    // printf("Q Matrix - Max Absolute Error: %lf\n", max_abs_error_Q);
    // // K=Wk*X
    // // cpu
    // for(int i=0; i<Wk_M_GLOBAL;i++)
    // {
    //     for(int j=0;j<X_N_GLOBAL;j++)
    //     {
    //         K_host_cpu[i*X_N_GLOBAL+j] = __float2bfloat16(0.0f);
    //         for(int k=0;k<Wk_N_GLOBAL;k++)
    //         {
    //             K_host_cpu[i*X_N_GLOBAL+j] += Wk_host[i*Wk_N_GLOBAL+k] * X_host[j*X_M_GLOBAL+k];
    //         }
    //     }
    // }

    // zipserv
    BF16TripleBitmap_MM_API(
        0,
        Wk_sign_mantissa_gpu,Wk_compressed_full_gpu,
        Wk_bitmap1_gpu,Wk_bitmap2_gpu,Wk_bitmap3_gpu,
        Wk_TileOffsets_median_gpu,Wk_TileOffsets_global_gpu,
        Wk_max_high_freq_count,Wk_max_full_count,
        Wk_start_exp,
        X_device,
        K_device,
        Wk_M_GLOBAL, X_N_GLOBAL, Wk_N_GLOBAL,
        nullptr,
        1);

    // K_host = (__nv_bfloat16*)malloc(sizeof(__nv_bfloat16) * Wk_M_GLOBAL * X_N_GLOBAL);
    // cudaMemcpy(K_host, K_device, sizeof(__nv_bfloat16) * Wk_M_GLOBAL * X_N_GLOBAL, cudaMemcpyDeviceToHost); 
    // // cublas
    // cublasHandle_t handle_K;
    // cublasCreate(&handle_K);
    // cublasSetStream(handle_K, 0);
    // cublasSetMathMode(handle_K, CUBLAS_PEDANTIC_MATH);
    // cudaDeviceSynchronize();
    // m = Wk_M_GLOBAL; 
    // n = X_N_GLOBAL; 
    // k = Wk_N_GLOBAL;
    // cublasGemmAlgo_t CuBlasALG_K = static_cast<cublasGemmAlgo_t>(0);
    // cublasGemmEx(
    //     handle_K,
    //     CUBLAS_OP_T,
    //     CUBLAS_OP_N,
    //     m,
    //     n,
    //     k,
    //     &alpha,
    //     Wk_device,
    //     CUDA_R_16BF,
    //     k,
    //     X_device,
    //     CUDA_R_16BF,
    //     k,
    //     &beta,
    //     K_device_cublas,
    //     CUDA_R_16BF,
    //     m,
    //     CUDA_R_32F,
    //     CuBlasALG_K);
    // K_host_cublas = (__nv_bfloat16*)malloc(sizeof(__nv_bfloat16) * Wk_M_GLOBAL * X_N_GLOBAL);
    // cudaMemcpy(K_host_cublas, K_device_cublas, sizeof(__nv_bfloat16) * Wk_M_GLOBAL * X_N_GLOBAL, cudaMemcpyDeviceToHost); 
    // compare results
    // double max_abs_error_K=ComputeTotalError_BF16(K_host_cublas, K_host, Wk_M_GLOBAL, X_N_GLOBAL);
    // printf("K Matrix - Max Absolute Error: %lf\n", max_abs_error_K);
    // // V=Wv*X
    // // cpu
    // for(int i=0; i<Wv_M_GLOBAL;i++)
    // {
    //     for(int j=0;j<X_N_GLOBAL;j++)
    //     {
    //         V_host_cpu[i*X_N_GLOBAL+j] = __float2bfloat16(0.0f);
    //         for(int k=0;k<Wv_N_GLOBAL;k++)
    //         {
    //             V_host_cpu[i*X_N_GLOBAL+j] += Wv_host[i*Wv_N_GLOBAL+k] * X_host[j*X_M_GLOBAL+k];
    //         }
    //     }
    // }

    // zipserv
    BF16TripleBitmap_MM_API(
        0,
        Wv_sign_mantissa_gpu,Wv_compressed_full_gpu,
        Wv_bitmap1_gpu,Wv_bitmap2_gpu,Wv_bitmap3_gpu,
        Wv_TileOffsets_median_gpu,Wv_TileOffsets_global_gpu,
        Wv_max_high_freq_count,Wv_max_full_count,
        Wv_start_exp,
        X_device,
        V_device,
        Wv_M_GLOBAL, X_N_GLOBAL, Wv_N_GLOBAL,
        nullptr,
        1);

    // V_host = (__nv_bfloat16*)malloc(sizeof(__nv_bfloat16) * Wv_M_GLOBAL * X_N_GLOBAL);
    // cudaMemcpy(V_host, V_device, sizeof(__nv_bfloat16) * Wv_M_GLOBAL * X_N_GLOBAL, cudaMemcpyDeviceToHost); 
    // // cublas
    // cublasHandle_t handle_V;
    // cublasCreate(&handle_V);
    // cublasSetStream(handle_V, 0);
    // cublasSetMathMode(handle_V, CUBLAS_PEDANTIC_MATH);
    // cudaDeviceSynchronize();
    // m = Wv_M_GLOBAL;
    // n = X_N_GLOBAL;
    // k = Wv_N_GLOBAL;
    // cublasGemmAlgo_t CuBlasALG_V = static_cast<cublasGemmAlgo_t>(0);
    // cublasGemmEx(
    //     handle_V,
    //     CUBLAS_OP_T,
    //     CUBLAS_OP_N,
    //     m,
    //     n,
    //     k,
    //     &alpha,
    //     Wv_device,
    //     CUDA_R_16BF,
    //     k,
    //     X_device,
    //     CUDA_R_16BF,
    //     k,
    //     &beta,
    //     V_device_cublas,
    //     CUDA_R_16BF,
    //     m,
    //     CUDA_R_32F,
    //     CuBlasALG_V);
    // V_host_cublas = (__nv_bfloat16*)malloc(sizeof(__nv_bfloat16) * Wv_M_GLOBAL * X_N_GLOBAL);
    // cudaMemcpy(V_host_cublas, V_device_cublas, sizeof(__nv_bfloat16) * Wv_M_GLOBAL * X_N_GLOBAL, cudaMemcpyDeviceToHost); 
    // compare results
    // double max_abs_error_V=ComputeTotalError_BF16(V_host_cublas, V_host, Wv_M_GLOBAL, X_N_GLOBAL);
    // printf("V Matrix - Max Absolute Error: %lf\n", max_abs_error_V);

    cudaDeviceSynchronize();

    dim3 gridDim(Wq_M_GLOBAL/kBlockM, X_M_GLOBAL/HeadDim, 1);
    dim3 blockDim(32*4,1,1);
    // int shared_mem_size = kBlockM * HeadDim * sizeof(__nv_bfloat16)*5;
    int shared_mem_size = (kBlockM * HeadDim * sizeof(__nv_bfloat16));         // V
    shared_mem_size += (kBlockM * HeadDim * sizeof(__nv_bfloat16));            // K
    shared_mem_size += (kBlockM * HeadDim * sizeof(__nv_bfloat16)*2);          // prepare Q double buffer
    shared_mem_size += (sizeof(uint64_t)*64*3*2);                              // prepare Q 3 bitmaps double buffer
    shared_mem_size += sizeof(__nv_bfloat16)*Wq_max_high_freq_count*2;         // prepare Q full value double buffer
    shared_mem_size += sizeof(uint8_t)*Wq_max_full_count*2;                    // prepare Q high-freq sign+mantissa double buffer
    // shared_mem_size += max(sizeof(__nv_bfloat16)*Wq_max_high_freq_count*2,
    //                     sizeof(__nv_bfloat16)*Wk_max_high_freq_count*2);    
    // shared_mem_size += max(sizeof(uint8_t)*Wq_max_full_count*2,
    //                     sizeof(uint8_t)*Wk_max_full_count*2);               
    int shared_mem_maxsize = 0;        
    CUDA_CHECK(cudaDeviceGetAttribute(&shared_mem_maxsize,cudaDevAttrMaxSharedMemoryPerBlock,0));
    shared_mem_size = min(shared_mem_size, shared_mem_maxsize);
    printf("%d KB shared memory per block\n", shared_mem_size / 1024);

    // fa2
    if(mode==0)
    {
        for (int i = 0; i < WARM_UP_ITERATION; i++) 
        {
            BF16TripleBitmap_MM_API(
                0,
                Wq_sign_mantissa_gpu,Wq_compressed_full_gpu,
                Wq_bitmap1_gpu,Wq_bitmap2_gpu,Wq_bitmap3_gpu,
                Wq_TileOffsets_median_gpu,Wq_TileOffsets_global_gpu,
                Wq_max_high_freq_count,Wq_max_full_count,
                Wq_start_exp,
                X_device,
                Q_device,
                Wq_M_GLOBAL, X_N_GLOBAL, Wq_N_GLOBAL,
                nullptr,
                1);

            // BF16TripleBitmap_MM_API(
            //     0,
            //     Wk_sign_mantissa_gpu,Wk_compressed_full_gpu,
            //     Wk_bitmap1_gpu,Wk_bitmap2_gpu,Wk_bitmap3_gpu,
            //     Wk_TileOffsets_median_gpu,Wk_TileOffsets_global_gpu,
            //     Wk_max_high_freq_count,Wk_max_full_count,
            //     Wk_start_exp,
            //     X_device,
            //     K_device,
            //     Wk_M_GLOBAL, X_N_GLOBAL, Wk_N_GLOBAL,
            //     nullptr,
            //     1);

            compute_attn_v2<<<gridDim, blockDim, shared_mem_size>>>(
                O_device, 
                Q_device, 
                K_device,
                V_device,
                Wq_M_GLOBAL,
                Wk_M_GLOBAL,
                Wv_M_GLOBAL,
                Wq_M_GLOBAL,
                X_N_GLOBAL,
                0.125f);
        }

        float total_milliseconds_compute_attn = 0.0f;
        for (int i = 0; i < BENCHMARK_ITERATION; i++) 
        {
            // Flush L2 cache before each iteration to simulate real-world cold cache scenario
            flush_l2_cache();
            
            // Measure only the GEMM operation time, excluding cache flush overhead
            cudaEventRecord(start);

            BF16TripleBitmap_MM_API(
                0,
                Wq_sign_mantissa_gpu,Wq_compressed_full_gpu,
                Wq_bitmap1_gpu,Wq_bitmap2_gpu,Wq_bitmap3_gpu,
                Wq_TileOffsets_median_gpu,Wq_TileOffsets_global_gpu,
                Wq_max_high_freq_count,Wq_max_full_count,
                Wq_start_exp,
                X_device,
                Q_device,
                Wq_M_GLOBAL, X_N_GLOBAL, Wq_N_GLOBAL,
                nullptr,
                1);

            // BF16TripleBitmap_MM_API(
            //     0,
            //     Wk_sign_mantissa_gpu,Wk_compressed_full_gpu,
            //     Wk_bitmap1_gpu,Wk_bitmap2_gpu,Wk_bitmap3_gpu,
            //     Wk_TileOffsets_median_gpu,Wk_TileOffsets_global_gpu,
            //     Wk_max_high_freq_count,Wk_max_full_count,
            //     Wk_start_exp,
            //     X_device,
            //     K_device,
            //     Wk_M_GLOBAL, X_N_GLOBAL, Wk_N_GLOBAL,
            //     nullptr,
            //     1);

            compute_attn_v2<<<gridDim, blockDim, shared_mem_size>>>(
                O_device, 
                Q_device, 
                K_device,
                V_device,
                Wq_M_GLOBAL,
                Wk_M_GLOBAL,
                Wv_M_GLOBAL,
                Wq_M_GLOBAL,
                X_N_GLOBAL,
                0.125f);
                
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            checkLastCudaError(__LINE__);
            float iter_time = 0.0f;
            cudaEventElapsedTime(&iter_time, start, stop);
            total_milliseconds_compute_attn += iter_time;
        }
            
        float milliseconds_compute_attn = total_milliseconds_compute_attn / BENCHMARK_ITERATION;
        printf("Average compute_attn_v2 execution time over %d iterations: %f ms\n", BENCHMARK_ITERATION, milliseconds_compute_attn);
        
        cudaDeviceSynchronize();
            CUDA_CHECK(cudaMemcpy(O_host, O_device, sizeof(__nv_bfloat16) * X_N_GLOBAL * Wq_M_GLOBAL, cudaMemcpyDeviceToHost)); 

        std::memcpy(O_host_baseline, O_host, sizeof(__nv_bfloat16) * Wq_M_GLOBAL * X_N_GLOBAL);
        cudaDeviceSynchronize();
    }
    else if(mode == 1)
    {
        for (int i = 0; i < WARM_UP_ITERATION; i++) 
        {
            BF16TripleBitmap_MM_API(
                0,
                Wq_sign_mantissa_gpu,Wq_compressed_full_gpu,
                Wq_bitmap1_gpu,Wq_bitmap2_gpu,Wq_bitmap3_gpu,
                Wq_TileOffsets_median_gpu,Wq_TileOffsets_global_gpu,
                Wq_max_high_freq_count,Wq_max_full_count,
                Wq_start_exp,
                X_device,
                Q_device,
                Wq_M_GLOBAL, X_N_GLOBAL, Wq_N_GLOBAL,
                nullptr,
                1);

            BF16TripleBitmap_MM_API(
                0,
                Wk_sign_mantissa_gpu,Wk_compressed_full_gpu,
                Wk_bitmap1_gpu,Wk_bitmap2_gpu,Wk_bitmap3_gpu,
                Wk_TileOffsets_median_gpu,Wk_TileOffsets_global_gpu,
                Wk_max_high_freq_count,Wk_max_full_count,
                Wk_start_exp,
                X_device,
                K_device,
                Wk_M_GLOBAL, X_N_GLOBAL, Wk_N_GLOBAL,
                nullptr,
                1);

            compute_attn_v2<<<gridDim, blockDim, shared_mem_size>>>(
                O_device, 
                Q_device, 
                K_device,
                V_device,
                Wq_M_GLOBAL,
                Wk_M_GLOBAL,
                Wv_M_GLOBAL,
                Wq_M_GLOBAL,
                X_N_GLOBAL,
                0.125f);
        }

        float total_milliseconds_compute_attn = 0.0f;
        for (int i = 0; i < BENCHMARK_ITERATION; i++) 
        {
            // Flush L2 cache before each iteration to simulate real-world cold cache scenario
            flush_l2_cache();
            
            // Measure only the GEMM operation time, excluding cache flush overhead
            cudaEventRecord(start);

            BF16TripleBitmap_MM_API(
                0,
                Wq_sign_mantissa_gpu,Wq_compressed_full_gpu,
                Wq_bitmap1_gpu,Wq_bitmap2_gpu,Wq_bitmap3_gpu,
                Wq_TileOffsets_median_gpu,Wq_TileOffsets_global_gpu,
                Wq_max_high_freq_count,Wq_max_full_count,
                Wq_start_exp,
                X_device,
                Q_device,
                Wq_M_GLOBAL, X_N_GLOBAL, Wq_N_GLOBAL,
                nullptr,
                1);

            BF16TripleBitmap_MM_API(
                0,
                Wk_sign_mantissa_gpu,Wk_compressed_full_gpu,
                Wk_bitmap1_gpu,Wk_bitmap2_gpu,Wk_bitmap3_gpu,
                Wk_TileOffsets_median_gpu,Wk_TileOffsets_global_gpu,
                Wk_max_high_freq_count,Wk_max_full_count,
                Wk_start_exp,
                X_device,
                K_device,
                Wk_M_GLOBAL, X_N_GLOBAL, Wk_N_GLOBAL,
                nullptr,
                1);

            compute_attn_v2<<<gridDim, blockDim, shared_mem_size>>>(
                O_device, 
                Q_device, 
                K_device,
                V_device,
                Wq_M_GLOBAL,
                Wk_M_GLOBAL,
                Wv_M_GLOBAL,
                Wq_M_GLOBAL,
                X_N_GLOBAL,
                0.125f);
                
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            checkLastCudaError(__LINE__);
            float iter_time = 0.0f;
            cudaEventElapsedTime(&iter_time, start, stop);
            total_milliseconds_compute_attn += iter_time;
        }
            
        float milliseconds_compute_attn = total_milliseconds_compute_attn / BENCHMARK_ITERATION;
        printf("Average compute_attn_v2 execution time over %d iterations: %f ms\n", BENCHMARK_ITERATION, milliseconds_compute_attn);
        
        cudaDeviceSynchronize();
            CUDA_CHECK(cudaMemcpy(O_host, O_device, sizeof(__nv_bfloat16) * X_N_GLOBAL * Wq_M_GLOBAL, cudaMemcpyDeviceToHost)); 

        std::memcpy(O_host_baseline, O_host, sizeof(__nv_bfloat16) * Wq_M_GLOBAL * X_N_GLOBAL);
        cudaDeviceSynchronize();
    }
    else if(mode ==2)
    {
        for (int i = 0; i < WARM_UP_ITERATION; i++) 
        {

            BF16TripleBitmap_MM_API(
                0,
                Wk_sign_mantissa_gpu,Wk_compressed_full_gpu,
                Wk_bitmap1_gpu,Wk_bitmap2_gpu,Wk_bitmap3_gpu,
                Wk_TileOffsets_median_gpu,Wk_TileOffsets_global_gpu,
                Wk_max_high_freq_count,Wk_max_full_count,
                Wk_start_exp,
                X_device,
                K_device,
                Wk_M_GLOBAL, X_N_GLOBAL, Wk_N_GLOBAL,
                nullptr,
                1);

            compute_attn_v2<<<gridDim, blockDim, shared_mem_size>>>(
                O_device, 
                Q_device, 
                K_device,
                V_device,
                Wq_M_GLOBAL,
                Wk_M_GLOBAL,
                Wv_M_GLOBAL,
                Wq_M_GLOBAL,
                X_N_GLOBAL,
                0.125f);
        }

        float total_milliseconds_compute_attn = 0.0f;
        for (int i = 0; i < BENCHMARK_ITERATION; i++) 
        {
            // Flush L2 cache before each iteration to simulate real-world cold cache scenario
            flush_l2_cache();
            
            // Measure only the GEMM operation time, excluding cache flush overhead
            cudaEventRecord(start);

            BF16TripleBitmap_MM_API(
                0,
                Wk_sign_mantissa_gpu,Wk_compressed_full_gpu,
                Wk_bitmap1_gpu,Wk_bitmap2_gpu,Wk_bitmap3_gpu,
                Wk_TileOffsets_median_gpu,Wk_TileOffsets_global_gpu,
                Wk_max_high_freq_count,Wk_max_full_count,
                Wk_start_exp,
                X_device,
                K_device,
                Wk_M_GLOBAL, X_N_GLOBAL, Wk_N_GLOBAL,
                nullptr,
                1);

            compute_attn_v2<<<gridDim, blockDim, shared_mem_size>>>(
                O_device, 
                Q_device, 
                K_device,
                V_device,
                Wq_M_GLOBAL,
                Wk_M_GLOBAL,
                Wv_M_GLOBAL,
                Wq_M_GLOBAL,
                X_N_GLOBAL,
                0.125f);
                
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            checkLastCudaError(__LINE__);
            float iter_time = 0.0f;
            cudaEventElapsedTime(&iter_time, start, stop);
            total_milliseconds_compute_attn += iter_time;
        }
            
        float milliseconds_compute_attn = total_milliseconds_compute_attn / BENCHMARK_ITERATION;
        printf("Average compute_attn_v2 execution time over %d iterations: %f ms\n", BENCHMARK_ITERATION, milliseconds_compute_attn);
        
        cudaDeviceSynchronize();
            CUDA_CHECK(cudaMemcpy(O_host, O_device, sizeof(__nv_bfloat16) * X_N_GLOBAL * Wq_M_GLOBAL, cudaMemcpyDeviceToHost)); 

        std::memcpy(O_host_baseline, O_host, sizeof(__nv_bfloat16) * Wq_M_GLOBAL * X_N_GLOBAL);
        cudaDeviceSynchronize();
    }

    // zipserv_fa2
    if(mode == 0)
    {
        for (int i = 0; i < WARM_UP_ITERATION; i++) 
        {
            compute_attn_v2_zipserv<<<gridDim, blockDim, shared_mem_size>>>(
                O_device, 
                K_device, V_device, 
                Wk_N_GLOBAL, Wv_N_GLOBAL,  X_N_GLOBAL,
                X_N_GLOBAL, 
                0.125f,
                Wq_sign_mantissa_gpu,
                Wq_compressed_full_gpu,
                Wq_bitmap1_gpu,
                Wq_bitmap2_gpu,
                Wq_bitmap3_gpu,
                Wq_TileOffsets_median_gpu,
                Wq_TileOffsets_global_gpu,
                Wq_max_high_freq_count,
                Wq_max_full_count,
                Wq_start_exp,
                Wq_M_GLOBAL,
                X_N_GLOBAL,
                Wq_N_GLOBAL,                                                               
                X_device);
        }

        flush_l2_cache();

        float total_milliseconds_compute_attn_zipserv = 0.0f;
        for (int i = 0; i < BENCHMARK_ITERATION; i++) 
        {
            flush_l2_cache();
            cudaEventRecord(start);
            compute_attn_v2_zipserv<<<gridDim, blockDim, shared_mem_size>>>(
                O_device, 
                K_device, V_device, 
                Wk_N_GLOBAL, Wv_N_GLOBAL,  X_N_GLOBAL,
                X_N_GLOBAL, 
                0.125f,
                Wq_sign_mantissa_gpu,
                Wq_compressed_full_gpu,
                Wq_bitmap1_gpu,
                Wq_bitmap2_gpu,
                Wq_bitmap3_gpu,
                Wq_TileOffsets_median_gpu,
                Wq_TileOffsets_global_gpu,
                Wq_max_high_freq_count,
                Wq_max_full_count,
                Wq_start_exp,
                Wq_M_GLOBAL,
                X_N_GLOBAL,
                Wq_N_GLOBAL,                                                               
                X_device);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float iter_time = 0.0f;
            cudaEventElapsedTime(&iter_time, start, stop);
            total_milliseconds_compute_attn_zipserv += iter_time;
        }
        float milliseconds_compute_attn_zipserv = total_milliseconds_compute_attn_zipserv / BENCHMARK_ITERATION;
        printf("Average compute_attn_v2_zipserv execution time over %d iterations: %f ms\n", BENCHMARK_ITERATION, milliseconds_compute_attn_zipserv);
        
        cudaDeviceSynchronize();

        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(O_host, O_device, sizeof(__nv_bfloat16) * X_N_GLOBAL * Wq_M_GLOBAL, cudaMemcpyDeviceToHost)); 
    }
    else if(mode == 1)
    {
        for (int i = 0; i < WARM_UP_ITERATION; i++) 
        {
            compute_attn_v2_zipserv_prepareQK<<<gridDim, blockDim, shared_mem_size>>>(
                O_device, 
                V_device, 
                Wk_N_GLOBAL, Wv_N_GLOBAL,  X_N_GLOBAL,
                X_N_GLOBAL, 
                0.125f,
                Wq_sign_mantissa_gpu,
                Wq_compressed_full_gpu,
                Wq_bitmap1_gpu,
                Wq_bitmap2_gpu,
                Wq_bitmap3_gpu,
                Wq_TileOffsets_median_gpu,
                Wq_TileOffsets_global_gpu,
                Wq_max_high_freq_count,
                Wq_max_full_count,
                Wq_start_exp,
                Wq_M_GLOBAL,
                X_N_GLOBAL,
                Wq_N_GLOBAL,
                Wk_sign_mantissa_gpu,
                Wk_compressed_full_gpu,
                Wk_bitmap1_gpu,
                Wk_bitmap2_gpu,
                Wk_bitmap3_gpu,
                Wk_TileOffsets_median_gpu,
                Wk_TileOffsets_global_gpu,
                Wk_max_high_freq_count,
                Wk_max_full_count,
                Wk_start_exp,
                Wk_M_GLOBAL,
                X_N_GLOBAL,
                Wk_N_GLOBAL,                                                                
                X_device);
        }

        flush_l2_cache();

        float total_milliseconds_compute_attn_zipserv = 0.0f;
        for (int i = 0; i < BENCHMARK_ITERATION; i++) 
        {
            flush_l2_cache();
            cudaEventRecord(start);
            compute_attn_v2_zipserv_prepareQK<<<gridDim, blockDim, shared_mem_size>>>(
                O_device, 
                V_device, 
                Wk_N_GLOBAL, Wv_N_GLOBAL,  X_N_GLOBAL,
                X_N_GLOBAL, 
                0.125f,
                Wq_sign_mantissa_gpu,
                Wq_compressed_full_gpu,
                Wq_bitmap1_gpu,
                Wq_bitmap2_gpu,
                Wq_bitmap3_gpu,
                Wq_TileOffsets_median_gpu,
                Wq_TileOffsets_global_gpu,
                Wq_max_high_freq_count,
                Wq_max_full_count,
                Wq_start_exp,
                Wq_M_GLOBAL,
                X_N_GLOBAL,
                Wq_N_GLOBAL,
                Wk_sign_mantissa_gpu,
                Wk_compressed_full_gpu,
                Wk_bitmap1_gpu,
                Wk_bitmap2_gpu,
                Wk_bitmap3_gpu,
                Wk_TileOffsets_median_gpu,
                Wk_TileOffsets_global_gpu,
                Wk_max_high_freq_count,
                Wk_max_full_count,
                Wk_start_exp,
                Wk_M_GLOBAL,
                X_N_GLOBAL,
                Wk_N_GLOBAL,                                                                
                X_device);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float iter_time = 0.0f;
            cudaEventElapsedTime(&iter_time, start, stop);
            total_milliseconds_compute_attn_zipserv += iter_time;
        }
        float milliseconds_compute_attn_zipserv_prepareQK = total_milliseconds_compute_attn_zipserv / BENCHMARK_ITERATION;
        printf("Average compute_attn_v2_zipserv_prepareQK execution time over %d iterations: %f ms\n", BENCHMARK_ITERATION, milliseconds_compute_attn_zipserv_prepareQK);
        
        cudaDeviceSynchronize();

        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(O_host, O_device, sizeof(__nv_bfloat16) * X_N_GLOBAL * Wq_M_GLOBAL, cudaMemcpyDeviceToHost)); 
    }
    else if(mode == 2)
    {
        for (int i = 0; i < WARM_UP_ITERATION; i++) 
        {
            compute_attn_v2_zipserv_prepareK<<<gridDim, blockDim, shared_mem_size>>>(
                O_device, 
                Q_device, V_device, 
                Wq_N_GLOBAL, Wv_N_GLOBAL,  X_N_GLOBAL,
                X_N_GLOBAL, 
                0.125f,
                Wk_sign_mantissa_gpu,
                Wk_compressed_full_gpu,
                Wk_bitmap1_gpu,
                Wk_bitmap2_gpu,
                Wk_bitmap3_gpu,
                Wk_TileOffsets_median_gpu,
                Wk_TileOffsets_global_gpu,
                Wk_max_high_freq_count,
                Wk_max_full_count,
                Wk_start_exp,
                Wk_M_GLOBAL,
                X_N_GLOBAL,
                Wk_N_GLOBAL,                                                                
                X_device);
        }

        flush_l2_cache();

        float total_milliseconds_compute_attn_zipserv = 0.0f;
        for (int i = 0; i < BENCHMARK_ITERATION; i++) 
        {
            flush_l2_cache();
            cudaEventRecord(start);
            compute_attn_v2_zipserv_prepareK<<<gridDim, blockDim, shared_mem_size>>>(
                O_device, 
                Q_device, V_device, 
                Wq_N_GLOBAL, Wv_N_GLOBAL,  X_N_GLOBAL,
                X_N_GLOBAL, 
                0.125f,
                Wk_sign_mantissa_gpu,
                Wk_compressed_full_gpu,
                Wk_bitmap1_gpu,
                Wk_bitmap2_gpu,
                Wk_bitmap3_gpu,
                Wk_TileOffsets_median_gpu,
                Wk_TileOffsets_global_gpu,
                Wk_max_high_freq_count,
                Wk_max_full_count,
                Wk_start_exp,
                Wk_M_GLOBAL,
                X_N_GLOBAL,
                Wk_N_GLOBAL,                                                                
                X_device);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float iter_time = 0.0f;
            cudaEventElapsedTime(&iter_time, start, stop);
            total_milliseconds_compute_attn_zipserv += iter_time;
        }
        float milliseconds_compute_attn_zipserv_prepareK = total_milliseconds_compute_attn_zipserv / BENCHMARK_ITERATION;
        printf("Average compute_attn_v2_zipserv_prepareK execution time over %d iterations: %f ms\n", BENCHMARK_ITERATION, milliseconds_compute_attn_zipserv_prepareK);
        
        cudaDeviceSynchronize();

        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(O_host, O_device, sizeof(__nv_bfloat16) * X_N_GLOBAL * Wq_M_GLOBAL, cudaMemcpyDeviceToHost)); 
    }


    // print_bf16_matrix("Output O zipserv (compute_attn_v2_zipserv)", O_host, Wq_M_GLOBAL, X_N_GLOBAL);

    double max_abs_error_O = 0.0;
    double mean_abs_error_O = 0.0;
    double ref_l2_sq = 0.0;
    double diff_l2_sq = 0.0;
    int max_err_idx = 0;
    int total_O_elems = Wq_M_GLOBAL * X_N_GLOBAL;
    int count_abs_gt_1e3 = 0;
    int count_abs_gt_1e4 = 0;
    std::vector<double> abs_err_list;
    abs_err_list.reserve(total_O_elems);
    for (int i = 0; i < total_O_elems; ++i) {
        float ref_v = __bfloat162float(O_host_baseline[i]);
        float zip_v = __bfloat162float(O_host[i]);
        double diff = (double)ref_v - (double)zip_v;
        double abs_err = fabs(diff);
        mean_abs_error_O += abs_err;
        ref_l2_sq += (double)ref_v * (double)ref_v;
        diff_l2_sq += diff * diff;
        abs_err_list.push_back(abs_err);
        if (abs_err > 1e-3) ++count_abs_gt_1e3;
        if (abs_err > 1e-4) ++count_abs_gt_1e4;
        if (abs_err > max_abs_error_O) {
            max_abs_error_O = abs_err;
            max_err_idx = i;
        }
    }
    mean_abs_error_O /= total_O_elems;

    std::sort(abs_err_list.begin(), abs_err_list.end());
    auto percentile_value = [&](double p) {
        int idx = static_cast<int>(p * (total_O_elems - 1));
        return abs_err_list[idx];
    };
    double p50_abs_error_O = percentile_value(0.50);
    double p90_abs_error_O = percentile_value(0.90);
    double p99_abs_error_O = percentile_value(0.99);

    double rel_l2_error_O = 0.0;
    if (ref_l2_sq > 0.0) 
    {
        rel_l2_error_O = sqrt(diff_l2_sq / ref_l2_sq);
    }

    int max_err_row = max_err_idx / X_N_GLOBAL;
    int max_err_col = max_err_idx % X_N_GLOBAL;
    float ref_max = __bfloat162float(O_host_baseline[max_err_idx]);
    float zip_max = __bfloat162float(O_host[max_err_idx]);
    printf("O baseline vs zipserv: max_abs_error=%lf, mean_abs_error=%lf\n", max_abs_error_O, mean_abs_error_O);
    printf("O error stats: rel_l2=%le, p50_abs=%le, p90_abs=%le, p99_abs=%le\n",
            rel_l2_error_O, p50_abs_error_O, p90_abs_error_O, p99_abs_error_O);
    printf("O error counts: abs>1e-3: %d/%d, abs>1e-4: %d/%d\n",
            count_abs_gt_1e3, total_O_elems, count_abs_gt_1e4, total_O_elems);
    printf("Max error at (row=%d, col=%d): baseline=%f zipserv=%f diff=%f\n",
            max_err_row, max_err_col, ref_max, zip_max, ref_max - zip_max);

    CUDA_CHECK(cudaFree(Wq_device));
    CUDA_CHECK(cudaFree(Wk_device));
    CUDA_CHECK(cudaFree(Wv_device));
    CUDA_CHECK(cudaFree(X_device));
    CUDA_CHECK(cudaFree(X_Transposed_device));
    CUDA_CHECK(cudaFree(Q_device));
    CUDA_CHECK(cudaFree(Q_device_cublas));
    CUDA_CHECK(cudaFree(K_device));
    CUDA_CHECK(cudaFree(K_device_cublas));
    CUDA_CHECK(cudaFree(V_device));
    CUDA_CHECK(cudaFree(V_device_cublas));
    CUDA_CHECK(cudaFree(O_device));

    CUDA_CHECK(cudaFree(Wq_top_exponents_gpu));
    CUDA_CHECK(cudaFree(Wq_compressed_full_gpu));
    CUDA_CHECK(cudaFree(Wq_sign_mantissa_gpu));
    CUDA_CHECK(cudaFree(Wq_bitmap1_gpu));
    CUDA_CHECK(cudaFree(Wq_bitmap2_gpu));
    CUDA_CHECK(cudaFree(Wq_bitmap3_gpu));
    CUDA_CHECK(cudaFree(Wq_TileOffsets_gpu));
    CUDA_CHECK(cudaFree(Wq_TileOffsets_median_gpu));
    CUDA_CHECK(cudaFree(Wq_TileOffsets_global_gpu));
    CUDA_CHECK(cudaFree(Wq_max_high_freq_gpu));
    CUDA_CHECK(cudaFree(Wq_max_full_gpu));

    CUDA_CHECK(cudaFree(Wk_top_exponents_gpu));
    CUDA_CHECK(cudaFree(Wk_compressed_full_gpu));
    CUDA_CHECK(cudaFree(Wk_sign_mantissa_gpu));
    CUDA_CHECK(cudaFree(Wk_bitmap1_gpu));
    CUDA_CHECK(cudaFree(Wk_bitmap2_gpu));
    CUDA_CHECK(cudaFree(Wk_bitmap3_gpu));
    CUDA_CHECK(cudaFree(Wk_TileOffsets_gpu));
    CUDA_CHECK(cudaFree(Wk_TileOffsets_median_gpu));
    CUDA_CHECK(cudaFree(Wk_TileOffsets_global_gpu));
    CUDA_CHECK(cudaFree(Wk_high_freq_gpu));
    CUDA_CHECK(cudaFree(Wk_full_gpu));

    CUDA_CHECK(cudaFree(Wv_top_exponents_gpu));
    CUDA_CHECK(cudaFree(Wv_compressed_full_gpu));
    CUDA_CHECK(cudaFree(Wv_sign_mantissa_gpu));
    CUDA_CHECK(cudaFree(Wv_bitmap1_gpu));
    CUDA_CHECK(cudaFree(Wv_bitmap2_gpu));
    CUDA_CHECK(cudaFree(Wv_bitmap3_gpu));
    CUDA_CHECK(cudaFree(Wv_TileOffsets_gpu));
    CUDA_CHECK(cudaFree(Wv_TileOffsets_median_gpu));
    CUDA_CHECK(cudaFree(Wv_TileOffsets_global_gpu));
    CUDA_CHECK(cudaFree(Wv_high_freq_gpu));
    CUDA_CHECK(cudaFree(Wv_full_gpu));

    free(Wq_host);
    free(Wk_host);
    free(Wv_host);
    free(X_host);
    free(X_Transposed_host);

    free(Q_host);
    free(Q_host_cublas);
    free(K_host);
    free(K_host_cublas);
    free(V_host);
    free(V_host_cublas);
    free(O_host);
    free(O_host_baseline);

    free(Q_host_cpu);
    free(K_host_cpu);
    free(V_host_cpu);

    return 0;
}