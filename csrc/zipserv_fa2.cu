// zipserv_fa2.cu
// 占位源文件，仅用于测试能否编译通过
#include "zipserv_fa2.cuh"

// 空的 kernel，仅保证能被编译
__global__ void dummy_kernel() {}

extern "C" void zipserv_fa2_dummy_export() {
    compute_attn<<<1, 1>>>(0, 0, 0, 0, 0, nullptr, nullptr, nullptr, nullptr, 0.0f, 0.0f);
}