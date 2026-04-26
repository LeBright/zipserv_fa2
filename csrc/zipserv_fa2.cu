// zipserv_fa2.cu
// 占位源文件，仅用于测试能否编译通过
#include "zipserv_fa2.cuh"

// 空的 kernel，仅保证能被编译
__global__ void dummy_kernel() {}

// 可选：导出一个符号，防止 .so 为空
extern "C" void zipserv_fa2_dummy_export() {}
