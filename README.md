# zipserv+fa2

## Build

```bash
# Initialize environment
source Init.sh

# Build core library
cd build && make

# Build benchmarks
cd ../kernel_benchmark && source test_env && make
```

```bash
source Init.sh && cd kernel_benchmark && source test_env &&cd ..

cd build && make clean && make && cd ../kernel_benchmark && make clean && make && cd ..

cd kernel_benchmark && ./test_zipserv_fa2 1024 128 0 && cd ..
cd kernel_benchmark && nsys profile --trace=cuda,nvtx,osrt ./test_zipserv_fa2 1024 128 0 && cd ..

cd kernel_benchmark && ./autotest_zipserv_fa2.sh && cd ..
```
