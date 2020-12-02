# MLIR Sample Dialect

## Building

Please make sure to build LLVM project first according to [the instruction](https://mlir.llvm.org/getting_started/).

```sh
mkdir build && cd build
LLVM_DIR=/path/to/llvm-project/build/lib/cmake/llvm \
  MLIR_DIR=/path/to/llvm-project/build/lib/cmake/mlir \
  cmake -G Ninja ..

cmake --build . --target check-sample
```

To build the documentation from the TableGen description of the dialect operations, run
```sh
cmake --build . --target mlir-doc
```


