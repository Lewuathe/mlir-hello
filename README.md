# MLIR Hello Dialect

This is the minimal example to look into the way to implement the hello-world kind of program with MLIR.

## Prerequisites

We need to build our own MLIR in the local machine in advance. Please follow the build instruction for MLIR [here](https://mlir.llvm.org/getting_started/.

## Building

Please make sure to build LLVM project first according to [the instruction](https://mlir.llvm.org/getting_started/).

```sh
mkdir build && cd build
LLVM_DIR=/path/to/llvm-project/build/lib/cmake/llvm \
  MLIR_DIR=/path/to/llvm-project/build/lib/cmake/mlir \
  cmake -G Ninja ..

cmake --build . --target hello-opt
```

To build the documentation from the TableGen description of the dialect operations, run
```sh
cmake --build . --target mlir-doc
```

## Execution

`hello-opt` will lower the MLIR into the bytecode of LLVM. 

```
# Lower MLIR to LLVM IR
$ ./build/bin/hello-opt ./test/Hello/print.mlir > /path/to/print.ll

# Execute the code with LL
$ lli /path/to/print.ll 

1.000000 1.000000 1.000000
1.000000 1.000000 1.000000
```

