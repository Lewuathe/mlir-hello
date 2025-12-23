# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an MLIR (Multi-Level Intermediate Representation) dialect example implementing a minimal "hello-world" style demonstration. The project defines a custom MLIR dialect called "Hello" and demonstrates how to lower it to LLVM IR for execution.

The code structure is based on LLVM's standalone and Toy language examples. This project tracks the latest LLVM/MLIR compatibility through nightly builds.

## Build System

### Initial Setup

The project uses a git submodule for LLVM/MLIR. Before building, initialize submodules:

```bash
git submodule update --init --recursive
```

### Building LLVM/MLIR (prerequisite)

You must build LLVM/MLIR first. The project includes `build_and_run.sh` for automated setup:

```bash
./build_and_run.sh
```

Or manually:

```bash
# Build LLVM/MLIR
LLVM_REPO=./thirdparty/llvm-project
BUILD_DIR=$LLVM_REPO/build
mkdir $BUILD_DIR

cmake "-H$LLVM_REPO/llvm" "-B$BUILD_DIR" \
  -DLLVM_INSTALL_UTILS=ON \
  -DLLVM_ENABLE_PROJECTS="mlir;clang" \
  -DLLVM_INCLUDE_TOOLS=ON \
  -DLLVM_BUILD_EXAMPLES=ON \
  -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_RTTI=ON

cmake --build $BUILD_DIR --target check-mlir -j 10
cd $BUILD_DIR && make lli  # lli needed for testing
```

### Building mlir-hello

```bash
mkdir build && cd build

cmake -G Ninja .. \
  -DLLVM_DIR=./thirdparty/llvm-project/build/lib/cmake/llvm \
  -DMLIR_DIR=./thirdparty/llvm-project/build/lib/cmake/mlir

cmake --build . --target hello-opt
```

### Running Tests

```bash
# From build directory
cmake --build . --target check-hello
```

### Building Documentation

```bash
# From build directory
cmake --build . --target mlir-doc
```

## Development Commands

### Run the Compiler

```bash
# Lower MLIR to LLVM IR
./build/bin/hello-opt ./test/Hello/print.mlir > /tmp/print.ll

# Execute with LLVM interpreter
./thirdparty/llvm-project/build/bin/lli /tmp/print.ll
```

### Code Formatting

```bash
./clang-format-all
```

### Update LLVM Submodule

```bash
./update-llvm.sh
```

## Architecture

### Dialect Definition (TableGen)

The Hello dialect is defined using MLIR's TableGen DSL:

- **include/Hello/HelloDialect.td**: Dialect declaration and base operation class
- **include/Hello/HelloOps.td**: Operation definitions (ConstantOp, PrintOp, WorldOp)

### C++ Implementation

- **lib/Hello/HelloDialect.cpp**: Dialect registration and initialization
- **lib/Hello/HelloOps.cpp**: Operation implementations and builders
- **lib/Hello/LowerToAffine.cpp**: Pass to lower Hello dialect to Affine/MemRef dialects
- **lib/Hello/LowerToLLVM.cpp**: Pass to lower to LLVM dialect (final lowering)

### Compiler Tool

- **hello-opt/hello-opt.cpp**: Main compiler driver that:
  1. Loads MLIR input
  2. Applies lowering passes (LowerToAffine → LowerToLLVM)
  3. Converts to LLVM IR
  4. Outputs executable LLVM IR

### Lowering Pipeline

The compilation follows a two-stage lowering:

1. **Hello → Affine/MemRef**: Converts high-level Hello operations (hello-opt.cpp:115)
   - Tensor types → MemRef types
   - Allocates/deallocates memory for tensors

2. **Affine/MemRef → LLVM**: Final lowering to LLVM dialect (hello-opt.cpp:116)
   - PrintOp → printf calls to LLVM
   - MemRef operations → LLVM pointer operations

### Operations

The Hello dialect defines three operations:

1. **hello.constant**: Creates SSA values from tensor literals
2. **hello.print**: Prints tensor values (lowered to printf)
3. **hello.world**: Prints "Hello, World" string

## Testing

Tests are located in `test/Hello/` and use LLVM's lit testing framework:

- **print.mlir**: Basic print operation test
- **hello_world.mlir**: Hello world operation test
- **affine.mlir**, **loop-tiling.mlir**, etc.: Various lowering tests

Test files use FileCheck for verification via `// RUN:` and `// CHECK:` directives.

## Key Files to Understand

When modifying the dialect:

1. Start with TableGen definitions: `include/Hello/HelloOps.td`
2. Add C++ operation logic: `lib/Hello/HelloOps.cpp`
3. Implement lowering patterns: `lib/Hello/LowerToAffine.cpp` or `lib/Hello/LowerToLLVM.cpp`
4. Register passes: `include/Hello/HelloPasses.h`
5. Add tests: `test/Hello/*.mlir`

## Common Patterns

### Adding a New Operation

1. Define operation in `include/Hello/HelloOps.td`
2. Implement builders/verifiers in `lib/Hello/HelloOps.cpp`
3. Add lowering pattern in `lib/Hello/LowerToAffine.cpp` or `lib/Hello/LowerToLLVM.cpp`
4. Create test in `test/Hello/`
5. Run `check-hello` target to verify

### Debugging Lowering

Use MLIR's `-mlir-print-ir-after-all` flag to see IR after each pass:

```bash
./build/bin/hello-opt -mlir-print-ir-after-all ./test/Hello/print.mlir
```
