// RUN: mlir-opt --convert-func-to-llvm %s | FileCheck %s

module {
  func.func @main() -> i32 {
    // CHECK: llvm.mlir.constant(42 : i32) : i32
    %v = arith.constant 42 : i32
    // CHECK: llvm.return %0 : i32
    return %v : i32
  }
}
