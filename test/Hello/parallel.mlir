// RUN: mlir-opt --convert-func-to-llvm --lower-affine --convert-scf-to-cf %s | FileCheck %s

// CHECK-LABEL: llvm.func @parallel_sum(%arg0: !llvm.ptr<i32>, %arg1: !llvm.ptr<i32>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64) -> i32 {
func.func @parallel_sum(%M: memref<10x10xi32>) -> i32 {
  %add0 = affine.parallel (%x, %y) = (0, 0) to (10, 10) reduce ("addi") -> i32 {
    %v = affine.load %M[%x, %y] : memref<10x10xi32>
    affine.yield %v : i32
  }

  return %add0 : i32
}
