// RUN: mlir-opt --lower-affine --convert-scf-to-cf --convert-func-to-llvm --convert-cf-to-llvm --finalize-memref-to-llvm -reconcile-unrealized-casts %s | FileCheck %s

// CHECK: llvm.func @main() -> f32
module {
  func.func @main() -> f32 {
    %A = memref.alloc() : memref<2x3xf32>
    %B = memref.alloc() : memref<2x3xf32>

    %sum = arith.constant 1.0 : f32

    affine.for %i = 0 to 2 {
      affine.for %j = 0 to 3 {
         %a = affine.load %A[%i, %j] : memref<2x3xf32>
         %v = arith.addf %sum, %a : f32
         affine.store %v, %B[%i, %j] : memref<2x3xf32>
      }
    }

    %ret = affine.load %B[0, 0] : memref<2x3xf32>

    // CHECK:  llvm.return %{{.*}} : f32
    return %ret : f32
  }
}