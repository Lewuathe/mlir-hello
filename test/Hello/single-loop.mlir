// RUN: mlir-opt --lower-affine --convert-scf-to-cf %s | FileCheck %s

// CHECK: module {
// CHECK:   func @main() {
// CHECK:     %{{.*}} = memref.alloc() : memref<3xf32>
// CHECK:     %{{.*}} = arith.constant 0 : index
// CHECK:     %{{.*}} = arith.constant 2 : index
// CHECK:     %{{.*}} = arith.constant 1 : index
// CHECK:     cf.br ^bb1(%{{.*}} : index)
// CHECK:   ^bb1(%{{.*}}: index):  // 2 preds: ^bb0, ^bb2
// CHECK:     %{{.*}} = arith.cmpi slt, %{{.*}}, %c2 : index
// CHECK:     cf.cond_br %{{.*}}, ^bb2, ^bb3
// CHECK:   ^bb2:  // pred: ^bb1
// CHECK:     %{{.*}} = memref.load %{{.*}}[%{{.*}}] : memref<3xf32>
// CHECK:     %{{.*}} = arith.addi %{{.*}}, %{{.*}} : index
// CHECK:     cf.br ^bb1(%{{.*}} : index)
// CHECK:   ^bb3:  // pred: ^bb1
// CHECK:     return
// CHECK:   }
// CHECK: }

module {
  func.func @main() {
    %A = memref.alloc() : memref<3xf32>

    affine.for %i = 0 to 2 {
      %a = affine.load %A[%i] : memref<3xf32>
    }

    return
  }
}