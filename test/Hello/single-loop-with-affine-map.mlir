// RUN: mlir-opt --lower-affine --convert-scf-to-cf %s | FileCheck %s

#map1 = affine_map<(d0) -> (d0+1)>

module {
  func.func @main() {
    %A = memref.alloc() : memref<10xf32>

    affine.for %i = 0 to 10 {
      // CHECK: %{{.*}} = arith.constant 1 : index
      %j = affine.apply #map1(%i)
      %a = affine.load %A[%j] : memref<10xf32>
    }

    return
  }
}