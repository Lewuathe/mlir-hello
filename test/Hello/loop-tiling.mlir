// RUN: mlir-opt -affine-loop-tile="tile-size=32" %s  | FileCheck %s

// CHECK-DAG: [[$UB:#map[0-9]+]] = affine_map<(d0) -> (d0 + 32)>
// CHECK-DAG: [[$ID:#map[0-9]*]] = affine_map<(d0) -> (d0)>

// CHECK-LABEL: func @loop_tiling()
// CHECK-NEXT:   affine.for %{{.*}} = 0 to 256 step 32 {
// CHECK-NEXT:     affine.for %{{.*}} = 0 to 512 step 32 {
// CHECK-NEXT:       affine.for %[[I:.*]] = [[$ID]](%{{.*}}) to [[$UB]](%{{.*}}) {
// CHECK-NEXT:         affine.for %[[J:.*]] = [[$ID]](%{{.*}}) to [[$UB]](%{{.*}}) {
// CHECK-NEXT:             "test.foo"(%[[I]], %[[J]]) : (index, index) -> ()
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   }
func.func @loop_tiling() {
  affine.for %i = 0 to 256 {
    affine.for %j = 0 to 512 {
      "test.foo"(%i, %j) : (index, index) -> ()
    }
  }
  return
}
