// RUN: mlir-opt %s \
// RUN: | mlir-opt -gpu-kernel-outlining \
// RUN: | mlir-opt -pass-pipeline='builtin.module(gpu.module(strip-debuginfo,convert-gpu-to-rocdl),rocdl-attach-target{chip=%chip})' \
// RUN: | mlir-opt -gpu-to-llvm -reconcile-unrealized-casts -gpu-module-to-binary \
// RUN: | mlir-runner \
// RUN:   --shared-libs=%mlir_rocm_runtime \
// RUN:   --shared-libs=%mlir_runner_utils \
// RUN:   --entry-point-result=void \
// RUN: | FileCheck %s

// CHECK: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
func.func @main() {
  %arg = memref.alloc() : memref<13xi32>
  %dst = memref.cast %arg : memref<13xi32> to memref<?xi32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %sx = memref.dim %dst, %c0 : memref<?xi32>
  %cast_dst = memref.cast %dst : memref<?xi32> to memref<*xi32>
  gpu.host_register %cast_dst : memref<*xi32>
  %dst_device = call @mgpuMemGetDeviceMemRef1dInt32(%dst) : (memref<?xi32>) -> (memref<?xi32>)
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %c1, %grid_y = %c1, %grid_z = %c1)
             threads(%tx, %ty, %tz) in (%block_x = %sx, %block_y = %c1, %block_z = %c1) {
    %t0 = arith.index_cast %tx : index to i32
    memref.store %t0, %dst_device[%tx] : memref<?xi32>
    gpu.terminator
  }
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %c1, %grid_y = %c1, %grid_z = %c1)
             threads(%tx, %ty, %tz) in (%block_x = %sx, %block_y = %c1, %block_z = %c1) {
    %t0 = arith.index_cast %tx : index to i32
    memref.store %t0, %dst_device[%tx] : memref<?xi32>
    gpu.terminator
  }
  call @printMemrefI32(%cast_dst) : (memref<*xi32>) -> ()
  return
}

func.func private @mgpuMemGetDeviceMemRef1dInt32(%ptr : memref<?xi32>) -> (memref<?xi32>)
func.func private @printMemrefI32(%ptr : memref<*xi32>)
