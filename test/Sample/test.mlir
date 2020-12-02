// RUN: sample-opt %s | sample-opt | FileCheck %s

llvm.func @main()  {
    %0 = constant 100 : i32
    // CHECK: %{{.*}} = sample.foo %{{.*}} : i32
    // %1 = sample.foo %0 : i32

    %2 = constant 1 : i1
    assert %2, "True"

    //%3 = llvm.constant(1.0 : f32) : !llvm.float
    //llvm.return %3 : !llvm.float
}