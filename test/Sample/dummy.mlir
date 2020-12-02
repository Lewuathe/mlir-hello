// RUN: sample-opt %s | sample-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func @bar() {
        %0 = constant 1 : i32
        // CHECK: %{{.*}} = sample.foo %{{.*}} : i32
        %res = sample.foo %0 : i32
        return
    }
}
