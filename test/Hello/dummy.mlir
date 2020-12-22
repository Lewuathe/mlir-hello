// RUN: hello-opt %s | FileCheck %s

// CHECK: ; ModuleID = 'LLVMDialectModule'
// CHECK-LABEL:  @bar
module {
    func @bar() {
        %0 = constant 1 : i32
        return
    }
}
