// RUN: hello-opt %s | FileCheck %s

// CHECK: ; ModuleID = 'LLVMDialectModule'
// CHECL: source_filename = "LLVMDialectModule"
func @main() {
    %0 = "hello.constant"() {value = dense<1.0> : tensor<2x3xf64>} : () -> tensor<2x3xf64>
    "hello.print"(%0) : (tensor<2x3xf64>) -> ()
    return
}
