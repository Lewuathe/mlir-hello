// RUN: hello-opt %s | hello-opt | FileCheck %s

func @main() {
    %0 = "hello.constant"() {value = dense<1.0> : tensor<2x3xf64>} : () -> tensor<2x3xf64>
    "hello.print"(%0) : (tensor<2x3xf64>) -> ()
    return
}
