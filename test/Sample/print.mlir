// RUN: sample-opt %s | sample-opt | FileCheck %s

func @main() {
    %0 = "sample.constant"() {value = dense<1.0> : tensor<2x3xf64>} : () -> tensor<2x3xf64>
    "sample.print"(%0) : (tensor<2x3xf64>) -> ()
    return
}
