// RUN: sample-opt %s | sample-opt | FileCheck %s

func @main()  {
    %0 = sample.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
    "sample.print"(%0) : (tensor<2x3xf64>) -> ()
    return
}