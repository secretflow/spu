// RUN: mlir-pphlo-opt --hlo-legalize-to-pphlo=input_vis_list=VIS_PUBLIC,VIS_PUBLIC --split-input-file %s | FileCheck %s

func.func @main(%arg0: tensor<3x2xi64>, %arg1: tensor<i64>) -> tensor<2x2xi64>   {
  // CHECK:  %0 = "pphlo.pad"(%arg0, %arg1) {edge_padding_high = dense<[1, 0]> : tensor<2xi64>, edge_padding_low = dense<[2, 0]> : tensor<2xi64>, interior_padding = dense<[1, 0]> : tensor<2xi64>} : (tensor<3x2x!pphlo.pub<i64>>, tensor<!pphlo.pub<i64>>) -> tensor<8x2x!pphlo.pub<i64>>
  // CHECK:  %1 = "pphlo.reduce_window"(%0, %arg1)
  %result = "stablehlo.reduce_window"(%arg0, %arg1) ({
    ^bb0(%arg2: tensor<i64>, %arg3: tensor<i64>):
      %0 = "stablehlo.add"(%arg2, %arg3) : (tensor<i64>, tensor<i64>) -> tensor<i64>
      "stablehlo.return"(%0) : (tensor<i64>) -> ()
  }) {
    window_dimensions = dense<[2, 1]> : tensor<2xi64>,
    window_strides = dense<[4, 1]> : tensor<2xi64>,
    base_dilations = dense<[2, 1]> : tensor<2xi64>,
    window_dilations = dense<[3, 1]> : tensor<2xi64>,
    padding = dense<[[2, 1], [0, 0]]> : tensor<2x2xi64>
  } : (tensor<3x2xi64>, tensor<i64>) -> tensor<2x2xi64>
return %result : tensor<2x2xi64>
}