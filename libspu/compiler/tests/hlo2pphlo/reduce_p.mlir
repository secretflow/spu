// RUN: mlir-pphlo-opt --hlo-legalize-to-pphlo=input_vis_list=VIS_PUBLIC --lower-conversion-cast %s --split-input-file  | FileCheck %s

// CHECK: func @main(%arg0: tensor<1024x1xf32>) -> tensor<1024xf32> {
func.func @main(%arg1: tensor<1024x1xf32>) -> (tensor<1024xf32>) {
    // CHECK: %0 = "pphlo.constant"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %0 = "stablehlo.constant"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<f32>
    // CHECK: %1 = "pphlo.reduce"(%arg0, %0) ({
    // CHECK:       ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
    // CHECK:        %2 = "pphlo.add"(%arg1, %arg2) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    // CHECK:        "pphlo.return"(%2) : (tensor<f32>) -> ()
    // CHECK: }) {dimensions = array<i64: 1>} : (tensor<1024x1xf32>, tensor<f32>) -> tensor<1024xf32>
    %1 = "stablehlo.reduce"(%arg1, %0) ( {
        ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):  // no predecessors
        %2 = "stablehlo.add"(%arg2, %arg3) : (tensor<f32>, tensor<f32>) -> tensor<f32>
        "stablehlo.return"(%2) : (tensor<f32>) -> ()
    }) {dimensions = dense<[1]> : tensor<1xi64>} : (tensor<1024x1xf32>, tensor<f32>) -> tensor<1024xf32>
    return %1 :  tensor<1024xf32>
}

// -----

func.func @main(%arg0: tensor<3x2xi64>) -> tensor<2x2xi64>   {
  %0 = "stablehlo.constant"() {value = dense<0> : tensor<i64>} : () -> tensor<i64>
  // CHECK:  %1 = "pphlo.pad"(%arg0, %0) {edge_padding_high = array<i64: 1, 0>, edge_padding_low = array<i64: 2, 0>, interior_padding = array<i64: 1, 0>} : (tensor<3x2xi64>, tensor<i64>) -> tensor<8x2xi64>
  // CHECK:  %2 = "pphlo.reduce_window"(%1, %0)
  %result = "stablehlo.reduce_window"(%arg0, %0) ({
    ^bb0(%arg2: tensor<i64>, %arg3: tensor<i64>):
      %1 = "stablehlo.add"(%arg2, %arg3) : (tensor<i64>, tensor<i64>) -> tensor<i64>
      "stablehlo.return"(%1) : (tensor<i64>) -> ()
  }) {
    window_dimensions = dense<[2, 1]> : tensor<2xi64>,
    window_strides = dense<[4, 1]> : tensor<2xi64>,
    base_dilations = dense<[2, 1]> : tensor<2xi64>,
    window_dilations = dense<[3, 1]> : tensor<2xi64>,
    padding = dense<[[2, 1], [0, 0]]> : tensor<2x2xi64>
  } : (tensor<3x2xi64>, tensor<i64>) -> tensor<2x2xi64>
return %result : tensor<2x2xi64>
}
