// RUN: spu-opt --hlo-legalize-to-pphlo=input_vis_list=VIS_SECRET --lower-conversion-cast %s --split-input-file  | FileCheck %s

func.func @main(%arg1: tensor<1024x1xf32>) -> (tensor<1024xf32>) {
    %0 = "stablehlo.constant"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<f32>
    // CHECK: %1 = pphlo.convert %0 : (tensor<f32>) -> tensor<!pphlo.secret<f32>>
    // CHECK: %2 = pphlo.reduce(%arg0 init: %1) applies pphlo.add across dimensions = [1] : (tensor<1024x1x!pphlo.secret<f32>>, tensor<!pphlo.secret<f32>>) -> tensor<1024x!pphlo.secret<f32>>
    %1 = "stablehlo.reduce"(%arg1, %0) ( {
        ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):  // no predecessors
        %2 = "stablehlo.add"(%arg2, %arg3) : (tensor<f32>, tensor<f32>) -> tensor<f32>
        "stablehlo.return"(%2) : (tensor<f32>) -> ()
    }) {dimensions = array<i64: 1>} : (tensor<1024x1xf32>, tensor<f32>) -> tensor<1024xf32>
    return %1 :  tensor<1024xf32>
}

// -----

func.func @main(%arg0: tensor<3x2xi64>) -> tensor<2x2xi64>   {
  %0 = "stablehlo.constant"() {value = dense<0> : tensor<i64>} : () -> tensor<i64>
  // CHECK: %[[CONVERT:.+]] = pphlo.convert %0 : (tensor<i64>) -> tensor<!pphlo.secret<i64>>
  // CHECK: %[[PAD:.+]] = pphlo.pad %arg0, %[[CONVERT]], low = [2, 0], high = [1, 0], interior = [1, 0] : (tensor<3x2x!pphlo.secret<i64>>, tensor<!pphlo.secret<i64>>) -> tensor<8x2x!pphlo.secret<i64>>
  // CHECK: "pphlo.reduce_window"(%[[PAD]], %[[CONVERT]])
  %result = "stablehlo.reduce_window"(%arg0, %0) ({
    ^bb0(%arg2: tensor<i64>, %arg3: tensor<i64>):
      %1 = "stablehlo.add"(%arg2, %arg3) : (tensor<i64>, tensor<i64>) -> tensor<i64>
      "stablehlo.return"(%1) : (tensor<i64>) -> ()
  }) {
    window_dimensions = array<i64: 2, 1>,
    window_strides = array<i64: 4, 1>,
    base_dilations = array<i64: 2, 1>,
    window_dilations = array<i64: 3, 1>,
    padding = dense<[[2, 1], [0, 0]]> : tensor<2x2xi64>
  } : (tensor<3x2xi64>, tensor<i64>) -> tensor<2x2xi64>
return %result : tensor<2x2xi64>
}
