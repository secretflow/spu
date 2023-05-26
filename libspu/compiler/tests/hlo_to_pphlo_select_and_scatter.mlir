// RUN: mlir-pphlo-opt --hlo-legalize-to-pphlo=input_vis_list=VIS_SECRET,VIS_PUBLIC,VIS_PUBLIC --split-input-file %s | FileCheck %s

func.func @main(%arg0: tensor<128x5x5x32xf32>, %arg1: tensor<128x4x4x32xf32>, %arg2: tensor<f32>) -> tensor<128x5x5x32xf32>   {
  // CHECK: %1 = "pphlo.select_and_scatter"(%arg0, %arg1, %0) ({
  // CHECK:   ^bb0(%arg3: tensor<!pphlo.sec<f32>>, %arg4: tensor<!pphlo.sec<f32>>):
  // CHECK:     %2 = "pphlo.greater_equal"(%arg3, %arg4) : (tensor<!pphlo.sec<f32>>, tensor<!pphlo.sec<f32>>) -> tensor<!pphlo.sec<i1>>
  // CHECK:     "pphlo.return"(%2) : (tensor<!pphlo.sec<i1>>) -> ()
  // CHECK:   }, {
  // CHECK:   ^bb0(%arg3: tensor<!pphlo.pub<f32>>, %arg4: tensor<!pphlo.sec<f32>>):
  // CHECK:     %2 = "pphlo.add"(%arg3, %arg4) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.sec<f32>>) -> tensor<!pphlo.sec<f32>>
  // CHECK:     "pphlo.return"(%2) : (tensor<!pphlo.sec<f32>>) -> ()
  // CHECK:   }) {padding = dense<0> : tensor<4x2xi64>, window_dimensions = dense<[1, 2, 2, 1]> : tensor<4xi64>, window_strides = dense<1> : tensor<4xi64>} : (tensor<128x5x5x32x!pphlo.sec<f32>>, tensor<128x4x4x32x!pphlo.pub<f32>>, tensor<!pphlo.sec<f32>>) -> tensor<128x5x5x32x!pphlo.sec<f32>>
  %0 = "stablehlo.select_and_scatter"(%arg0, %arg1, %arg2) ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
        %1 = "stablehlo.compare"(%arg3, %arg4) {comparison_direction = #stablehlo<comparison_direction GE>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
        "stablehlo.return"(%1) : (tensor<i1>) -> ()
    }, {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
        %1 = stablehlo.add %arg3, %arg4 : tensor<f32>
        "stablehlo.return"(%1) : (tensor<f32>) -> ()
    }) {padding = dense<0> : tensor<4x2xi64>, window_dimensions = dense<[1, 2, 2, 1]> : tensor<4xi64>, window_strides = dense<1> : tensor<4xi64>} : (tensor<128x5x5x32xf32>, tensor<128x4x4x32xf32>, tensor<f32>) -> tensor<128x5x5x32xf32>
  return %0 : tensor<128x5x5x32xf32>
}