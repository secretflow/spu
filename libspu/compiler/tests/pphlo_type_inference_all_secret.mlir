// RUN: mlir-pphlo-opt --hlo-legalize-to-pphlo=input_vis_list=VIS_SECRET,VIS_SECRET %s --split-input-file  | FileCheck %s

// CHECK: func @main(%arg0: tensor<2x2x!pphlo.sec<f32>>, %arg1: tensor<2x2x!pphlo.sec<f32>>) -> tensor<2x2x!pphlo.sec<f32>> {
func.func @main(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> (tensor<2x2xf32>) {
    // CHECK: %0 = "pphlo.sqrt"(%arg0) : (tensor<2x2x!pphlo.sec<f32>>) -> tensor<2x2x!pphlo.sec<f32>>
    %0 = "stablehlo.sqrt"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
    // CHECK: %1 = "pphlo.add"(%arg0, %arg1) : (tensor<2x2x!pphlo.sec<f32>>, tensor<2x2x!pphlo.sec<f32>>) -> tensor<2x2x!pphlo.sec<f32>>
    %1 = "stablehlo.add"(%arg0, %arg1) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    // CHECK: return %0 : tensor<2x2x!pphlo.sec<f32>>
    return %0 : tensor<2x2xf32>
}
