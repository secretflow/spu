// RUN: mlir-pphlo-opt --hlo-legalize-to-pphlo=input_vis_list=VIS_PUBLIC,VIS_PUBLIC %s --split-input-file  | FileCheck %s

func.func @main(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> (tensor<2x2xui8>) {
    // CHECK: "pphlo.constant"() {value = dense<1> : tensor<2x2xui8>} : () -> tensor<2x2x!pphlo.pub<ui8>>
    %0 = stablehlo.constant dense<1> : tensor<2x2xui8>
    // CHECK: %1 = "pphlo.sqrt"(%arg0) : (tensor<2x2x!pphlo.pub<f32>>) -> tensor<2x2x!pphlo.pub<f32>>
    %1 = "stablehlo.sqrt"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
    // CHECK: %2 = "pphlo.add"(%arg0, %arg1) : (tensor<2x2x!pphlo.pub<f32>>, tensor<2x2x!pphlo.pub<f32>>) -> tensor<2x2x!pphlo.pub<f32>>
    %2 = "stablehlo.add"(%arg0, %arg1) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    return %0 : tensor<2x2xui8>
}
