// RUN: mlir-pphlo-opt --hlo-legalize-to-pphlo=input_vis_list=VIS_PUBLIC --split-input-file %s | FileCheck %s

func.func @main(%arg0: tensor<2x2xi32>) -> (tensor<2x2xf32>) {
    // CHECK:  %0 = "pphlo.convert"(%arg0) : (tensor<2x2x!pphlo.pub<i32>>) -> tensor<2x2x!pphlo.pub<f32>>
    %0 = "stablehlo.convert"(%arg0) : (tensor<2x2xi32>) -> tensor<2x2xf32>
    // CHECK:  "pphlo.sqrt"(%0) : (tensor<2x2x!pphlo.pub<f32>>) -> tensor<2x2x!pphlo.pub<f32>>
    %2 = "stablehlo.sqrt"(%0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
    // CHECK:  "pphlo.negate"(%0) : (tensor<2x2x!pphlo.pub<f32>>) -> tensor<2x2x!pphlo.pub<f32>>
    %3 = "stablehlo.negate"(%0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
    // CHECK:  "pphlo.exponential"(%0) : (tensor<2x2x!pphlo.pub<f32>>) -> tensor<2x2x!pphlo.pub<f32>>
    %4 = "stablehlo.exponential"(%0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
    // CHECK:  "pphlo.log_plus_one"(%0) : (tensor<2x2x!pphlo.pub<f32>>) -> tensor<2x2x!pphlo.pub<f32>>
    %5 = "stablehlo.log_plus_one"(%0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
    // CHECK:  "pphlo.floor"(%0) : (tensor<2x2x!pphlo.pub<f32>>) -> tensor<2x2x!pphlo.pub<f32>>
    %6 = "stablehlo.floor"(%0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
    // CHECK:  "pphlo.ceil"(%0) : (tensor<2x2x!pphlo.pub<f32>>) -> tensor<2x2x!pphlo.pub<f32>>
    %7 = "stablehlo.ceil"(%0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
    // CHECK:  "pphlo.abs"(%0) : (tensor<2x2x!pphlo.pub<f32>>) -> tensor<2x2x!pphlo.pub<f32>>
    %8 = "stablehlo.abs"(%0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
    // CHECK:  "pphlo.logistic"(%0) : (tensor<2x2x!pphlo.pub<f32>>) -> tensor<2x2x!pphlo.pub<f32>>
    %9 = "stablehlo.logistic"(%0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
    // CHECK:  "pphlo.not"(%arg0) : (tensor<2x2x!pphlo.pub<i32>>) -> tensor<2x2x!pphlo.pub<i32>>
    %10 = "stablehlo.not"(%arg0) : (tensor<2x2xi32>) -> tensor<2x2xi32>
    // CHECK:  return %0 : tensor<2x2x!pphlo.pub<f32>>
    return %0 : tensor<2x2xf32>
}
