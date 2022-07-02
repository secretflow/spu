// RUN: mlir-pphlo-opt --hlo-legalize-to-pphlo --split-input-file %s | FileCheck %s

func @main(%arg0: tensor<2x2xi32>) -> (tensor<2x2xf32>) {
    // CHECK:  %0 = "pphlo.convert"(%arg0) : (tensor<2x2x!pphlo.pub<i32>>) -> tensor<2x2x!pphlo.pub<f32>>
    %0 = "mhlo.convert"(%arg0) : (tensor<2x2xi32>) -> tensor<2x2xf32>
    // CHECK:  "pphlo.sqrt"(%0) : (tensor<2x2x!pphlo.pub<f32>>) -> tensor<2x2x!pphlo.pub<f32>>
    %2 = "mhlo.sqrt"(%0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
    // CHECK:  "pphlo.negate"(%0) : (tensor<2x2x!pphlo.pub<f32>>) -> tensor<2x2x!pphlo.pub<f32>>
    %3 = "mhlo.negate"(%0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
    // CHECK:  "pphlo.exponential"(%0) : (tensor<2x2x!pphlo.pub<f32>>) -> tensor<2x2x!pphlo.pub<f32>>
    %4 = "mhlo.exponential"(%0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
    // CHECK:  "pphlo.log_plus_one"(%0) : (tensor<2x2x!pphlo.pub<f32>>) -> tensor<2x2x!pphlo.pub<f32>>
    %5 = "mhlo.log_plus_one"(%0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
    // CHECK:  "pphlo.floor"(%0) : (tensor<2x2x!pphlo.pub<f32>>) -> tensor<2x2x!pphlo.pub<f32>>
    %6 = "mhlo.floor"(%0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
    // CHECK:  "pphlo.ceil"(%0) : (tensor<2x2x!pphlo.pub<f32>>) -> tensor<2x2x!pphlo.pub<f32>>
    %7 = "mhlo.ceil"(%0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
    // CHECK:  "pphlo.abs"(%0) : (tensor<2x2x!pphlo.pub<f32>>) -> tensor<2x2x!pphlo.pub<f32>>
    %8 = "mhlo.abs"(%0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
    // CHECK:  "pphlo.logistic"(%0) : (tensor<2x2x!pphlo.pub<f32>>) -> tensor<2x2x!pphlo.pub<f32>>
    %9 = "mhlo.logistic"(%0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
    // CHECK:  "pphlo.not"(%arg0) : (tensor<2x2x!pphlo.pub<i32>>) -> tensor<2x2x!pphlo.pub<i32>>
    %10 = "mhlo.not"(%arg0) : (tensor<2x2xi32>) -> tensor<2x2xi32>
    // CHECK:  return %0 : tensor<2x2x!pphlo.pub<f32>>
    return %0 : tensor<2x2xf32>
}
