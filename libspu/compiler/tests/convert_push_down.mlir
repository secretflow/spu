// RUN: mlir-pphlo-opt --convert-push-down --cse --split-input-file %s | FileCheck %s

func.func @main(%arg0: tensor<4x!pphlo.pub<i32>>, %arg1: tensor<2x2x!pphlo.pub<f32>>) -> (tensor<2x2x!pphlo.pub<f32>>) {
    // CHECK: %0 = "pphlo.reshape"(%arg0) : (tensor<4x!pphlo.pub<i32>>) -> tensor<2x2x!pphlo.pub<i32>>
    // CHECK: %1 = "pphlo.convert"(%0) : (tensor<2x2x!pphlo.pub<i32>>) -> tensor<2x2x!pphlo.pub<f32>>
    %0 = "pphlo.convert"(%arg0) : (tensor<4x!pphlo.pub<i32>>) -> tensor<4x!pphlo.pub<f32>>
    %1 = "pphlo.reshape"(%0) : (tensor<4x!pphlo.pub<f32>>) -> tensor<2x2x!pphlo.pub<f32>>
    %2 = "pphlo.multiply"(%1, %arg1) : (tensor<2x2x!pphlo.pub<f32>>, tensor<2x2x!pphlo.pub<f32>>) -> tensor<2x2x!pphlo.pub<f32>>
    return %2 : tensor<2x2x!pphlo.pub<f32>>
}

// -----

func.func @main(%arg0: tensor<2x3x!pphlo.pub<i32>>, %arg1: tensor<2x3x!pphlo.pub<f32>>) -> (tensor<3x3x!pphlo.pub<f32>>) {
    // CHECK: %0 = "pphlo.transpose"(%arg0)
    // CHECK: %1 = "pphlo.convert"(%0)
    %0 = "pphlo.convert"(%arg0) : (tensor<2x3x!pphlo.pub<i32>>) -> tensor<2x3x!pphlo.pub<f32>>
    %1 = "pphlo.transpose"(%0) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<2x3x!pphlo.pub<f32>>) -> tensor<3x2x!pphlo.pub<f32>>
    %2 = "pphlo.dot"(%1, %arg1) : (tensor<3x2x!pphlo.pub<f32>>, tensor<2x3x!pphlo.pub<f32>>) -> tensor<3x3x!pphlo.pub<f32>>
    return %2 : tensor<3x3x!pphlo.pub<f32>>
}

// -----
