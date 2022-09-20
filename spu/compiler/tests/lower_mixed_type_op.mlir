// RUN: mlir-pphlo-opt --lower-mixed-type-op --split-input-file %s | FileCheck %s

func @mul(%arg0: tensor<2x2x!pphlo.pub<f32>>, %arg1: tensor<2x2x!pphlo.pub<i32>>) -> (tensor<2x2x!pphlo.pub<f32>>) {
    //CHECK: %0 = "pphlo.mixed_multiply"(%arg0, %arg1) : (tensor<2x2x!pphlo.pub<f32>>, tensor<2x2x!pphlo.pub<i32>>) -> tensor<2x2x!pphlo.pub<f32>>
    %0 = "pphlo.convert"(%arg1) : (tensor<2x2x!pphlo.pub<i32>>) -> tensor<2x2x!pphlo.pub<f32>>
    %1 = "pphlo.multiply"(%arg0, %0) : (tensor<2x2x!pphlo.pub<f32>>, tensor<2x2x!pphlo.pub<f32>>) -> tensor<2x2x!pphlo.pub<f32>>
    return %1 : tensor<2x2x!pphlo.pub<f32>>
}


func @dot(%arg0: tensor<2x2x!pphlo.pub<f32>>, %arg1: tensor<2x2x!pphlo.pub<i32>>) -> (tensor<2x2x!pphlo.pub<f32>>) {
    //CHECK: %0 = "pphlo.mixed_dot"(%arg0, %arg1) : (tensor<2x2x!pphlo.pub<f32>>, tensor<2x2x!pphlo.pub<i32>>) -> tensor<2x2x!pphlo.pub<f32>>
    %0 = "pphlo.convert"(%arg1) : (tensor<2x2x!pphlo.pub<i32>>) -> tensor<2x2x!pphlo.pub<f32>>
    %1 = "pphlo.dot"(%arg0, %0) : (tensor<2x2x!pphlo.pub<f32>>, tensor<2x2x!pphlo.pub<f32>>) -> tensor<2x2x!pphlo.pub<f32>>
    return %1 : tensor<2x2x!pphlo.pub<f32>>
}


func @dot_negative(%arg0: tensor<2x2x!pphlo.pub<f32>>, %arg1: tensor<2x2x!pphlo.pub<i32>>) -> (tensor<2x2x!pphlo.pub<i32>>) {
    //CHECK: %1 = "pphlo.dot"(%arg1, %0) : (tensor<2x2x!pphlo.pub<i32>>, tensor<2x2x!pphlo.pub<i32>>) -> tensor<2x2x!pphlo.pub<i32>>
    %0 = "pphlo.convert"(%arg0) : (tensor<2x2x!pphlo.pub<f32>>) -> tensor<2x2x!pphlo.pub<i32>>
    %1 = "pphlo.dot"(%arg1, %0) : (tensor<2x2x!pphlo.pub<i32>>, tensor<2x2x!pphlo.pub<i32>>) -> tensor<2x2x!pphlo.pub<i32>>
    return %1 : tensor<2x2x!pphlo.pub<i32>>
}
