// RUN: mlir-pphlo-opt --lower-mixed-type-op --split-input-file %s | FileCheck %s

func.func @main(%arg0: tensor<2x2x!pphlo.pub<f32>>, %arg1: tensor<2x2x!pphlo.pub<i32>>) -> (tensor<2x2x!pphlo.pub<f32>>) {
    //CHECK: %0 = "pphlo.multiply"(%arg0, %arg1) : (tensor<2x2x!pphlo.pub<f32>>, tensor<2x2x!pphlo.pub<i32>>) -> tensor<2x2x!pphlo.pub<f32>>
    %0 = "pphlo.convert"(%arg1) : (tensor<2x2x!pphlo.pub<i32>>) -> tensor<2x2x!pphlo.pub<f32>>
    %1 = "pphlo.multiply"(%arg0, %0) : (tensor<2x2x!pphlo.pub<f32>>, tensor<2x2x!pphlo.pub<f32>>) -> tensor<2x2x!pphlo.pub<f32>>
    return %1 : tensor<2x2x!pphlo.pub<f32>>
}

// -----

func.func @main(%arg0: tensor<2x2x!pphlo.pub<f32>>, %arg1: tensor<2x2x!pphlo.pub<i32>>) -> (tensor<2x2x!pphlo.pub<f32>>) {
    //CHECK: %0 = "pphlo.dot"(%arg0, %arg1) : (tensor<2x2x!pphlo.pub<f32>>, tensor<2x2x!pphlo.pub<i32>>) -> tensor<2x2x!pphlo.pub<f32>>
    %0 = "pphlo.convert"(%arg1) : (tensor<2x2x!pphlo.pub<i32>>) -> tensor<2x2x!pphlo.pub<f32>>
    %1 = "pphlo.dot"(%arg0, %0) : (tensor<2x2x!pphlo.pub<f32>>, tensor<2x2x!pphlo.pub<f32>>) -> tensor<2x2x!pphlo.pub<f32>>
    return %1 : tensor<2x2x!pphlo.pub<f32>>
}

// -----

func.func @main(%arg0: tensor<3x1x4x!pphlo.pub<f32>>, %arg1: tensor<3x4x5x!pphlo.pub<i32>>) -> (tensor<3x5x!pphlo.pub<f32>>) {
    //CHECK: %0 = "pphlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #pphlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<3x1x4x!pphlo.pub<f32>>, tensor<3x4x5x!pphlo.pub<i32>>) -> tensor<3x5x!pphlo.pub<f32>>
    %0 = "pphlo.convert"(%arg1) : (tensor<3x4x5x!pphlo.pub<i32>>) -> tensor<3x4x5x!pphlo.pub<f32>>
    %1 = "pphlo.dot_general"(%arg0, %0) {dot_dimension_numbers = #pphlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<3x1x4x!pphlo.pub<f32>>, tensor<3x4x5x!pphlo.pub<f32>>) -> tensor<3x5x!pphlo.pub<f32>>
    return %1 : tensor<3x5x!pphlo.pub<f32>>
}

// -----

func.func @main(%arg0: tensor<2x2x!pphlo.pub<f32>>, %arg1: tensor<2x2x!pphlo.pub<i32>>) -> (tensor<2x2x!pphlo.pub<i32>>) {
    //CHECK: %1 = "pphlo.dot"(%arg1, %0) : (tensor<2x2x!pphlo.pub<i32>>, tensor<2x2x!pphlo.pub<i32>>) -> tensor<2x2x!pphlo.pub<i32>>
    %0 = "pphlo.convert"(%arg0) : (tensor<2x2x!pphlo.pub<f32>>) -> tensor<2x2x!pphlo.pub<i32>>
    %1 = "pphlo.dot"(%arg1, %0) : (tensor<2x2x!pphlo.pub<i32>>, tensor<2x2x!pphlo.pub<i32>>) -> tensor<2x2x!pphlo.pub<i32>>
    return %1 : tensor<2x2x!pphlo.pub<i32>>
}
