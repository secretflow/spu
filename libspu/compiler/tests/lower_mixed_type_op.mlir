// RUN: mlir-pphlo-opt --lower-mixed-type-op --split-input-file %s | FileCheck %s

func.func @main(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xi32>) -> (tensor<2x2xf32>) {
    //CHECK: %0 = "pphlo.multiply"(%arg0, %arg1) : (tensor<2x2xf32>, tensor<2x2xi32>) -> tensor<2x2xf32>
    %0 = "pphlo.convert"(%arg1) : (tensor<2x2xi32>) -> tensor<2x2xf32>
    %1 = "pphlo.multiply"(%arg0, %0) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    return %1 : tensor<2x2xf32>
}

// -----

func.func @main(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xi32>) -> (tensor<2x2xf32>) {
    //CHECK: %0 = "pphlo.dot"(%arg0, %arg1) : (tensor<2x2xf32>, tensor<2x2xi32>) -> tensor<2x2xf32>
    %0 = "pphlo.convert"(%arg1) : (tensor<2x2xi32>) -> tensor<2x2xf32>
    %1 = "pphlo.dot"(%arg0, %0) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    return %1 : tensor<2x2xf32>
}

// -----

func.func @main(%arg0: tensor<3x1x4xf32>, %arg1: tensor<3x4x5xi32>) -> (tensor<3x5xf32>) {
    //CHECK: %0 = "pphlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #pphlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<3x1x4xf32>, tensor<3x4x5xi32>) -> tensor<3x5xf32>
    %0 = "pphlo.convert"(%arg1) : (tensor<3x4x5xi32>) -> tensor<3x4x5xf32>
    %1 = "pphlo.dot_general"(%arg0, %0) {dot_dimension_numbers = #pphlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<3x1x4xf32>, tensor<3x4x5xf32>) -> tensor<3x5xf32>
    return %1 : tensor<3x5xf32>
}

// -----

func.func @main(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xi32>) -> (tensor<2x2xi32>) {
    //CHECK: %1 = "pphlo.dot"(%arg1, %0) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
    %0 = "pphlo.convert"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xi32>
    %1 = "pphlo.dot"(%arg1, %0) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
    return %1 : tensor<2x2xi32>
}

// -----

func.func @main(%arg0: tensor<2x2xf16>, %arg1: tensor<2x2xf32>) -> (tensor<2x2xf32>) {
    //CHECK: "pphlo.dot"(%arg1, %0) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    %0 = "pphlo.convert"(%arg0) : (tensor<2x2xf16>) -> tensor<2x2xf32>
    %1 = "pphlo.dot"(%arg1, %0) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    return %1 : tensor<2x2xf32>
}
