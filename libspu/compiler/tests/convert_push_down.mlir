// RUN: mlir-pphlo-opt --convert-push-down --cse --split-input-file %s | FileCheck %s

func.func @main(%arg0: tensor<4xi32>, %arg1: tensor<2x2xf32>) -> (tensor<2x2xf32>) {
    // CHECK: %0 = "pphlo.reshape"(%arg0) : (tensor<4xi32>) -> tensor<2x2xi32>
    // CHECK: %1 = "pphlo.convert"(%0) : (tensor<2x2xi32>) -> tensor<2x2xf32>
    %0 = "pphlo.convert"(%arg0) : (tensor<4xi32>) -> tensor<4xf32>
    %1 = "pphlo.reshape"(%0) : (tensor<4xf32>) -> tensor<2x2xf32>
    %2 = "pphlo.multiply"(%1, %arg1) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    return %2 : tensor<2x2xf32>
}

// -----

func.func @main(%arg0: tensor<2x3xi32>, %arg1: tensor<2x3xf32>) -> (tensor<3x3xf32>) {
    // CHECK: %0 = "pphlo.transpose"(%arg0)
    // CHECK: %1 = "pphlo.convert"(%0)
    %0 = "pphlo.convert"(%arg0) : (tensor<2x3xi32>) -> tensor<2x3xf32>
    %1 = "pphlo.transpose"(%0) {permutation = array<i64: 1, 0>} : (tensor<2x3xf32>) -> tensor<3x2xf32>
    %2 = "pphlo.dot"(%1, %arg1) : (tensor<3x2xf32>, tensor<2x3xf32>) -> tensor<3x3xf32>
    return %2 : tensor<3x3xf32>
}

// -----
