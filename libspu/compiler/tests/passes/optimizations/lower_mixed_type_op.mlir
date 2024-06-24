// RUN: spu-opt --lower-mixed-type-op --split-input-file %s | FileCheck %s

func.func @main(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xi32>) -> (tensor<2x2xf32>) {
    //CHECK: %0 = pphlo.multiply %arg0, %arg1 : (tensor<2x2xf32>, tensor<2x2xi32>) -> tensor<2x2xf32>
    %0 = pphlo.convert %arg1 : (tensor<2x2xi32>) -> tensor<2x2xf32>
    %1 = pphlo.multiply %arg0, %0 : tensor<2x2xf32>
    return %1 : tensor<2x2xf32>
}

// -----

func.func @main(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xi32>) -> (tensor<2x2xf32>) {
    //CHECK: %0 = pphlo.dot %arg0, %arg1 : (tensor<2x2xf32>, tensor<2x2xi32>) -> tensor<2x2xf32>
    %0 = pphlo.convert %arg1 : (tensor<2x2xi32>) -> tensor<2x2xf32>
    %1 = pphlo.dot %arg0, %0 : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    return %1 : tensor<2x2xf32>
}

// -----

func.func @main(%arg0: tensor<3x1x4xf32>, %arg1: tensor<3x4x5xi32>) -> (tensor<3x5xf32>) {
    //CHECK:  pphlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<3x1x4xf32>, tensor<3x4x5xi32>) -> tensor<3x5xf32>
    %0 = pphlo.convert %arg1 : (tensor<3x4x5xi32>) -> tensor<3x4x5xf32>
    %1 = pphlo.dot_general %arg0, %0, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<3x1x4xf32>, tensor<3x4x5xf32>) -> tensor<3x5xf32>
    return %1 : tensor<3x5xf32>
}

// -----

func.func @main(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xi32>) -> (tensor<2x2xi32>) {
    //CHECK: %1 = pphlo.dot %arg1, %0 : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
    %0 = pphlo.convert %arg0 : (tensor<2x2xf32>) -> tensor<2x2xi32>
    %1 = pphlo.dot %arg1, %0 : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
    return %1 : tensor<2x2xi32>
}

// -----

func.func @main(%arg0: tensor<2x2xf16>, %arg1: tensor<2x2xf32>) -> (tensor<2x2xf32>) {
    //CHECK: pphlo.dot %arg1, %0 : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    %0 = pphlo.convert %arg0 : (tensor<2x2xf16>) -> tensor<2x2xf32>
    %1 = pphlo.dot %arg1, %0 : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    return %1 : tensor<2x2xf32>
}

// -----

func.func @main(%arg0: tensor<3x4x!pphlo.secret<f32>>, %arg1: tensor<3x4x!pphlo.secret<i32>>) -> tensor<3x4x!pphlo.secret<f32>> {
    //CHECK: pphlo.multiply %arg0, %arg1
    %0 = pphlo.convert %arg1 : (tensor<3x4x!pphlo.secret<i32>>) -> tensor<3x4x!pphlo.secret<f32>>
    %1 = pphlo.multiply %arg0, %0 : tensor<3x4x!pphlo.secret<f32>>
    return %1 : tensor<3x4x!pphlo.secret<f32>>
}
