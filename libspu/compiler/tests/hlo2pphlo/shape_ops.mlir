// RUN: pphlo-opt --hlo-legalize-to-pphlo=input_vis_list=VIS_PUBLIC --lower-conversion-cast --split-input-file %s | FileCheck %s

func.func @main(%arg0: tensor<16xf32>) -> (tensor<1024x16xf32>) {
    // CHECK: %0 = pphlo.broadcast %arg0, dims = [1] : (tensor<16xf32>) -> tensor<1024x16xf32>
    %0 = "stablehlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = array<i64: 1>} : (tensor<16xf32>) -> tensor<1024x16xf32>
    return %0 : tensor<1024x16xf32>
}

// -----

func.func @main(%arg0: tensor<16xf32>) -> (tensor<1x16xf32>) {
    // CHECK: %0 = pphlo.reshape %arg0 : (tensor<16xf32>) -> tensor<1x16xf32>
    %0 = "stablehlo.reshape"(%arg0) : (tensor<16xf32>) -> tensor<1x16xf32>
    return %0 : tensor<1x16xf32>
}

// -----

func.func @main(%arg0: tensor<1x16xf32>) -> (tensor<16x1xf32>) {
    // CHECK: %0 = pphlo.transpose %arg0, dims = [1, 0] : (tensor<1x16xf32>) -> tensor<16x1xf32>
    %0 = "stablehlo.transpose"(%arg0) {permutation = array<i64: 1, 0>} : (tensor<1x16xf32>) -> tensor<16x1xf32>
    return %0 : tensor<16x1xf32>
}

// -----

func.func @main(%arg0: tensor<16x1xf32>) -> (tensor<16x2xf32>) {
    // CHECK: %0 = pphlo.concatenate %arg0, %arg0 dim = 1 : (tensor<16x1xf32>, tensor<16x1xf32>) -> tensor<16x2xf32>
    %0 = "stablehlo.concatenate"(%arg0, %arg0) {dimension = 1 : i64} : (tensor<16x1xf32>, tensor<16x1xf32>) -> tensor<16x2xf32>
    return %0 : tensor<16x2xf32>
}

// -----

func.func @main(%arg4: tensor<3x4xi32>) -> (tensor<1x2xi32>) {
    // CHECK: %0 = pphlo.slice %arg0 [1:1:2, 0:2:4] : (tensor<3x4xi32>) -> tensor<1x2xi32>
    %0 = "stablehlo.slice"(%arg4) {start_indices = array<i64: 1, 0>, limit_indices = array<i64: 2, 4>, strides = array<i64: 1, 2>} : (tensor<3x4xi32>) -> tensor<1x2xi32>
    return %0 : tensor<1x2xi32>
}
