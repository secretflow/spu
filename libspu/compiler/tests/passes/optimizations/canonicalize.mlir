// RUN: spu-opt --canonicalize --split-input-file %s | FileCheck %s

func.func @mul_fp_cf32(%arg0: tensor<2xf32>) -> (tensor<2xf32>) {
    // CHECK: return %arg0 : tensor<2xf32>
    %0 = arith.constant dense<1.000000e+00> : tensor<2xf32>
    %1 = pphlo.multiply %arg0, %0 : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    return %1 : tensor<2xf32>
}

// -----

func.func @mul_fp_ci32(%arg0: tensor<2xf32>) -> (tensor<2xf32>) {
    // CHECK: return %arg0 : tensor<2xf32>
    %0 = arith.constant dense<1> : tensor<2xi32>
    %1 = pphlo.multiply %arg0, %0 : (tensor<2xf32>, tensor<2xi32>) -> tensor<2xf32>
    return %1 : tensor<2xf32>
}

// -----

func.func @mul_fp_ci32_ci32(%arg0: tensor<2xf32>) -> (tensor<2xf32>) {
    // CHECK: %cst = arith.constant dense<3> : tensor<2xi32>
    // CHECK: %0 = pphlo.multiply %arg0, %cst : (tensor<2xf32>, tensor<2xi32>) -> tensor<2xf32>
    // CHECK: return %0 : tensor<2xf32>
    %0 = arith.constant dense<1> : tensor<2xi32>
    %1 = arith.constant dense<3> : tensor<2xi32>
    %2 = pphlo.multiply %0, %1 : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
    %3 = pphlo.multiply %arg0, %2 : (tensor<2xf32>, tensor<2xi32>) -> tensor<2xf32>
    return %3 : tensor<2xf32>
}

// -----

func.func @mul_fp_ci32_cf32(%arg0: tensor<2xf32>) -> (tensor<2xf32>) {
    // CHECK: %cst = arith.constant dense<6.000000e+00> : tensor<2xf32>
    // CHECK: %0 = pphlo.multiply %arg0, %cst : tensor<2xf32>
    // CHECK: return %0 : tensor<2xf32>
    %0 = arith.constant dense<2> : tensor<2xi32>
    %1 = arith.constant dense<3.0> : tensor<2xf32>
    %2 = pphlo.multiply %0, %1 : (tensor<2xi32>, tensor<2xf32>) -> tensor<2xf32>
    %3 = pphlo.multiply %arg0, %2 : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    return %3 : tensor<2xf32>
}

// -----

func.func @slice_full(%arg0: tensor<2xf32>) -> (tensor<2xf32>) {
    //CHECK: return %arg0
    %0 = pphlo.slice %arg0 [0:1:2] : (tensor<2xf32>) -> tensor<2xf32>
    return %0 : tensor<2xf32>
}

// -----

func.func @slice_full_1d(%arg0: tensor<2x2xf32>) -> (tensor<2x1xf32>) {
    //CHECK: %0 = pphlo.slice
    //CHECK: return %0
    %0 = pphlo.slice %arg0 [0:1:2, 0:1:1] : (tensor<2x2xf32>) -> tensor<2x1xf32>
    return %0 : tensor<2x1xf32>
}

// -----

func.func @dot_vv(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> (tensor<f32>) {
    //CHECK: %0 = pphlo.reshape %arg0 : (tensor<2xf32>) -> tensor<1x2xf32>
    //CHECK: %1 = pphlo.reshape %arg1 : (tensor<2xf32>) -> tensor<2x1xf32>
    //CHECK: %2 = pphlo.dot %0, %1 : (tensor<1x2xf32>, tensor<2x1xf32>) -> tensor<1x1xf32>
    //CHECK: %3 = pphlo.reshape %2 : (tensor<1x1xf32>) -> tensor<f32>
    %0 = pphlo.dot %arg0,%arg1 : (tensor<2xf32>, tensor<2xf32>) -> tensor<f32>
    return %0 : tensor<f32>
}

// -----

func.func @dot_mv(%arg0: tensor<2x2xf32>, %arg1: tensor<2xf32>) -> (tensor<2xf32>) {
    //CHECK: %0 = pphlo.reshape %arg1 : (tensor<2xf32>) -> tensor<2x1xf32>
    //CHECK: %1 = pphlo.dot %arg0, %0 : (tensor<2x2xf32>, tensor<2x1xf32>) -> tensor<2x1xf32>
    //CHECK: %2 = pphlo.reshape %1 : (tensor<2x1xf32>) -> tensor<2xf32>
    %0 = pphlo.dot %arg0,%arg1 : (tensor<2x2xf32>, tensor<2xf32>) -> tensor<2xf32>
    return %0 : tensor<2xf32>
}

// -----

func.func @main(%arg0: tensor<1x1x4xf32>, %arg1: tensor<1x1x2xf32>) -> tensor<1x1x2xf32> {
    //CHECK: %3 = pphlo.reshape %1 : (tensor<1x4x1xf32>) -> tensor<1x1x4x1xf32>
    //CHECK: %4 = pphlo.reshape %2 : (tensor<3x1x1xf32>) -> tensor<1x3x1x1xf32>
    //CHECK: %5 = pphlo.convolution(%3, %4) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1]} : (tensor<1x1x4x1xf32>, tensor<1x3x1x1xf32>) -> tensor<1x1x2x1xf32>
    //CHECK: %6 = pphlo.reshape %5 : (tensor<1x1x2x1xf32>) -> tensor<1x2x1xf32>
    //CHECK: %7 = pphlo.transpose %6, dims = [0, 2, 1] : (tensor<1x2x1xf32>) -> tensor<1x1x2xf32>
    %0 = arith.constant dense<0.000000e+00> : tensor<f32>
    %1 = pphlo.pad %arg1, %0, low = [0, 0, 0], high = [0, 0, 0], interior = [0, 0, 1] : (tensor<1x1x2xf32>, tensor<f32>) -> tensor<1x1x3xf32>
    %2 = pphlo.convolution(%arg0, %1)
            dim_numbers = [b, f, 0]x[o, i, 0]->[b, f, 0],
            window = {stride = [1]} : (tensor<1x1x4xf32>, tensor<1x1x3xf32>) -> tensor<1x1x2xf32>
    return %2 : tensor<1x1x2xf32>
}
