// RUN: spu-opt -hlo-legalize-to-pphlo=input_vis_list=VIS_PUBLIC,VIS_PUBLIC --lower-conversion-cast %s --split-input-file  | FileCheck %s

func.func @main(%arg0: tensor<1x1x4x4xf32>, %arg1: tensor<1x1x2x2xf32>) -> (tensor<1x1x4x4xf32>) {
    //CHECK: %0 = pphlo.pad %arg0, %cst, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 0, 0] : (tensor<1x1x4x4xf32>, tensor<f32>) -> tensor<1x1x5x5xf32>
    //CHECK: %1 = pphlo.convolution(%0, %arg1)
    //CHECK:         dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
    //CHECK:         window = {stride = [1, 1]} : (tensor<1x1x5x5xf32>, tensor<1x1x2x2xf32>) -> tensor<1x1x4x4xf32>
    %0 = stablehlo.convolution(%arg0, %arg1)
            dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
            window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
            {
              batch_group_count = 1 : i64,
              feature_group_count = 1 : i64
            } : (tensor<1x1x4x4xf32>, tensor<1x1x2x2xf32>) -> tensor<1x1x4x4xf32>
    return %0 : tensor<1x1x4x4xf32>
}

// -----

func.func @main(%arg0: tensor<2x3x1x4xf32>, %arg1:tensor<1x3x2x3xf32>) -> (tensor<1x1x1x2xf32>) {
    //CHECK: %0 = pphlo.convolution(%arg0, %arg1)
    //CHECK:            dim_numbers = [f, 0, b, 1]x[o, 1, i, 0]->[f, 0, b, 1],
    //CHECK:            window = {stride = [1, 1]} : (tensor<2x3x1x4xf32>, tensor<1x3x2x3xf32>) -> tensor<1x1x1x2xf32>
    %0 = stablehlo.convolution(%arg0, %arg1)
          dim_numbers = [f, 0, b, 1]x[o, 1, i,0]->[f, 0, b, 1],
          window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
          {
            batch_group_count = 1 : i64,
            feature_group_count = 1 : i64
          } : (tensor<2x3x1x4xf32>,tensor<1x3x2x3xf32>) -> tensor<1x1x1x2xf32>
    return %0 : tensor<1x1x1x2xf32>
}

// -----

func.func @main(%arg0: tensor<1x1x4x4xf32>, %arg1: tensor<1x1x2x2xf32>) -> (tensor<1x1x7x7xf32>) {
    //CHECK: %cst = arith.constant dense<0.000000e+00> : tensor<f32>
    //CHECK: %0 = pphlo.pad %arg0, %cst, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<1x1x4x4xf32>, tensor<f32>) -> tensor<1x1x8x8xf32>
    //CHECK: %1 = pphlo.convolution(%0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1]} : (tensor<1x1x8x8xf32>, tensor<1x1x2x2xf32>) -> tensor<1x1x7x7xf32>
    %0 = stablehlo.convolution(%arg0, %arg1)
          dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
          window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]}
          {
            batch_group_count = 1 : i64,
            feature_group_count = 1 : i64
          } : (tensor<1x1x4x4xf32>, tensor<1x1x2x2xf32>) -> tensor<1x1x7x7xf32>
    return %0 : tensor<1x1x7x7xf32>
}

// -----

func.func @main(%arg0: tensor<1x1x4x4xf32>, %arg1: tensor<1x1x2x2xf32>) -> (tensor<1x1x8x8xf32>) {
    //CHECK: %cst = arith.constant dense<0.000000e+00> : tensor<f32>
    //CHECK: %0 = pphlo.pad %arg0, %cst, low = [0, 0, 1, 1], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<1x1x4x4xf32>, tensor<f32>) -> tensor<1x1x9x9xf32>
    //CHECK: %1 = pphlo.convolution(%0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1]} : (tensor<1x1x9x9xf32>, tensor<1x1x2x2xf32>) -> tensor<1x1x8x8xf32>
    %0 = stablehlo.convolution(%arg0, %arg1)
          dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
          window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]}
          {
            batch_group_count = 1 : i64,
            feature_group_count = 1 : i64
          } : (tensor<1x1x4x4xf32>, tensor<1x1x2x2xf32>) -> tensor<1x1x8x8xf32>
    return %0 : tensor<1x1x8x8xf32>
}

// -----

func.func @main(%arg0: tensor<1x1x4x6xf32>, %arg1: tensor<1x1x2x3xf32>) -> (tensor<1x1x2x2xf32>) {
    //CHECK: %cst = arith.constant dense<0.000000e+00> : tensor<f32>
    //CHECK: %0 = pphlo.pad %arg1, %cst, low = [0, 0, 0, 0], high = [0, 0, 0, 0], interior = [0, 0, 1, 1] : (tensor<1x1x2x3xf32>, tensor<f32>) -> tensor<1x1x3x5xf32>
    //CHECK: %1 = pphlo.convolution(%arg0, %0) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1]} : (tensor<1x1x4x6xf32>, tensor<1x1x3x5xf32>) -> tensor<1x1x2x2xf32>
    %0 = stablehlo.convolution(%arg0, %arg1)
          dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
          window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]}
          {
            batch_group_count = 1 : i64,
            feature_group_count = 1 : i64
          } : (tensor<1x1x4x6xf32>, tensor<1x1x2x3xf32>) -> tensor<1x1x2x2xf32>
    return %0 : tensor<1x1x2x2xf32>
}
