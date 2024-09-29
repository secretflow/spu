// RUN: spu-opt --dot-general-to-dot --cse --split-input-file %s | FileCheck %s

func.func @main(%arg0: tensor<3x1x4xf32>, %arg1: tensor<3x4x5xi32>) -> (tensor<3x5xf32>) {
    //CHECK: %1 = pphlo.slice %arg0 [0:1:1, 0:1:1, 0:1:4] : (tensor<3x1x4xf32>) -> tensor<1x1x4xf32>
    //CHECK: %2 = pphlo.slice %0 [0:1:1, 0:1:4, 0:1:5] : (tensor<3x4x5xf32>) -> tensor<1x4x5xf32>
    //CHECK: %3 = pphlo.reshape %1 : (tensor<1x1x4xf32>) -> tensor<1x4xf32>
    //CHECK: %4 = pphlo.reshape %2 : (tensor<1x4x5xf32>) -> tensor<4x5xf32>
    //CHECK: %5 = pphlo.dot %3, %4 : (tensor<1x4xf32>, tensor<4x5xf32>) -> tensor<1x5xf32>
    //CHECK: %6 = pphlo.reshape %5 : (tensor<1x5xf32>) -> tensor<1x1x5xf32>
    //CHECK: %7 = pphlo.slice %arg0 [1:1:2, 0:1:1, 0:1:4] : (tensor<3x1x4xf32>) -> tensor<1x1x4xf32>
    //CHECK: %8 = pphlo.slice %0 [1:1:2, 0:1:4, 0:1:5] : (tensor<3x4x5xf32>) -> tensor<1x4x5xf32>
    //CHECK: %9 = pphlo.reshape %7 : (tensor<1x1x4xf32>) -> tensor<1x4xf32>
    //CHECK: %10 = pphlo.reshape %8 : (tensor<1x4x5xf32>) -> tensor<4x5xf32>
    //CHECK: %11 = pphlo.dot %9, %10 : (tensor<1x4xf32>, tensor<4x5xf32>) -> tensor<1x5xf32>
    //CHECK: %12 = pphlo.reshape %11 : (tensor<1x5xf32>) -> tensor<1x1x5xf32>
    //CHECK: %13 = pphlo.slice %arg0 [2:1:3, 0:1:1, 0:1:4] : (tensor<3x1x4xf32>) -> tensor<1x1x4xf32>
    //CHECK: %14 = pphlo.slice %0 [2:1:3, 0:1:4, 0:1:5] : (tensor<3x4x5xf32>) -> tensor<1x4x5xf32>
    //CHECK: %15 = pphlo.reshape %13 : (tensor<1x1x4xf32>) -> tensor<1x4xf32>
    //CHECK: %16 = pphlo.reshape %14 : (tensor<1x4x5xf32>) -> tensor<4x5xf32>
    //CHECK: %17 = pphlo.dot %15, %16 : (tensor<1x4xf32>, tensor<4x5xf32>) -> tensor<1x5xf32>
    //CHECK: %18 = pphlo.reshape %17 : (tensor<1x5xf32>) -> tensor<1x1x5xf32>
    //CHECK: %19 = pphlo.concatenate %6, %12, %18 dim = 0 : (tensor<1x1x5xf32>, tensor<1x1x5xf32>, tensor<1x1x5xf32>) -> tensor<3x1x5xf32>
    //CHECK: %20 = pphlo.reshape %19 : (tensor<3x1x5xf32>) -> tensor<3x5xf32>
    //CHECK: return %20 : tensor<3x5xf32>
    %0 = pphlo.convert %arg1 : (tensor<3x4x5xi32>) -> tensor<3x4x5xf32>
    %1 = pphlo.dot_general %arg0, %0, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<3x1x4xf32>, tensor<3x4x5xf32>) -> tensor<3x5xf32>
    return %1 : tensor<3x5xf32>
}

