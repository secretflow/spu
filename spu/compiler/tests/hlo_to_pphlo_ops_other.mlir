// RUN: mlir-pphlo-opt --hlo-legalize-to-pphlo --split-input-file %s | FileCheck %s

func @main(%arg0: tensor<16xf32>,%arg1: tensor<1024x1xi1>, %arg2: tensor<1024x1xf32>, %arg3: tensor<1024x1xf32>, %arg4: tensor<3x4xi32>) -> (tensor<1024x16xf32>) {
    // CHECK: %0 = "pphlo.broadcast"(%arg0) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<16x!pphlo.pub<f32>>) -> tensor<1024x16x!pphlo.pub<f32>>
    %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<16xf32>) -> tensor<1024x16xf32>    
    // CHECK: %1 = "pphlo.reshape"(%arg0) : (tensor<16x!pphlo.pub<f32>>) -> tensor<1x16x!pphlo.pub<f32>>
    %1 = "mhlo.reshape"(%arg0) : (tensor<16xf32>) -> tensor<1x16xf32>
    // CHECK: %2 = "pphlo.transpose"(%1) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<1x16x!pphlo.pub<f32>>) -> tensor<16x1x!pphlo.pub<f32>>
    %2 = "mhlo.transpose"(%1) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<1x16xf32>) -> tensor<16x1xf32>
    // CHECK: %3 = "pphlo.dot"(%0, %2) : (tensor<1024x16x!pphlo.pub<f32>>, tensor<16x1x!pphlo.pub<f32>>) -> tensor<1024x1x!pphlo.pub<f32>>
    %3 = "mhlo.dot"(%0, %2) {precision_config = [#mhlo<"precision DEFAULT">, #mhlo<"precision DEFAULT">]} : (tensor<1024x16xf32>, tensor<16x1xf32>) -> tensor<1024x1xf32> 
    // CHECK: %6 = "pphlo.concatenate"(%4, %5) {dimension = 1 : i64} : (tensor<1024x16x!pphlo.pub<f32>>, tensor<1024x1x!pphlo.pub<f32>>) -> tensor<1024x17x!pphlo.pub<f32>>
    %4 = "mhlo.concatenate"(%0, %3) {dimension = 1 : i64} : (tensor<1024x16xf32>, tensor<1024x1xf32>) -> tensor<1024x17xf32>
    // CHECK: %7 = "pphlo.select"(%arg1, %arg2, %arg3) : (tensor<1024x1x!pphlo.pub<i1>>, tensor<1024x1x!pphlo.pub<f32>>, tensor<1024x1x!pphlo.pub<f32>>) -> tensor<1024x1x!pphlo.pub<f32>>
    %5 = "mhlo.select"(%arg1, %arg2, %arg3) : (tensor<1024x1xi1>, tensor<1024x1xf32>, tensor<1024x1xf32>) -> tensor<1024x1xf32>
    // CHECK: %8 = "pphlo.slice"(%arg4) {limit_indices = dense<[2, 4]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<[1, 2]> : tensor<2xi64>} : (tensor<3x4x!pphlo.pub<i32>>) -> tensor<1x2x!pphlo.pub<i32>>
    %6 = "mhlo.slice"(%arg4) {start_indices = dense<[1, 0]> : tensor<2xi64>, limit_indices = dense<[2, 4]> : tensor<2xi64>, strides = dense<[1, 2]> : tensor<2xi64>} : (tensor<3x4xi32>) -> tensor<1x2xi32>
    return %0 : tensor<1024x16xf32>
}
