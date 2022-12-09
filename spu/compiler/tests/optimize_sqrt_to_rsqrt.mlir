// RUN: mlir-pphlo-opt --optimize-sqrt-to-rsqrt --split-input-file %s | FileCheck %s

func.func @do_optimize(%arg0: tensor<!pphlo.pub<f32>>, %arg1: tensor<!pphlo.pub<f32>>) -> (tensor<!pphlo.pub<f32>>) {
    %0 = "pphlo.constant"() {value = dense<9.99999993E-9> : tensor<f32>} : () -> tensor<!pphlo.pub<f32>>
    %1 = "pphlo.sqrt"(%arg0) : (tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    %2 = "pphlo.add"(%1, %0) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    //CHECK: pphlo.rsqrt
    //CHECK: pphlo.mul
    %3 = "pphlo.divide"(%arg1, %2) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    return %3: tensor<!pphlo.pub<f32>>
}

func.func @to_large_const(%arg0: tensor<!pphlo.pub<f32>>, %arg1: tensor<!pphlo.pub<f32>>) -> (tensor<!pphlo.pub<f32>>) {
    %0 = "pphlo.constant"() {value = dense<9.99999993> : tensor<f32>} : () -> tensor<!pphlo.pub<f32>>
    %1 = "pphlo.sqrt"(%arg0) : (tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    %2 = "pphlo.add"(%1, %0) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    //CHECK-NOT: pphlo.rsqrt
    %3 = "pphlo.divide"(%arg1, %2) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    return %3: tensor<!pphlo.pub<f32>>
}

