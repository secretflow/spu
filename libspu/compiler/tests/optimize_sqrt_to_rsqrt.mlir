// RUN: mlir-pphlo-opt --optimize-sqrt-plus-eps --rewrite-div-sqrt-pattern --split-input-file %s | FileCheck %s

func.func @main(%arg0: tensor<!pphlo.pub<f32>>, %arg1: tensor<!pphlo.pub<f32>>) -> (tensor<!pphlo.pub<f32>>) {
    %0 = "pphlo.constant"() {value = dense<9.99999993E-9> : tensor<f32>} : () -> tensor<!pphlo.pub<f32>>
    %1 = "pphlo.sqrt"(%arg0) : (tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    %2 = "pphlo.add"(%1, %0) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    //CHECK: pphlo.rsqrt
    //CHECK: pphlo.mul
    %3 = "pphlo.divide"(%arg1, %2) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    return %3: tensor<!pphlo.pub<f32>>
}

// -----

func.func @main(%arg0: tensor<!pphlo.pub<f32>>, %arg1: tensor<!pphlo.pub<f32>>) -> (tensor<!pphlo.pub<f32>>) {
    %0 = "pphlo.constant"() {value = dense<9.99999993> : tensor<f32>} : () -> tensor<!pphlo.pub<f32>>
    %1 = "pphlo.sqrt"(%arg0) : (tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    %2 = "pphlo.add"(%1, %0) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    //CHECK-NOT: pphlo.rsqrt
    %3 = "pphlo.divide"(%arg1, %2) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    return %3: tensor<!pphlo.pub<f32>>
}

// -----

func.func @main(%arg0: tensor<!pphlo.pub<f32>>, %arg1: tensor<!pphlo.pub<f32>>) -> (tensor<!pphlo.pub<f32>>) {
    %0 = "pphlo.constant"() {value = dense<9.99999993E-9> : tensor<f32>} : () -> tensor<!pphlo.pub<f32>>
    %1 = "pphlo.sqrt"(%arg0) : (tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    %2 = "pphlo.add"(%1, %0) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    %3 = "pphlo.multiply"(%1, %1) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    //CHECK: pphlo.rsqrt
    //CHECK: pphlo.mul
    %4 = "pphlo.divide"(%arg1, %3) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    return %4: tensor<!pphlo.pub<f32>>
}

