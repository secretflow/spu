// RUN: mlir-pphlo-opt --optimize-select --split-input-file %s | FileCheck %s

func.func @main() -> (tensor<!pphlo.pub<f32>>) {
    %0 = "pphlo.constant"() {value = dense<0xFF800000> : tensor<f32>} : () -> tensor<!pphlo.pub<f32>>
    %1 = "pphlo.constant"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<!pphlo.pub<f32>>
    %2 = "pphlo.less"(%0, %1): (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<i1>>
    //CHECK-NOT: pphlo.prefer_a
    %3 = "pphlo.select"(%2, %0, %1): (tensor<!pphlo.pub<i1>>, tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    return %3: tensor<!pphlo.pub<f32>>
}

// -----

func.func @main() -> (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) {
    %0 = "pphlo.constant"() {value = dense<0xFF800000> : tensor<f32>} : () -> tensor<!pphlo.pub<f32>>
    %1 = "pphlo.constant"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<!pphlo.pub<f32>>
    %2 = "pphlo.less"(%0, %1): (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<i1>>
    //CHECK: pphlo.prefer_a
    %3 = "pphlo.select"(%2, %0, %1): (tensor<!pphlo.pub<i1>>, tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    %4 = "pphlo.select"(%2, %1, %0): (tensor<!pphlo.pub<i1>>, tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    return %3, %4: tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>
}

// -----

func.func @main() -> (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<i1>>) {
    %0 = "pphlo.constant"() {value = dense<0xFF800000> : tensor<f32>} : () -> tensor<!pphlo.pub<f32>>
    %1 = "pphlo.constant"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<!pphlo.pub<f32>>
    %2 = "pphlo.less"(%0, %1): (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<i1>>
    //CHECK-NOT: pphlo.prefer_a
    %3 = "pphlo.select"(%2, %0, %1): (tensor<!pphlo.pub<i1>>, tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    %4 = "pphlo.not"(%2): (tensor<!pphlo.pub<i1>>) -> tensor<!pphlo.pub<i1>>
    return %3, %4: tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<i1>>
}

// -----

func.func @main(%arg0: tensor<!pphlo.pub<i1>>) -> (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) {
    %0 = "pphlo.constant"() {value = dense<0xFF800000> : tensor<f32>} : () -> tensor<!pphlo.pub<f32>>
    %1 = "pphlo.constant"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<!pphlo.pub<f32>>
    //CHECK: pphlo.prefer_a
    %2 = "pphlo.select"(%arg0, %0, %1): (tensor<!pphlo.pub<i1>>, tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    %3 = "pphlo.select"(%arg0, %1, %0): (tensor<!pphlo.pub<i1>>, tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    return %2, %3: tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>
}
