// RUN: mlir-pphlo-opt --optimize-select --split-input-file %s | FileCheck %s

func.func @main() -> (tensor<f32>) {
    %0 = "pphlo.constant"() {value = dense<0xFF800000> : tensor<f32>} : () -> tensor<f32>
    %1 = "pphlo.constant"() {value = dense<1.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %2 = "pphlo.less"(%0, %1): (tensor<f32>, tensor<f32>) -> tensor<i1>
    //CHECK-NOT: pphlo.prefer_a
    %3 = "pphlo.select"(%2, %0, %1): (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    return %3: tensor<f32>
}

// -----

func.func @main() -> (tensor<f32>, tensor<f32>) {
    %0 = "pphlo.constant"() {value = dense<0xFF800000> : tensor<f32>} : () -> tensor<f32>
    %1 = "pphlo.constant"() {value = dense<1.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %2 = "pphlo.less"(%0, %1): (tensor<f32>, tensor<f32>) -> tensor<i1>
    //CHECK: pphlo.prefer_a
    %3 = "pphlo.select"(%2, %0, %1): (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %4 = "pphlo.select"(%2, %1, %0): (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    return %3, %4: tensor<f32>, tensor<f32>
}

// -----

func.func @main() -> (tensor<f32>, tensor<i1>) {
    %0 = "pphlo.constant"() {value = dense<0xFF800000> : tensor<f32>} : () -> tensor<f32>
    %1 = "pphlo.constant"() {value = dense<1.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %2 = "pphlo.less"(%0, %1): (tensor<f32>, tensor<f32>) -> tensor<i1>
    //CHECK-NOT: pphlo.prefer_a
    %3 = "pphlo.select"(%2, %0, %1): (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %4 = "pphlo.not"(%2): (tensor<i1>) -> tensor<i1>
    return %3, %4: tensor<f32>, tensor<i1>
}

// -----

func.func @main(%arg0: tensor<i1>) -> (tensor<f32>, tensor<f32>) {
    %0 = "pphlo.constant"() {value = dense<0xFF800000> : tensor<f32>} : () -> tensor<f32>
    %1 = "pphlo.constant"() {value = dense<1.000000e+00> : tensor<f32>} : () -> tensor<f32>
    //CHECK: pphlo.prefer_a
    %2 = "pphlo.select"(%arg0, %0, %1): (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %3 = "pphlo.select"(%arg0, %1, %0): (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    return %2, %3: tensor<f32>, tensor<f32>
}


// -----

func.func @main(%arg0: tensor<i1>, %arg1: tensor<f32>) -> (tensor<f32>) {
    %1 = "pphlo.constant"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<f32>
    //CHECK: pphlo.multiply
    %2 = "pphlo.select"(%arg0, %arg1, %1): (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    return %2: tensor<f32>
}
