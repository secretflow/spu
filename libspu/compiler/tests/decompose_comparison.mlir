// RUN: mlir-pphlo-opt --decompose-comparison --cse --split-input-file %s | FileCheck %s

func.func @main(%arg0: tensor<2x2x!pphlo.pub<i32>>, %arg1: tensor<2x2x!pphlo.pub<i32>>) -> (tensor<2x2x!pphlo.pub<i1>>) {
    // CHECK: %0 = "pphlo.equal"(%arg0, %arg1) : (tensor<2x2x!pphlo.pub<i32>>, tensor<2x2x!pphlo.pub<i32>>) -> tensor<2x2x!pphlo.pub<i1>>
    // CHECK: %1 = "pphlo.not"(%0) : (tensor<2x2x!pphlo.pub<i1>>) -> tensor<2x2x!pphlo.pub<i1>>
    %0 = "pphlo.not_equal"(%arg0, %arg1) : (tensor<2x2x!pphlo.pub<i32>>, tensor<2x2x!pphlo.pub<i32>>) -> tensor<2x2x!pphlo.pub<i1>>
    return %0 : tensor<2x2x!pphlo.pub<i1>>
}

// -----

func.func @main(%arg0: tensor<2x2x!pphlo.pub<i32>>, %arg1: tensor<2x2x!pphlo.pub<i32>>) -> (tensor<2x2x!pphlo.pub<i1>>) {
    // CHECK: %0 = "pphlo.greater"(%arg0, %arg1) : (tensor<2x2x!pphlo.pub<i32>>, tensor<2x2x!pphlo.pub<i32>>) -> tensor<2x2x!pphlo.pub<i1>>
    // CHECK: %1 = "pphlo.not"(%0) : (tensor<2x2x!pphlo.pub<i1>>) -> tensor<2x2x!pphlo.pub<i1>>
    %0 = "pphlo.less_equal"(%arg0, %arg1) : (tensor<2x2x!pphlo.pub<i32>>, tensor<2x2x!pphlo.pub<i32>>) -> tensor<2x2x!pphlo.pub<i1>>
    return %0 : tensor<2x2x!pphlo.pub<i1>>
}

// -----

func.func @main(%arg0: tensor<2x2x!pphlo.pub<i32>>, %arg1: tensor<2x2x!pphlo.pub<i32>>) -> (tensor<2x2x!pphlo.pub<i1>>) {
    // CHECK: %0 = "pphlo.less"(%arg0, %arg1) : (tensor<2x2x!pphlo.pub<i32>>, tensor<2x2x!pphlo.pub<i32>>) -> tensor<2x2x!pphlo.pub<i1>>
    // CHECK: %1 = "pphlo.not"(%0) : (tensor<2x2x!pphlo.pub<i1>>) -> tensor<2x2x!pphlo.pub<i1>>
    %0 = "pphlo.greater_equal"(%arg0, %arg1) : (tensor<2x2x!pphlo.pub<i32>>, tensor<2x2x!pphlo.pub<i32>>) -> tensor<2x2x!pphlo.pub<i1>>
    return %0 : tensor<2x2x!pphlo.pub<i1>>
}

// -----

func.func @main(%arg0: tensor<2x2x!pphlo.pub<i32>>, %arg1: tensor<2x2x!pphlo.pub<i32>>) -> (tensor<2x2x!pphlo.pub<i1>>, tensor<2x2x!pphlo.pub<i1>>, tensor<2x2x!pphlo.pub<i1>>, tensor<2x2x!pphlo.pub<i1>>, tensor<2x2x!pphlo.pub<i1>>, tensor<2x2x!pphlo.pub<i1>>) {
    //CHECK: %0 = "pphlo.equal"(%arg0, %arg1) : (tensor<2x2x!pphlo.pub<i32>>, tensor<2x2x!pphlo.pub<i32>>) -> tensor<2x2x!pphlo.pub<i1>>
    //CHECK: %1 = "pphlo.not"(%0) : (tensor<2x2x!pphlo.pub<i1>>) -> tensor<2x2x!pphlo.pub<i1>>
    //CHECK: %2 = "pphlo.less"(%arg0, %arg1) : (tensor<2x2x!pphlo.pub<i32>>, tensor<2x2x!pphlo.pub<i32>>) -> tensor<2x2x!pphlo.pub<i1>>
    //CHECK: %3 = "pphlo.greater"(%arg0, %arg1) : (tensor<2x2x!pphlo.pub<i32>>, tensor<2x2x!pphlo.pub<i32>>) -> tensor<2x2x!pphlo.pub<i1>>
    //CHECK: %4 = "pphlo.not"(%3) : (tensor<2x2x!pphlo.pub<i1>>) -> tensor<2x2x!pphlo.pub<i1>>
    //CHECK: %5 = "pphlo.not"(%2) : (tensor<2x2x!pphlo.pub<i1>>) -> tensor<2x2x!pphlo.pub<i1>>
    %0 = "pphlo.equal"(%arg0, %arg1) : (tensor<2x2x!pphlo.pub<i32>>, tensor<2x2x!pphlo.pub<i32>>) -> tensor<2x2x!pphlo.pub<i1>>
    %1 = "pphlo.not_equal"(%arg0, %arg1) : (tensor<2x2x!pphlo.pub<i32>>, tensor<2x2x!pphlo.pub<i32>>) -> tensor<2x2x!pphlo.pub<i1>>
    %2 = "pphlo.less"(%arg0, %arg1) : (tensor<2x2x!pphlo.pub<i32>>, tensor<2x2x!pphlo.pub<i32>>) -> tensor<2x2x!pphlo.pub<i1>>
    %3 = "pphlo.greater"(%arg0, %arg1) : (tensor<2x2x!pphlo.pub<i32>>, tensor<2x2x!pphlo.pub<i32>>) -> tensor<2x2x!pphlo.pub<i1>>
    %4 = "pphlo.less_equal"(%arg0, %arg1) : (tensor<2x2x!pphlo.pub<i32>>, tensor<2x2x!pphlo.pub<i32>>) -> tensor<2x2x!pphlo.pub<i1>>
    %5 = "pphlo.greater_equal"(%arg0, %arg1) : (tensor<2x2x!pphlo.pub<i32>>, tensor<2x2x!pphlo.pub<i32>>) -> tensor<2x2x!pphlo.pub<i1>>
    return %0, %1, %2, %3, %4, %5 : tensor<2x2x!pphlo.pub<i1>>, tensor<2x2x!pphlo.pub<i1>>, tensor<2x2x!pphlo.pub<i1>>, tensor<2x2x!pphlo.pub<i1>>, tensor<2x2x!pphlo.pub<i1>>, tensor<2x2x!pphlo.pub<i1>>
  }
