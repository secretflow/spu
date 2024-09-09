// RUN: spu-opt --decompose-ops --canonicalize --cse --split-input-file %s | FileCheck %s

func.func @main(%arg0: tensor<2x2xi32>, %arg1: tensor<2x2xi32>) -> (tensor<2x2xi1>) {
    // CHECK: %0 = pphlo.equal %arg0, %arg1 : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi1>
    // CHECK: %1 = pphlo.not %0 : tensor<2x2xi1>
    %0 = pphlo.not_equal %arg0, %arg1 : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi1>
    return %0 : tensor<2x2xi1>
}

// -----

func.func @main(%arg0: tensor<2x2xi32>, %arg1: tensor<2x2xi32>) -> (tensor<2x2xi1>) {
    // CHECK: %0 = pphlo.less %arg1, %arg0 : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi1>
    // CHECK: %1 = pphlo.not %0 : tensor<2x2xi1>
    %0 = pphlo.less_equal %arg0, %arg1 : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi1>
    return %0 : tensor<2x2xi1>
}

// -----

func.func @main(%arg0: tensor<2x2xi32>, %arg1: tensor<2x2xi32>) -> (tensor<2x2xi1>) {
    // CHECK: %0 = pphlo.less %arg0, %arg1 : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi1>
    // CHECK: %1 = pphlo.not %0 : tensor<2x2xi1>
    %0 = pphlo.greater_equal %arg0, %arg1 : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi1>
    return %0 : tensor<2x2xi1>
}

// -----

func.func @main(%arg0: tensor<2x2xi32>, %arg1: tensor<2x2xi32>) -> (tensor<2x2xi1>, tensor<2x2xi1>, tensor<2x2xi1>, tensor<2x2xi1>, tensor<2x2xi1>, tensor<2x2xi1>) {
    //CHECK: %0 = pphlo.equal %arg0, %arg1 : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi1>
    //CHECK: %1 = pphlo.not %0 : tensor<2x2xi1>
    //CHECK: %2 = pphlo.less %arg0, %arg1 : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi1>
    //CHECK: %3 = pphlo.less %arg1, %arg0 : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi1>
    //CHECK: %4 = pphlo.not %3 : tensor<2x2xi1>
    //CHECK: %5 = pphlo.not %2 : tensor<2x2xi1>
    %0 = pphlo.equal %arg0, %arg1 : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi1>
    %1 = pphlo.not_equal %arg0, %arg1 : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi1>
    %2 = pphlo.less %arg0, %arg1 : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi1>
    %3 = pphlo.greater %arg0, %arg1 : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi1>
    %4 = pphlo.less_equal %arg0, %arg1 : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi1>
    %5 = pphlo.greater_equal %arg0, %arg1 : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi1>
    return %0, %1, %2, %3, %4, %5 : tensor<2x2xi1>, tensor<2x2xi1>, tensor<2x2xi1>, tensor<2x2xi1>, tensor<2x2xi1>, tensor<2x2xi1>
  }

// -----

func.func @main(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> (tensor<2x2xf32>) {
    //CHECK: %0 = pphlo.less %arg1, %arg0 : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xi1>
    //CHECK: %1 = pphlo.select %0, %arg0, %arg1 : (tensor<2x2xi1>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    %0 = pphlo.maximum %arg0, %arg1 : tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
}

// -----

func.func @main(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> (tensor<2x2xf32>) {
    //CHECK: %0 = pphlo.less %arg0, %arg1 : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xi1>
    //CHECK: %1 = pphlo.select %0, %arg0, %arg1 : (tensor<2x2xi1>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    %0 = pphlo.minimum %arg0, %arg1 : tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
}

// -----

func.func @main(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> (tensor<2x2xf32>) {
    //CHECK: %0 = pphlo.negate %arg1 : tensor<2x2xf32>
    //CHECK: %1 = pphlo.add %arg0, %0 : tensor<2x2xf32>
    %0 = pphlo.subtract %arg0, %arg1 : tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
}

// -----

func.func @main(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>, %arg2: tensor<2x2xf32>) -> (tensor<2x2xf32>) {
    //CHECK: %0 = pphlo.less %arg1, %arg0 : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xi1>
    //CHECK: %1 = pphlo.select %0, %arg0, %arg1 : (tensor<2x2xi1>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    //CHECK: %2 = pphlo.less %1, %arg2 : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xi1>
    //CHECK: %3 = pphlo.select %2, %1, %arg2 : (tensor<2x2xi1>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    %0 = pphlo.clamp %arg0, %arg1, %arg2 : tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
}
