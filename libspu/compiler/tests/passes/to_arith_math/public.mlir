// RUN: spu-opt --legalize-to-arith --cse --split-input-file %s | FileCheck %s

func.func @main(%arg0: tensor<2x2xf32>) -> (tensor<2x2xf32>) {
    //CHECK: %0 = math.cos %arg0 : tensor<2x2xf32>
    %0 = pphlo.cosine %arg0 : tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
}

// -----

func.func @main(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> (tensor<2x2xf32>) {
    //CHECK: %0 = arith.divf %arg0, %arg1 : tensor<2x2xf32>
    %0 = pphlo.divide %arg0, %arg1 : tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
}

// -----

func.func @main(%arg0: tensor<2x2xf32>) -> (tensor<2x2xf32>) {
    //CHECK: %0 = math.tanh %arg0 : tensor<2x2xf32>
    %0 = pphlo.tanh %arg0 : tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
}

// -----

func.func @main(%arg0: tensor<2x2xf32>) -> (tensor<2x2xf32>) {
    //CHECK: %0 = math.rsqrt %arg0 : tensor<2x2xf32>
    %0 = pphlo.rsqrt %arg0 : tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
}

// -----

func.func @main(%arg0: tensor<2x2xf32>) -> (tensor<2x2xf32>) {
    //CHECK: %0 = math.sqrt %arg0 : tensor<2x2xf32>
    %0 = pphlo.sqrt %arg0 : tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
}

// -----

func.func @main(%arg0: tensor<2x2xf32>) -> (tensor<2x2xf32>) {
    //CHECK: %0 = math.sin %arg0 : tensor<2x2xf32>
    %0 = pphlo.sine %arg0 : tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
}

// -----

func.func @main(%arg0: tensor<2xf32>) -> tensor<2xf32> {
    //CHECK: %cst = arith.constant dense<1.000000e+00> : tensor<2xf32>
    //CHECK: %0 = arith.divf %cst, %arg0 : tensor<2xf32>
    %0 = pphlo.reciprocal %arg0 : tensor<2xf32>
    return %0 : tensor<2xf32>
}

// -----

func.func @main(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> (tensor<2x2xf32>) {
    //CHECK: %0 = math.powf %arg0, %arg1 : tensor<2x2xf32>
    %0 = pphlo.power %arg0, %arg1 : tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
}

// -----

func.func @main(%arg0: tensor<2x2xi32>, %arg1: tensor<2x2xi32>) -> (tensor<2x2xi32>) {
    //CHECK-NOT: math.powf
    //CHECK: scf.for
    %0 = pphlo.power %arg0, %arg1 : tensor<2x2xi32>
    return %0 : tensor<2x2xi32>
}

// -----

func.func @main(%arg0: tensor<2x2xf32>) -> (tensor<2x2xf32>) {
    //CHECK: %cst = arith.constant dense<1.000000e+00> : tensor<2x2xf32>
    //CHECK: %0 = arith.negf %arg0 : tensor<2x2xf32>
    //CHECK: %1 = math.exp %0 : tensor<2x2xf32>
    //CHECK: %2 = arith.addf %cst, %1 : tensor<2x2xf32>
    //CHECK: %3 = arith.divf %cst, %2 : tensor<2x2xf32>
    %0 = pphlo.logistic %arg0 : tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
}

// -----

func.func @main(%arg0: tensor<2x2xf32>) -> (tensor<2x2xf32>) {
    //CHECK: %0 = math.log %arg0 : tensor<2x2xf32>
    %0 = pphlo.log %arg0 : tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
}

// -----

func.func @main(%arg0: tensor<2x2xf32>) -> (tensor<2x2xf32>) {
    //CHECK: %0 = math.exp %arg0 : tensor<2x2xf32>
    %0 = pphlo.exponential %arg0 : tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
}

// -----

func.func @main(%arg0: tensor<2xf32>) -> tensor<2xf32> {
    //CHECK: %0 = math.erf %arg0 : tensor<2xf32>
    %0 = pphlo.custom_call @mhlo.erf(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
    return %0 : tensor<2xf32>
}
