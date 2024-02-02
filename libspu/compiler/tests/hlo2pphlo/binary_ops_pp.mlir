// RUN: mlir-pphlo-opt --hlo-legalize-to-pphlo=input_vis_list=VIS_PUBLIC,VIS_PUBLIC --lower-conversion-cast --split-input-file %s | FileCheck %s

func.func @main(%arg0: tensor<2x2xi32>, %arg1: tensor<2x2xi32>) -> (tensor<2x2xi32>) {
    // CHECK: pphlo.subtract %arg0, %arg1 : tensor<2x2xi32>
    %0 = "stablehlo.subtract"(%arg0, %arg1) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
    return %0 : tensor<2x2xi32>
}

// -----

func.func @main(%arg0: tensor<2x2xi32>, %arg1: tensor<2x2xi32>) -> (tensor<2x2xi32>) {
    // CHECK: pphlo.maximum %arg0, %arg1 : tensor<2x2xi32>
    %0 = "stablehlo.maximum"(%arg0, %arg1) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
    return %0 : tensor<2x2xi32>
}

// -----

func.func @main(%arg0: tensor<2x2xi32>, %arg1: tensor<2x2xi32>) -> (tensor<2x2xi32>) {
    // CHECK: pphlo.minimum %arg0, %arg1 : tensor<2x2xi32>
    %0 = "stablehlo.minimum"(%arg0, %arg1) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
    return %0 : tensor<2x2xi32>
}

// -----

func.func @main(%arg0: tensor<2x2xi32>, %arg1: tensor<2x2xi32>) -> (tensor<2x2xi32>) {
    // CHECK: pphlo.divide %arg0, %arg1 : tensor<2x2xi32>
    %0 = "stablehlo.divide"(%arg0, %arg1) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
    return %0 : tensor<2x2xi32>
}

// -----

func.func @main(%arg0: tensor<2x2xi32>, %arg1: tensor<2x2xi32>) -> (tensor<2x2xi32>) {
    // CHECK: pphlo.add %arg0, %arg1 : tensor<2x2xi32>
    %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
    return %0 : tensor<2x2xi32>
}

// -----

func.func @main(%arg0: tensor<2x2xi32>, %arg1: tensor<2x2xi32>) -> (tensor<2x2xi32>) {
    // CHECK: pphlo.multiply %arg0, %arg1 : tensor<2x2xi32>
    %0 = "stablehlo.multiply"(%arg0, %arg1) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
    return %0 : tensor<2x2xi32>
}

// -----

func.func @main(%arg0: tensor<2x2xi32>, %arg1: tensor<2x2xi32>) -> (tensor<2x2xi32>) {
    // CHECK: pphlo.power %arg0, %arg1 : tensor<2x2xi32>
    %0 = "stablehlo.power"(%arg0, %arg1) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
    return %0 : tensor<2x2xi32>
}

// -----

func.func @main(%arg0: tensor<2x2xi32>, %arg1: tensor<2x2xi32>) -> (tensor<2x2xi32>) {
    // CHECK: pphlo.and %arg0, %arg1 : tensor<2x2xi32>
    %0 = "stablehlo.and"(%arg0, %arg1) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
    return %0 : tensor<2x2xi32>
}

// -----

func.func @main(%arg0: tensor<2x2xi32>, %arg1: tensor<2x2xi32>) -> (tensor<2x2xi32>) {
    // CHECK: pphlo.dot %arg0, %arg1 : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
    %0 = "stablehlo.dot"(%arg0, %arg1) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
    return %0 : tensor<2x2xi32>
}