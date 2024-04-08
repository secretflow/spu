// RUN: pphlo-opt --decompose-minmax --split-input-file %s | FileCheck %s

func.func @main(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> (tensor<2x2xf32>) {
    //CHECK: %0 = pphlo.greater %arg0, %arg1 : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xi1>
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
