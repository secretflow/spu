// RUN: mlir-pphlo-opt --reduce-truncation --split-input-file %s | FileCheck %s

func.func @main(%arg0: tensor<31x1xf32>, %arg1: tensor<31x1x!pphlo.secret<f32>>) -> (tensor<31x1x!pphlo.secret<f32>>) {
    //CHECK: %0 = pphlo.multiply %arg0, %arg0 : tensor<31x1xf32>
    //CHECK: %1 = pphlo.multiply %0, %arg1 : (tensor<31x1xf32>, tensor<31x1x!pphlo.secret<f32>>) -> tensor<31x1x!pphlo.secret<f32>>
    %0 = pphlo.multiply %arg0, %arg1 : (tensor<31x1xf32>, tensor<31x1x!pphlo.secret<f32>>) -> tensor<31x1x!pphlo.secret<f32>>
    %1 = pphlo.multiply %0, %arg0 : (tensor<31x1x!pphlo.secret<f32>>, tensor<31x1xf32>) -> tensor<31x1x!pphlo.secret<f32>>
    return %1 : tensor<31x1x!pphlo.secret<f32>>
}
