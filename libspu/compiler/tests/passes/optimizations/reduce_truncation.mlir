// RUN: spu-opt --reduce-truncation --split-input-file %s | FileCheck %s

func.func @main(%arg0: tensor<31x1xf32>, %arg1: tensor<31x1x!pphlo.secret<f32>>) -> (tensor<31x1x!pphlo.secret<f32>>) {
    //CHECK: %[[MUL0:.*]] = pphlo.multiply %arg0, %arg0 : tensor<31x1xf32>
    //CHECK: %[[MUL1:.*]] = pphlo.multiply %[[MUL0]], %arg1 : (tensor<31x1xf32>, tensor<31x1x!pphlo.secret<f32>>) -> tensor<31x1x!pphlo.secret<f32>>
    //CHECK: return %[[MUL1]]
    %0 = pphlo.multiply %arg0, %arg1 : (tensor<31x1xf32>, tensor<31x1x!pphlo.secret<f32>>) -> tensor<31x1x!pphlo.secret<f32>>
    %1 = pphlo.multiply %0, %arg0 : (tensor<31x1x!pphlo.secret<f32>>, tensor<31x1xf32>) -> tensor<31x1x!pphlo.secret<f32>>
    return %1 : tensor<31x1x!pphlo.secret<f32>>
}
