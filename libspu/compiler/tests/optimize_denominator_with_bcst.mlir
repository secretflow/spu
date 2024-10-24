// RUN: mlir-pphlo-opt --optimize-denominator-with-broadcast --split-input-file %s | FileCheck %s

func.func @main(%arg0: tensor<16x!pphlo.sec<f32>>, %arg1: tensor<16x10000x!pphlo.sec<f32>>) -> (tensor<16x10000x!pphlo.sec<f32>>) {
    //CHECK: %0 = "pphlo.reciprocal"(%arg0)
    //CHECK: %1 = "pphlo.broadcast"(%0)
    //CHECK: %2 = "pphlo.multiply"(%arg1, %1)
    //CHECK: return %2
    %0 = "pphlo.broadcast"(%arg0) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<16x!pphlo.sec<f32>>) -> tensor<16x10000x!pphlo.sec<f32>>
    %1 = "pphlo.divide"(%arg1, %0) : (tensor<16x10000x!pphlo.sec<f32>>, tensor<16x10000x!pphlo.sec<f32>>) -> tensor<16x10000x!pphlo.sec<f32>>
    return %1 : tensor<16x10000x!pphlo.sec<f32>>
}
