// RUN: spu-opt --lower-sfloat-to-fxp --secret-decompose-ops --split-input-file %s | FileCheck %s

func.func @main(%arg0: tensor<2x2x!pphlo.secret<f32>>) -> (tensor<2x2x!pphlo.secret<f32>>) {
    //CHECK: %cst = arith.constant dense<18> : tensor<2x2xi64>
    //CHECK: %0 = pphlo.bitcast_convert %arg0 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<i64>>
    //CHECK: %1 = pphlo.shift_right_arithmetic %0, %cst : (tensor<2x2x!pphlo.secret<i64>>, tensor<2x2xi64>) -> tensor<2x2x!pphlo.secret<i64>>
    //CHECK: %2 = pphlo.shift_left %1, %cst : (tensor<2x2x!pphlo.secret<i64>>, tensor<2x2xi64>) -> tensor<2x2x!pphlo.secret<i64>>
    //CHECK: %3 = pphlo.bitcast_convert %2 : (tensor<2x2x!pphlo.secret<i64>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: return %3 : tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    %0 = pphlo.floor %arg0 : tensor<2x2x!pphlo.secret<f32>>
    return %0 : tensor<2x2x!pphlo.secret<f32>>
}
