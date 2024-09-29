// RUN: spu-opt --lower-sfloat-to-fxp --expand-fixedpoint-approx="sigmoid_mode=seg3" --lower-pphlo-float-inputs --split-input-file %s | FileCheck %s

func.func @main(%arg0: tensor<2x2x!pphlo.secret<f32>>) -> (tensor<2x2x!pphlo.secret<f32>>) {
    //CHECK: %cst = arith.constant dense<1.000000e+00> : tensor<2x2xf32>
    //CHECK: %0 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %cst_0 = arith.constant dense<0.000000e+00> : tensor<2x2xf32>
    //CHECK: %1 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_0) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %cst_1 = arith.constant dense<4.000000e+00> : tensor<2x2xf32>
    //CHECK: %2 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_1) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %cst_2 = arith.constant dense<-4.000000e+00> : tensor<2x2xf32>
    //CHECK: %3 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_2) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %cst_3 = arith.constant dense<5.000000e-01> : tensor<2x2xf32>
    //CHECK: %4 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_3) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %cst_4 = arith.constant dense<1.250000e-01> : tensor<2x2xf32>
    //CHECK: %5 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_4) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %6 = pphlo.multiply %arg0, %5 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %7 = pphlo.truncate %6 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %8 = pphlo.add %7, %4 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %9 = pphlo.greater %arg0, %2 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.secret<i1>>
    //CHECK: %10 = pphlo.select %9, %0, %8 : (tensor<2x2x!pphlo.secret<i1>>, tensor<2x2x!pphlo.fxp<64, 18>>, tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %11 = pphlo.less %arg0, %3 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.secret<i1>>
    //CHECK: %12 = pphlo.select %11, %1, %10 : (tensor<2x2x!pphlo.secret<i1>>, tensor<2x2x!pphlo.fxp<64, 18>>, tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: return %12 : tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    %0 = pphlo.logistic %arg0 : tensor<2x2x!pphlo.secret<f32>>
    return %0 : tensor<2x2x!pphlo.secret<f32>>
}
