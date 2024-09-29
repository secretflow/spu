// RUN: spu-opt --lower-sfloat-to-fxp --expand-fixedpoint-approx --lower-pphlo-float-inputs %s | FileCheck %s

func.func @main(%arg0: tensor<2x!pphlo.secret<f32>>) -> tensor<2x!pphlo.secret<f32>> {
    //CHECK: %cst = arith.constant dense<1.000000e+00> : tensor<2xf32>
    //CHECK: %0 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst) {allow_float = true} : (tensor<2xf32>) -> tensor<2x!pphlo.fxp<64, 18>>
    //CHECK: %cst_0 = arith.constant dense<2.914200e+00> : tensor<2xf32>
    //CHECK: %1 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_0) {allow_float = true} : (tensor<2xf32>) -> tensor<2x!pphlo.fxp<64, 18>>
    //CHECK: %cst_1 = arith.constant dense<2> : tensor<2xi64>
    //CHECK: %cst_2 = arith.constant dense<68719476735> : tensor<2xui64>
    //CHECK: %cst_3 = arith.constant dense<1> : tensor<2xui64>
    //CHECK: %2 = pphlo.bitcast_convert %arg0 : (tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x!pphlo.secret<i64>>
    //CHECK: %3 = pphlo.sign %2 {ignore_zero = true} : tensor<2x!pphlo.secret<i64>>
    //CHECK: %4 = pphlo.multiply %3, %arg0 : (tensor<2x!pphlo.secret<i64>>, tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %5 = pphlo.bitcast_convert %4 : (tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x!pphlo.secret<ui64>>
    //CHECK: %6 = pphlo.prefix_or %5 : tensor<2x!pphlo.secret<ui64>>
    //CHECK: %7 = pphlo.shift_right_logical %6, %cst_3 : (tensor<2x!pphlo.secret<ui64>>, tensor<2xui64>) -> tensor<2x!pphlo.secret<ui64>>
    //CHECK: %8 = pphlo.xor %6, %7 : tensor<2x!pphlo.secret<ui64>>
    //CHECK: %9 = pphlo.bitrev %8 {end = 36 : i64, start = 0 : i64} : tensor<2x!pphlo.secret<ui64>>
    //CHECK: %10 = pphlo.and %9, %cst_2 : (tensor<2x!pphlo.secret<ui64>>, tensor<2xui64>) -> tensor<2x!pphlo.secret<ui64>>
    //CHECK: %11 = pphlo.bitcast_convert %10 : (tensor<2x!pphlo.secret<ui64>>) -> tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %12 = pphlo.multiply %4, %11 : (tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %13 = pphlo.truncate %12 {sign = #pphlo<sign_type Positive>} : (tensor<2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %14 = pphlo.multiply %cst_1, %13 : (tensor<2xi64>, tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %15 = pphlo.subtract %1, %14 : (tensor<2x!pphlo.fxp<64, 18>>, tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %16 = pphlo.multiply %13, %15 : (tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %17 = pphlo.truncate %16 {sign = #pphlo<sign_type Positive>} : (tensor<2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %18 = pphlo.subtract %0, %17 : (tensor<2x!pphlo.fxp<64, 18>>, tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %19 = pphlo.add %18, %0 : (tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x!pphlo.fxp<64, 18>>) -> tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %20 = pphlo.multiply %15, %19 : (tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %21 = pphlo.truncate %20 {sign = #pphlo<sign_type Positive>} : (tensor<2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %22 = pphlo.multiply %18, %18 : (tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %23 = pphlo.truncate %22 {sign = #pphlo<sign_type Positive>} : (tensor<2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %24 = pphlo.add %23, %0 : (tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x!pphlo.fxp<64, 18>>) -> tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %25 = pphlo.multiply %21, %24 : (tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %26 = pphlo.truncate %25 {sign = #pphlo<sign_type Positive>} : (tensor<2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %27 = pphlo.multiply %26, %11 : (tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %28 = pphlo.truncate %27 {sign = #pphlo<sign_type Positive>} : (tensor<2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %29 = pphlo.multiply %3, %28 : (tensor<2x!pphlo.secret<i64>>, tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: return %29 : tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    %0 = pphlo.reciprocal %arg0 : tensor<2x!pphlo.secret<f32>>
    return %0 : tensor<2x!pphlo.secret<f32>>
}
