// RUN: spu-opt --lower-sfloat-to-fxp --expand-fixedpoint-approx --lower-pphlo-float-inputs %s | FileCheck %s

func.func @main(%arg0: tensor<2x!pphlo.secret<f32>>) -> tensor<2x!pphlo.secret<f32>> {
    //CHECK: %cst = arith.constant dense<1.000000e+00> : tensor<2xf32>
    //CHECK: %0 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst) {allow_float = true} : (tensor<2xf32>) -> tensor<2x!pphlo.fxp<64, 18>>
    //CHECK: %cst_0 = arith.constant dense<2.914200e+00> : tensor<2xf32>
    //CHECK: %1 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_0) {allow_float = true} : (tensor<2xf32>) -> tensor<2x!pphlo.fxp<64, 18>>
    //CHECK: %cst_1 = arith.constant dense<2> : tensor<2xi64>
    //CHECK: %cst_2 = arith.constant dense<68719476735> : tensor<2xi64>
    //CHECK: %cst_3 = arith.constant dense<1> : tensor<2xi64>
    //CHECK: %2 = pphlo.bitcast_convert %arg0 : (tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x!pphlo.secret<i64>>
    //CHECK: %3 = pphlo.sign %2 {ignore_zero = true} : tensor<2x!pphlo.secret<i64>>
    //CHECK: %4 = pphlo.multiply %3, %arg0 : (tensor<2x!pphlo.secret<i64>>, tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %5 = pphlo.bitcast_convert %4 : (tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x!pphlo.secret<ui64>>
    //CHECK: %6 = pphlo.prefix_or %5 : tensor<2x!pphlo.secret<ui64>>
    //CHECK: %7 = pphlo.bitcast_convert %cst_3 : (tensor<2xi64>) -> tensor<2xui64>
    //CHECK: %8 = pphlo.shift_right_logical %6, %7 : (tensor<2x!pphlo.secret<ui64>>, tensor<2xui64>) -> tensor<2x!pphlo.secret<ui64>>
    //CHECK: %9 = pphlo.xor %6, %8 : tensor<2x!pphlo.secret<ui64>>
    //CHECK: %10 = pphlo.bitrev %9 {end = 36 : i64, start = 0 : i64} : tensor<2x!pphlo.secret<ui64>>
    //CHECK: %11 = pphlo.bitcast_convert %cst_2 : (tensor<2xi64>) -> tensor<2xui64>
    //CHECK: %12 = pphlo.and %10, %11 : (tensor<2x!pphlo.secret<ui64>>, tensor<2xui64>) -> tensor<2x!pphlo.secret<ui64>>
    //CHECK: %13 = pphlo.bitcast_convert %12 : (tensor<2x!pphlo.secret<ui64>>) -> tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %14 = pphlo.multiply %4, %13 : (tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %15 = pphlo.truncate %14 {sign = #pphlo<sign_type Positive>} : (tensor<2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %16 = pphlo.multiply %cst_1, %15 : (tensor<2xi64>, tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %17 = pphlo.subtract %1, %16 : (tensor<2x!pphlo.fxp<64, 18>>, tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %18 = pphlo.multiply %15, %17 : (tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %19 = pphlo.truncate %18 {sign = #pphlo<sign_type Positive>} : (tensor<2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %20 = pphlo.subtract %0, %19 : (tensor<2x!pphlo.fxp<64, 18>>, tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %21 = pphlo.add %20, %0 : (tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x!pphlo.fxp<64, 18>>) -> tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %22 = pphlo.multiply %17, %21 : (tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %23 = pphlo.truncate %22 {sign = #pphlo<sign_type Positive>} : (tensor<2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %24 = pphlo.multiply %20, %20 : (tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %25 = pphlo.truncate %24 {sign = #pphlo<sign_type Positive>} : (tensor<2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %26 = pphlo.add %25, %0 : (tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x!pphlo.fxp<64, 18>>) -> tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %27 = pphlo.multiply %23, %26 : (tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %28 = pphlo.truncate %27 {sign = #pphlo<sign_type Positive>} : (tensor<2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %29 = pphlo.multiply %28, %13 : (tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %30 = pphlo.truncate %29 {sign = #pphlo<sign_type Positive>} : (tensor<2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %31 = pphlo.multiply %3, %30 : (tensor<2x!pphlo.secret<i64>>, tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: return %31 : tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    %0 = pphlo.reciprocal %arg0 : tensor<2x!pphlo.secret<f32>>
    return %0 : tensor<2x!pphlo.secret<f32>>
}
