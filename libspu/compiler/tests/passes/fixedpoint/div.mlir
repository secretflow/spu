// RUN: spu-opt --lower-sfloat-to-fxp --expand-fixedpoint-approx --lower-pphlo-float-inputs --split-input-file %s | FileCheck %s

func.func @main(%arg0: tensor<2x2x!pphlo.secret<f32>>, %arg1: tensor<2x2xf32>) -> (tensor<2x2x!pphlo.secret<f32>>) {
    //CHECK: %cst = arith.constant dense<1.000000e+00> : tensor<2x2xf32>
    //CHECK: %0 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %cst_0 = arith.constant dense<2.914200e+00> : tensor<2x2xf32>
    //CHECK: %1 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_0) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %cst_1 = arith.constant dense<2> : tensor<2x2xi64>
    //CHECK: %cst_2 = arith.constant dense<68719476735> : tensor<2x2xui64>
    //CHECK: %cst_3 = arith.constant dense<1> : tensor<2x2xui64>
    //CHECK: %2 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%arg1) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %3 = pphlo.bitcast_convert %2 : (tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2xi64>
    //CHECK: %4 = pphlo.sign %3 {ignore_zero = true} : tensor<2x2xi64>
    //CHECK: %5 = pphlo.multiply %4, %2 : (tensor<2x2xi64>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %6 = pphlo.bitcast_convert %5 : (tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2xui64>
    //CHECK: %7 = pphlo.prefix_or %6 : tensor<2x2xui64>
    //CHECK: %8 = pphlo.shift_right_logical %7, %cst_3 : tensor<2x2xui64>
    //CHECK: %9 = pphlo.xor %7, %8 : tensor<2x2xui64>
    //CHECK: %10 = pphlo.bitrev %9 {end = 36 : i64, start = 0 : i64} : tensor<2x2xui64>
    //CHECK: %11 = pphlo.and %10, %cst_2 : tensor<2x2xui64>
    //CHECK: %12 = pphlo.bitcast_convert %11 : (tensor<2x2xui64>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %13 = pphlo.multiply %5, %12 : (tensor<2x2x!pphlo.fxp<64, 18>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.fxp<64, 36>>
    //CHECK: %14 = pphlo.truncate %13 {sign = #pphlo<sign_type Positive>} : (tensor<2x2x!pphlo.fxp<64, 36>>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %15 = pphlo.multiply %cst_1, %14 : (tensor<2x2xi64>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %16 = pphlo.subtract %1, %15 : tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %17 = pphlo.multiply %14, %16 : (tensor<2x2x!pphlo.fxp<64, 18>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.fxp<64, 36>>
    //CHECK: %18 = pphlo.truncate %17 {sign = #pphlo<sign_type Positive>} : (tensor<2x2x!pphlo.fxp<64, 36>>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %19 = pphlo.subtract %0, %18 : tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %20 = pphlo.add %19, %0 : tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %21 = pphlo.multiply %16, %20 : (tensor<2x2x!pphlo.fxp<64, 18>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.fxp<64, 36>>
    //CHECK: %22 = pphlo.truncate %21 {sign = #pphlo<sign_type Positive>} : (tensor<2x2x!pphlo.fxp<64, 36>>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %23 = pphlo.multiply %19, %19 : (tensor<2x2x!pphlo.fxp<64, 18>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.fxp<64, 36>>
    //CHECK: %24 = pphlo.truncate %23 {sign = #pphlo<sign_type Positive>} : (tensor<2x2x!pphlo.fxp<64, 36>>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %25 = pphlo.add %24, %0 : tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %26 = pphlo.multiply %22, %25 : (tensor<2x2x!pphlo.fxp<64, 18>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.fxp<64, 36>>
    //CHECK: %27 = pphlo.truncate %26 {sign = #pphlo<sign_type Positive>} : (tensor<2x2x!pphlo.fxp<64, 36>>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %28 = pphlo.multiply %27, %arg0 : (tensor<2x2x!pphlo.fxp<64, 18>>, tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %29 = pphlo.truncate %28 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %30 = pphlo.multiply %29, %12 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %31 = pphlo.truncate %30 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %32 = pphlo.multiply %31, %4 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2xi64>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: return %32 : tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    %0 = pphlo.divide %arg0, %arg1 : (tensor<2x2x!pphlo.secret<f32>>, tensor<2x2xf32>) -> tensor<2x2x!pphlo.secret<f32>>
    return %0 : tensor<2x2x!pphlo.secret<f32>>
}

// -----

func.func @main(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2x!pphlo.secret<f32>>) -> (tensor<2x2x!pphlo.secret<f32>>) {
    //CHECK: %cst = arith.constant dense<1.000000e+00> : tensor<2x2xf32>
    //CHECK: %0 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %cst_0 = arith.constant dense<2.914200e+00> : tensor<2x2xf32>
    //CHECK: %1 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_0) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %cst_1 = arith.constant dense<2> : tensor<2x2xi64>
    //CHECK: %cst_2 = arith.constant dense<68719476735> : tensor<2x2xui64>
    //CHECK: %cst_3 = arith.constant dense<1> : tensor<2x2xui64>
    //CHECK: %2 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%arg0) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %3 = pphlo.bitcast_convert %arg1 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<i64>>
    //CHECK: %4 = pphlo.sign %3 {ignore_zero = true} : tensor<2x2x!pphlo.secret<i64>>
    //CHECK: %5 = pphlo.multiply %4, %arg1 : (tensor<2x2x!pphlo.secret<i64>>, tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %6 = pphlo.bitcast_convert %5 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %7 = pphlo.prefix_or %6 : tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %8 = pphlo.shift_right_logical %7, %cst_3 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %9 = pphlo.xor %7, %8 : tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %10 = pphlo.bitrev %9 {end = 36 : i64, start = 0 : i64} : tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %11 = pphlo.and %10, %cst_2 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %12 = pphlo.bitcast_convert %11 : (tensor<2x2x!pphlo.secret<ui64>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %13 = pphlo.multiply %5, %12 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %14 = pphlo.truncate %13 {sign = #pphlo<sign_type Positive>} : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %15 = pphlo.multiply %cst_1, %14 : (tensor<2x2xi64>, tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %16 = pphlo.subtract %1, %15 : (tensor<2x2x!pphlo.fxp<64, 18>>, tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %17 = pphlo.multiply %14, %16 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %18 = pphlo.truncate %17 {sign = #pphlo<sign_type Positive>} : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %19 = pphlo.subtract %0, %18 : (tensor<2x2x!pphlo.fxp<64, 18>>, tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %20 = pphlo.add %19, %0 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %21 = pphlo.multiply %16, %20 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %22 = pphlo.truncate %21 {sign = #pphlo<sign_type Positive>} : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %23 = pphlo.multiply %19, %19 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %24 = pphlo.truncate %23 {sign = #pphlo<sign_type Positive>} : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %25 = pphlo.add %24, %0 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %26 = pphlo.multiply %22, %25 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %27 = pphlo.truncate %26 {sign = #pphlo<sign_type Positive>} : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %28 = pphlo.multiply %27, %2 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %29 = pphlo.truncate %28 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %30 = pphlo.multiply %29, %12 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %31 = pphlo.truncate %30 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %32 = pphlo.multiply %31, %4 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.secret<i64>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: return %32 : tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    %0 = pphlo.divide %arg0, %arg1 : (tensor<2x2xf32>, tensor<2x2x!pphlo.secret<f32>>) -> tensor<2x2x!pphlo.secret<f32>>
    return %0 : tensor<2x2x!pphlo.secret<f32>>
}

// -----

func.func @main(%arg0: tensor<2x2x!pphlo.secret<f32>>, %arg1: tensor<2x2xf32>) -> (tensor<2x2x!pphlo.secret<f32>>) {
    //CHECK: %cst = arith.constant dense<1.000000e+00> : tensor<2x2xf32>
    //CHECK: %0 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %cst_0 = arith.constant dense<2.914200e+00> : tensor<2x2xf32>
    //CHECK: %1 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_0) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %cst_1 = arith.constant dense<2> : tensor<2x2xi64>
    //CHECK: %cst_2 = arith.constant dense<68719476735> : tensor<2x2xui64>
    //CHECK: %cst_3 = arith.constant dense<1> : tensor<2x2xui64>
    //CHECK: %2 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%arg1) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %3 = pphlo.bitcast_convert %2 : (tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2xi64>
    //CHECK: %4 = pphlo.sign %3 {ignore_zero = true} : tensor<2x2xi64>
    //CHECK: %5 = pphlo.multiply %4, %2 : (tensor<2x2xi64>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %6 = pphlo.bitcast_convert %5 : (tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2xui64>
    //CHECK: %7 = pphlo.prefix_or %6 : tensor<2x2xui64>
    //CHECK: %8 = pphlo.shift_right_logical %7, %cst_3 : tensor<2x2xui64>
    //CHECK: %9 = pphlo.xor %7, %8 : tensor<2x2xui64>
    //CHECK: %10 = pphlo.bitrev %9 {end = 36 : i64, start = 0 : i64} : tensor<2x2xui64>
    //CHECK: %11 = pphlo.and %10, %cst_2 : tensor<2x2xui64>
    //CHECK: %12 = pphlo.bitcast_convert %11 : (tensor<2x2xui64>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %13 = pphlo.multiply %5, %12 : (tensor<2x2x!pphlo.fxp<64, 18>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.fxp<64, 36>>
    //CHECK: %14 = pphlo.truncate %13 {sign = #pphlo<sign_type Positive>} : (tensor<2x2x!pphlo.fxp<64, 36>>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %15 = pphlo.multiply %cst_1, %14 : (tensor<2x2xi64>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %16 = pphlo.subtract %1, %15 : tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %17 = pphlo.multiply %14, %16 : (tensor<2x2x!pphlo.fxp<64, 18>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.fxp<64, 36>>
    //CHECK: %18 = pphlo.truncate %17 {sign = #pphlo<sign_type Positive>} : (tensor<2x2x!pphlo.fxp<64, 36>>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %19 = pphlo.subtract %0, %18 : tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %20 = pphlo.add %19, %0 : tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %21 = pphlo.multiply %16, %20 : (tensor<2x2x!pphlo.fxp<64, 18>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.fxp<64, 36>>
    //CHECK: %22 = pphlo.truncate %21 {sign = #pphlo<sign_type Positive>} : (tensor<2x2x!pphlo.fxp<64, 36>>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %23 = pphlo.multiply %19, %19 : (tensor<2x2x!pphlo.fxp<64, 18>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.fxp<64, 36>>
    //CHECK: %24 = pphlo.truncate %23 {sign = #pphlo<sign_type Positive>} : (tensor<2x2x!pphlo.fxp<64, 36>>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %25 = pphlo.add %24, %0 : tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %26 = pphlo.multiply %22, %25 : (tensor<2x2x!pphlo.fxp<64, 18>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.fxp<64, 36>>
    //CHECK: %27 = pphlo.truncate %26 {sign = #pphlo<sign_type Positive>} : (tensor<2x2x!pphlo.fxp<64, 36>>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %28 = pphlo.multiply %27, %arg0 : (tensor<2x2x!pphlo.fxp<64, 18>>, tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %29 = pphlo.truncate %28 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %30 = pphlo.multiply %29, %12 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %31 = pphlo.truncate %30 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %32 = pphlo.multiply %31, %4 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2xi64>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: return %32 : tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    %0 = pphlo.divide %arg0, %arg1 : (tensor<2x2x!pphlo.secret<f32>>, tensor<2x2xf32>) -> tensor<2x2x!pphlo.secret<f32>>
    return %0 : tensor<2x2x!pphlo.secret<f32>>
}
