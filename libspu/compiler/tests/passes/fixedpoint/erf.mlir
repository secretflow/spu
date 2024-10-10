// RUN: spu-opt --lower-sfloat-to-fxp --expand-fixedpoint-approx --lower-pphlo-float-inputs --split-input-file %s | FileCheck %s

func.func @main(%arg0: tensor<2x!pphlo.secret<f32>>) -> tensor<2x!pphlo.secret<f32>> {
    //CHECK: %cst = arith.constant dense<2.914200e+00> : tensor<2xf32>
    //CHECK: %0 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst) {allow_float = true} : (tensor<2xf32>) -> tensor<2x!pphlo.fxp<64, 18>>
    //CHECK: %cst_0 = arith.constant dense<2> : tensor<2xi64>
    //CHECK: %cst_1 = arith.constant dense<68719476735> : tensor<2xi64>
    //CHECK: %cst_2 = arith.constant dense<1> : tensor<2xi64>
    //CHECK: %cst_3 = arith.constant dense<7.810800e-02> : tensor<2xf32>
    //CHECK: %1 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_3) {allow_float = true} : (tensor<2xf32>) -> tensor<2x!pphlo.fxp<64, 18>>
    //CHECK: %cst_4 = arith.constant dense<9.720000e-04> : tensor<2xf32>
    //CHECK: %2 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_4) {allow_float = true} : (tensor<2xf32>) -> tensor<2x!pphlo.fxp<64, 18>>
    //CHECK: %cst_5 = arith.constant dense<2.303890e-01> : tensor<2xf32>
    //CHECK: %3 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_5) {allow_float = true} : (tensor<2xf32>) -> tensor<2x!pphlo.fxp<64, 18>>
    //CHECK: %cst_6 = arith.constant dense<2.783930e-01> : tensor<2xf32>
    //CHECK: %4 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_6) {allow_float = true} : (tensor<2xf32>) -> tensor<2x!pphlo.fxp<64, 18>>
    //CHECK: %cst_7 = arith.constant dense<1.000000e+00> : tensor<2xf32>
    //CHECK: %5 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_7) {allow_float = true} : (tensor<2xf32>) -> tensor<2x!pphlo.fxp<64, 18>>
    //CHECK: %cst_8 = arith.constant dense<3.000000e+00> : tensor<2xf32>
    //CHECK: %6 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_8) {allow_float = true} : (tensor<2xf32>) -> tensor<2x!pphlo.fxp<64, 18>>
    //CHECK: %7 = pphlo.bitcast_convert %arg0 : (tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x!pphlo.secret<i64>>
    //CHECK: %8 = pphlo.sign %7 {ignore_zero = true} : tensor<2x!pphlo.secret<i64>>
    //CHECK: %9 = pphlo.multiply %8, %arg0 : (tensor<2x!pphlo.secret<i64>>, tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %10 = pphlo.less %9, %6 : (tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x!pphlo.fxp<64, 18>>) -> tensor<2x!pphlo.secret<i1>>
    //CHECK: %11 = pphlo.multiply %9, %4 : (tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x!pphlo.fxp<64, 18>>) -> tensor<2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %12 = pphlo.add %11, %5 : (tensor<2x!pphlo.secret<!pphlo.fxp<64, 36>>>, tensor<2x!pphlo.fxp<64, 18>>) -> tensor<2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %13 = pphlo.multiply %9, %9 : (tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %14 = pphlo.truncate %13 {sign = #pphlo<sign_type Positive>} : (tensor<2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %15 = pphlo.multiply %14, %3 : (tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x!pphlo.fxp<64, 18>>) -> tensor<2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %16 = pphlo.add %12, %15 : tensor<2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %17 = pphlo.multiply %14, %9 : (tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %18 = pphlo.truncate %17 {sign = #pphlo<sign_type Positive>} : (tensor<2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %19 = pphlo.multiply %18, %2 : (tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x!pphlo.fxp<64, 18>>) -> tensor<2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %20 = pphlo.add %16, %19 : tensor<2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %21 = pphlo.multiply %18, %9 : (tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %22 = pphlo.truncate %21 {sign = #pphlo<sign_type Positive>} : (tensor<2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %23 = pphlo.multiply %22, %1 : (tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x!pphlo.fxp<64, 18>>) -> tensor<2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %24 = pphlo.add %20, %23 : tensor<2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %25 = pphlo.truncate %24 {sign = #pphlo<sign_type Positive>} : (tensor<2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %26 = pphlo.multiply %25, %25 : (tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %27 = pphlo.truncate %26 {sign = #pphlo<sign_type Positive>} : (tensor<2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %28 = pphlo.multiply %27, %27 : (tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %29 = pphlo.truncate %28 {sign = #pphlo<sign_type Positive>} : (tensor<2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %30 = pphlo.bitcast_convert %29 : (tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x!pphlo.secret<ui64>>
    //CHECK: %31 = pphlo.prefix_or %30 : tensor<2x!pphlo.secret<ui64>>
    //CHECK: %32 = pphlo.bitcast_convert %cst_2 : (tensor<2xi64>) -> tensor<2xui64>
    //CHECK: %33 = pphlo.shift_right_logical %31, %32 : (tensor<2x!pphlo.secret<ui64>>, tensor<2xui64>) -> tensor<2x!pphlo.secret<ui64>>
    //CHECK: %34 = pphlo.xor %31, %33 : tensor<2x!pphlo.secret<ui64>>
    //CHECK: %35 = pphlo.bitrev %34 {end = 36 : i64, start = 0 : i64} : tensor<2x!pphlo.secret<ui64>>
    //CHECK: %36 = pphlo.bitcast_convert %cst_1 : (tensor<2xi64>) -> tensor<2xui64>
    //CHECK: %37 = pphlo.and %35, %36 : (tensor<2x!pphlo.secret<ui64>>, tensor<2xui64>) -> tensor<2x!pphlo.secret<ui64>>
    //CHECK: %38 = pphlo.bitcast_convert %37 : (tensor<2x!pphlo.secret<ui64>>) -> tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %39 = pphlo.multiply %29, %38 : (tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %40 = pphlo.truncate %39 {sign = #pphlo<sign_type Positive>} : (tensor<2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %41 = pphlo.multiply %cst_0, %40 : (tensor<2xi64>, tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %42 = pphlo.subtract %0, %41 : (tensor<2x!pphlo.fxp<64, 18>>, tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %43 = pphlo.multiply %40, %42 : (tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %44 = pphlo.truncate %43 {sign = #pphlo<sign_type Positive>} : (tensor<2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %45 = pphlo.subtract %5, %44 : (tensor<2x!pphlo.fxp<64, 18>>, tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %46 = pphlo.add %45, %5 : (tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x!pphlo.fxp<64, 18>>) -> tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %47 = pphlo.multiply %42, %46 : (tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %48 = pphlo.truncate %47 {sign = #pphlo<sign_type Positive>} : (tensor<2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %49 = pphlo.multiply %45, %45 : (tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %50 = pphlo.truncate %49 {sign = #pphlo<sign_type Positive>} : (tensor<2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %51 = pphlo.add %50, %5 : (tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x!pphlo.fxp<64, 18>>) -> tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %52 = pphlo.multiply %48, %51 : (tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %53 = pphlo.truncate %52 {sign = #pphlo<sign_type Positive>} : (tensor<2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %54 = pphlo.multiply %53, %38 : (tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %55 = pphlo.truncate %54 {sign = #pphlo<sign_type Positive>} : (tensor<2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %56 = pphlo.subtract %5, %55 : (tensor<2x!pphlo.fxp<64, 18>>, tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %57 = pphlo.select %10, %56, %5 : (tensor<2x!pphlo.secret<i1>>, tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x!pphlo.fxp<64, 18>>) -> tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %58 = pphlo.multiply %57, %8 : (tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x!pphlo.secret<i64>>) -> tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: return %58 : tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    %0 = pphlo.custom_call @mhlo.erf(%arg0) : (tensor<2x!pphlo.secret<f32>>) -> tensor<2x!pphlo.secret<f32>>
    return %0 : tensor<2x!pphlo.secret<f32>>
}
