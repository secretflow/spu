// RUN: spu-opt --lower-sfloat-to-fxp --expand-fixedpoint-approx="log_mode=minmax" --lower-pphlo-float-inputs --split-input-file %s | FileCheck %s

func.func @main(%arg0: tensor<2x2x!pphlo.secret<f32>>) -> (tensor<2x2x!pphlo.secret<f32>>) {
    //CHECK: %cst = arith.constant dense<0.693147182> : tensor<2x2xf32>
    //CHECK: %0 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %cst_0 = arith.constant dense<18> : tensor<2x2xi64>
    //CHECK: %cst_1 = arith.constant dense<19> : tensor<2x2xi64>
    //CHECK: %cst_2 = arith.constant dense<-0.00645354437> : tensor<2x2xf32>
    //CHECK: %1 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_2) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %cst_3 = arith.constant dense<0.0360884927> : tensor<2x2xf32>
    //CHECK: %2 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_3) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %cst_4 = arith.constant dense<-0.095329389> : tensor<2x2xf32>
    //CHECK: %3 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_4) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %cst_5 = arith.constant dense<0.167654067> : tensor<2x2xf32>
    //CHECK: %4 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_5) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %cst_6 = arith.constant dense<-0.240733802> : tensor<2x2xf32>
    //CHECK: %5 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_6) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %cst_7 = arith.constant dense<0.33179903> : tensor<2x2xf32>
    //CHECK: %6 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_7) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %cst_8 = arith.constant dense<-0.499874115> : tensor<2x2xf32>
    //CHECK: %7 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_8) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %cst_9 = arith.constant dense<0.999996423> : tensor<2x2xf32>
    //CHECK: %8 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_9) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %cst_10 = arith.constant dense<0.000000e+00> : tensor<2x2xf32>
    //CHECK: %9 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_10) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %cst_11 = arith.constant dense<1.000000e+00> : tensor<2x2xf32>
    //CHECK: %10 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_11) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %cst_12 = arith.constant dense<1> : tensor<2x2xui64>
    //CHECK: %11 = pphlo.bitcast_convert %arg0 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %12 = pphlo.prefix_or %11 : tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %13 = pphlo.popcnt %12 {bits = 38 : i64} : tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %14 = pphlo.bitcast_convert %13 : (tensor<2x2x!pphlo.secret<ui64>>) -> tensor<2x2x!pphlo.secret<i64>>
    //CHECK: %15 = pphlo.shift_right_logical %12, %cst_12 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %16 = pphlo.xor %12, %15 : tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %17 = pphlo.bitrev %16 {end = 37 : i64, start = 0 : i64} : tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %18 = pphlo.bitcast_convert %17 : (tensor<2x2x!pphlo.secret<ui64>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %19 = pphlo.multiply %arg0, %18 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %20 = pphlo.truncate %19 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %21 = pphlo.subtract %20, %10 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %22 = pphlo.multiply %21, %8 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %23 = pphlo.add %22, %9 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %24 = pphlo.multiply %21, %21 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %25 = pphlo.truncate %24 {sign = #pphlo<sign_type Positive>} : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %26 = pphlo.multiply %25, %7 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %27 = pphlo.add %23, %26 : tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %28 = pphlo.multiply %25, %21 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %29 = pphlo.truncate %28 {sign = #pphlo<sign_type Positive>} : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %30 = pphlo.multiply %29, %6 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %31 = pphlo.add %27, %30 : tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %32 = pphlo.multiply %29, %21 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %33 = pphlo.truncate %32 {sign = #pphlo<sign_type Positive>} : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %34 = pphlo.multiply %33, %5 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %35 = pphlo.add %31, %34 : tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %36 = pphlo.multiply %33, %21 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %37 = pphlo.truncate %36 {sign = #pphlo<sign_type Positive>} : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %38 = pphlo.multiply %37, %4 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %39 = pphlo.add %35, %38 : tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %40 = pphlo.multiply %37, %21 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %41 = pphlo.truncate %40 {sign = #pphlo<sign_type Positive>} : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %42 = pphlo.multiply %41, %3 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %43 = pphlo.add %39, %42 : tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %44 = pphlo.multiply %41, %21 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %45 = pphlo.truncate %44 {sign = #pphlo<sign_type Positive>} : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %46 = pphlo.multiply %45, %2 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %47 = pphlo.add %43, %46 : tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %48 = pphlo.multiply %45, %21 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %49 = pphlo.truncate %48 {sign = #pphlo<sign_type Positive>} : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %50 = pphlo.multiply %49, %1 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %51 = pphlo.add %47, %50 : tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %52 = pphlo.truncate %51 {sign = #pphlo<sign_type Positive>} : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %53 = pphlo.subtract %14, %cst_1 : (tensor<2x2x!pphlo.secret<i64>>, tensor<2x2xi64>) -> tensor<2x2x!pphlo.secret<i64>>
    //CHECK: %54 = pphlo.shift_left %53, %cst_0 : (tensor<2x2x!pphlo.secret<i64>>, tensor<2x2xi64>) -> tensor<2x2x!pphlo.secret<i64>>
    //CHECK: %55 = pphlo.bitcast_convert %54 : (tensor<2x2x!pphlo.secret<i64>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %56 = pphlo.multiply %55, %0 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %57 = pphlo.truncate %56 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %58 = pphlo.add %52, %57 : tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: return %58 : tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    %0 = pphlo.log %arg0 : tensor<2x2x!pphlo.secret<f32>>
    return %0 : tensor<2x2x!pphlo.secret<f32>>
}
