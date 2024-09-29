// RUN: spu-opt --lower-sfloat-to-fxp --expand-fixedpoint-approx --lower-pphlo-float-inputs --split-input-file %s | FileCheck %s

func.func @main(%arg0: tensor<2x2x!pphlo.secret<f32>>, %arg1: tensor<2x2xf32>) -> (tensor<2x2x!pphlo.secret<f32>>) {
    //CHECK: %cst = arith.constant dense<1> : tensor<2x2xi64>
    //CHECK: %cst_0 = arith.constant dense<32> : tensor<2x2xui64>
    //CHECK: %cst_1 = arith.constant dense<65536> : tensor<2x2xui64>
    //CHECK: %cst_2 = arith.constant dense<256> : tensor<2x2xui64>
    //CHECK: %cst_3 = arith.constant dense<3> : tensor<2x2xui64>
    //CHECK: %cst_4 = arith.constant dense<16> : tensor<2x2xui64>
    //CHECK: %cst_5 = arith.constant dense<4> : tensor<2x2xui64>
    //CHECK: %cst_6 = arith.constant dense<2> : tensor<2x2xui64>
    //CHECK: %cst_7 = arith.constant dense<0> : tensor<2x2xui64>
    //CHECK: %cst_8 = arith.constant dense<0.0013327304> : tensor<2x2xf32>
    //CHECK: %0 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_8) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %cst_9 = arith.constant dense<0.00961834099> : tensor<2x2xf32>
    //CHECK: %1 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_9) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %cst_10 = arith.constant dense<0.055504065> : tensor<2x2xf32>
    //CHECK: %2 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_10) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %cst_11 = arith.constant dense<0.240226507> : tensor<2x2xf32>
    //CHECK: %3 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_11) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %cst_12 = arith.constant dense<1.00000012> : tensor<2x2xf32>
    //CHECK: %4 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_12) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %cst_13 = arith.constant dense<18> : tensor<2x2xui64>
    //CHECK: %cst_14 = arith.constant dense<63> : tensor<2x2xui64>
    //CHECK: %cst_15 = arith.constant dense<1.44269502> : tensor<2x2xf32>
    //CHECK: %5 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_15) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %cst_16 = arith.constant dense<0.693147182> : tensor<2x2xf32>
    //CHECK: %6 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_16) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %cst_17 = arith.constant dense<18> : tensor<2x2xi64>
    //CHECK: %cst_18 = arith.constant dense<19> : tensor<2x2xi64>
    //CHECK: %cst_19 = arith.constant dense<-0.00645354437> : tensor<2x2xf32>
    //CHECK: %7 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_19) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %cst_20 = arith.constant dense<0.0360884927> : tensor<2x2xf32>
    //CHECK: %8 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_20) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %cst_21 = arith.constant dense<-0.095329389> : tensor<2x2xf32>
    //CHECK: %9 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_21) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %cst_22 = arith.constant dense<0.167654067> : tensor<2x2xf32>
    //CHECK: %10 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_22) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %cst_23 = arith.constant dense<-0.240733802> : tensor<2x2xf32>
    //CHECK: %11 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_23) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %cst_24 = arith.constant dense<0.33179903> : tensor<2x2xf32>
    //CHECK: %12 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_24) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %cst_25 = arith.constant dense<-0.499874115> : tensor<2x2xf32>
    //CHECK: %13 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_25) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %cst_26 = arith.constant dense<0.999996423> : tensor<2x2xf32>
    //CHECK: %14 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_26) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %cst_27 = arith.constant dense<1.000000e+00> : tensor<2x2xf32>
    //CHECK: %15 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_27) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %cst_28 = arith.constant dense<1> : tensor<2x2xui64>
    //CHECK: %cst_29 = arith.constant dense<0.000000e+00> : tensor<2x2xf32>
    //CHECK: %16 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_29) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %17 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%arg1) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %18 = pphlo.less %arg0, %16 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.secret<i1>>
    //CHECK: %19 = pphlo.negate %arg0 : tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %20 = pphlo.select %18, %19, %arg0 : (tensor<2x2x!pphlo.secret<i1>>, tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %21 = pphlo.bitcast_convert %20 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %22 = pphlo.prefix_or %21 : tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %23 = pphlo.popcnt %22 {bits = 38 : i64} : tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %24 = pphlo.bitcast_convert %23 : (tensor<2x2x!pphlo.secret<ui64>>) -> tensor<2x2x!pphlo.secret<i64>>
    //CHECK: %25 = pphlo.shift_right_logical %22, %cst_28 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %26 = pphlo.xor %22, %25 : tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %27 = pphlo.bitrev %26 {end = 37 : i64, start = 0 : i64} : tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %28 = pphlo.bitcast_convert %27 : (tensor<2x2x!pphlo.secret<ui64>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %29 = pphlo.multiply %20, %28 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %30 = pphlo.truncate %29 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %31 = pphlo.subtract %30, %15 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %32 = pphlo.multiply %31, %14 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %33 = pphlo.add %32, %16 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %34 = pphlo.multiply %31, %31 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %35 = pphlo.truncate %34 {sign = #pphlo<sign_type Positive>} : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %36 = pphlo.multiply %35, %13 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %37 = pphlo.add %33, %36 : tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %38 = pphlo.multiply %35, %31 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %39 = pphlo.truncate %38 {sign = #pphlo<sign_type Positive>} : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %40 = pphlo.multiply %39, %12 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %41 = pphlo.add %37, %40 : tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %42 = pphlo.multiply %39, %31 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %43 = pphlo.truncate %42 {sign = #pphlo<sign_type Positive>} : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %44 = pphlo.multiply %43, %11 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %45 = pphlo.add %41, %44 : tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %46 = pphlo.multiply %43, %31 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %47 = pphlo.truncate %46 {sign = #pphlo<sign_type Positive>} : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %48 = pphlo.multiply %47, %10 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %49 = pphlo.add %45, %48 : tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %50 = pphlo.multiply %47, %31 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %51 = pphlo.truncate %50 {sign = #pphlo<sign_type Positive>} : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %52 = pphlo.multiply %51, %9 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %53 = pphlo.add %49, %52 : tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %54 = pphlo.multiply %51, %31 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %55 = pphlo.truncate %54 {sign = #pphlo<sign_type Positive>} : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %56 = pphlo.multiply %55, %8 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %57 = pphlo.add %53, %56 : tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %58 = pphlo.multiply %55, %31 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %59 = pphlo.truncate %58 {sign = #pphlo<sign_type Positive>} : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %60 = pphlo.multiply %59, %7 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %61 = pphlo.add %57, %60 : tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %62 = pphlo.truncate %61 {sign = #pphlo<sign_type Positive>} : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %63 = pphlo.subtract %24, %cst_18 : (tensor<2x2x!pphlo.secret<i64>>, tensor<2x2xi64>) -> tensor<2x2x!pphlo.secret<i64>>
    //CHECK: %64 = pphlo.shift_left %63, %cst_17 : (tensor<2x2x!pphlo.secret<i64>>, tensor<2x2xi64>) -> tensor<2x2x!pphlo.secret<i64>>
    //CHECK: %65 = pphlo.bitcast_convert %64 : (tensor<2x2x!pphlo.secret<i64>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %66 = pphlo.multiply %65, %6 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %67 = pphlo.truncate %66 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %68 = pphlo.add %62, %67 : tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %69 = pphlo.multiply %17, %68 : (tensor<2x2x!pphlo.fxp<64, 18>>, tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %70 = pphlo.truncate %69 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %71 = pphlo.multiply %70, %5 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %72 = pphlo.truncate %71 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %73 = pphlo.bitcast_convert %72 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %74 = pphlo.shift_right_logical %73, %cst_14 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %75 = pphlo.bitcast_convert %72 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %76 = pphlo.shift_right_logical %75, %cst_13 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %77 = pphlo.shift_left %76, %cst_13 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %78 = pphlo.bitcast_convert %77 : (tensor<2x2x!pphlo.secret<ui64>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %79 = pphlo.subtract %72, %78 : tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %80 = pphlo.multiply %79, %79 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %81 = pphlo.truncate %80 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %82 = pphlo.multiply %79, %81 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %83 = pphlo.truncate %82 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %84 = pphlo.multiply %79, %83 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %85 = pphlo.truncate %84 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %86 = pphlo.multiply %79, %85 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %87 = pphlo.truncate %86 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %88 = pphlo.multiply %79, %6 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %89 = pphlo.multiply %81, %3 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %90 = pphlo.add %88, %89 : tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %91 = pphlo.multiply %83, %2 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %92 = pphlo.add %90, %91 : tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %93 = pphlo.multiply %85, %1 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %94 = pphlo.add %92, %93 : tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %95 = pphlo.multiply %87, %0 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %96 = pphlo.add %94, %95 : tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %97 = pphlo.truncate %96 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %98 = pphlo.add %97, %4 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %99 = pphlo.shift_right_logical %76, %cst_7 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %100 = pphlo.and %99, %cst_28 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %101 = pphlo.subtract %cst_28, %100 : (tensor<2x2xui64>, tensor<2x2x!pphlo.secret<ui64>>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %102 = pphlo.multiply %100, %cst_6 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %103 = pphlo.add %102, %101 : tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %104 = pphlo.multiply %98, %103 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.secret<ui64>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %105 = pphlo.shift_right_logical %76, %cst_28 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %106 = pphlo.and %105, %cst_28 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %107 = pphlo.subtract %cst_28, %106 : (tensor<2x2xui64>, tensor<2x2x!pphlo.secret<ui64>>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %108 = pphlo.multiply %106, %cst_5 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %109 = pphlo.add %108, %107 : tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %110 = pphlo.multiply %104, %109 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.secret<ui64>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %111 = pphlo.shift_right_logical %76, %cst_6 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %112 = pphlo.and %111, %cst_28 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %113 = pphlo.subtract %cst_28, %112 : (tensor<2x2xui64>, tensor<2x2x!pphlo.secret<ui64>>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %114 = pphlo.multiply %112, %cst_4 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %115 = pphlo.add %114, %113 : tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %116 = pphlo.multiply %110, %115 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.secret<ui64>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %117 = pphlo.shift_right_logical %76, %cst_3 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %118 = pphlo.and %117, %cst_28 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %119 = pphlo.subtract %cst_28, %118 : (tensor<2x2xui64>, tensor<2x2x!pphlo.secret<ui64>>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %120 = pphlo.multiply %118, %cst_2 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %121 = pphlo.add %120, %119 : tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %122 = pphlo.multiply %116, %121 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.secret<ui64>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %123 = pphlo.shift_right_logical %76, %cst_5 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %124 = pphlo.and %123, %cst_28 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %125 = pphlo.subtract %cst_28, %124 : (tensor<2x2xui64>, tensor<2x2x!pphlo.secret<ui64>>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %126 = pphlo.multiply %124, %cst_1 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %127 = pphlo.add %126, %125 : tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %128 = pphlo.multiply %122, %127 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.secret<ui64>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %129 = pphlo.bitcast_convert %128 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %130 = pphlo.shift_right_logical %129, %cst_0 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %131 = pphlo.bitcast_convert %130 : (tensor<2x2x!pphlo.secret<ui64>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %132 = pphlo.subtract %131, %128 : tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %133 = pphlo.multiply %74, %132 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %134 = pphlo.add %128, %133 : tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %135 = pphlo.bitcast_convert %17 : (tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2xi64>
    //CHECK: %136 = pphlo.shift_right_logical %135, %cst_17 : tensor<2x2xi64>
    //CHECK: %137 = pphlo.and %136, %cst : tensor<2x2xi64>
    //CHECK: %138 = pphlo.convert %137 : (tensor<2x2xi64>) -> tensor<2x2x!pphlo.secret<i1>>
    //CHECK: %139 = pphlo.and %18, %138 : tensor<2x2x!pphlo.secret<i1>>
    //CHECK: %140 = pphlo.negate %134 : tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %141 = pphlo.select %139, %140, %134 : (tensor<2x2x!pphlo.secret<i1>>, tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: return %141 : tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    %0 = pphlo.power %arg0, %arg1 : (tensor<2x2x!pphlo.secret<f32>>, tensor<2x2xf32>) -> tensor<2x2x!pphlo.secret<f32>>
    return %0 : tensor<2x2x!pphlo.secret<f32>>
}

// -----

func.func @main(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2x!pphlo.secret<f32>>) -> (tensor<2x2x!pphlo.secret<f32>>) {
    //CHECK-NOT: pphlo.power
    %0 = pphlo.power %arg0, %arg1 : (tensor<2x2xf32>, tensor<2x2x!pphlo.secret<f32>>) -> tensor<2x2x!pphlo.secret<f32>>
    return %0 : tensor<2x2x!pphlo.secret<f32>>
}

// -----

func.func @main(%arg0: tensor<2x2x!pphlo.secret<f32>>, %arg1: tensor<2x2xf32>) -> (tensor<2x2x!pphlo.secret<f32>>) {
    //CHECK-NOT: pphlo.power
    %0 = pphlo.power %arg0, %arg1 : (tensor<2x2x!pphlo.secret<f32>>, tensor<2x2xf32>) -> tensor<2x2x!pphlo.secret<f32>>
    return %0 : tensor<2x2x!pphlo.secret<f32>>
}
