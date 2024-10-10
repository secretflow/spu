// RUN: spu-opt --lower-sfloat-to-fxp --expand-fixedpoint-approx --lower-pphlo-float-inputs --split-input-file %s | FileCheck %s

func.func @main(%arg0: tensor<2x2x!pphlo.secret<f32>>, %arg1: tensor<2x2xf32>) -> (tensor<2x2x!pphlo.secret<f32>>) {
    //CHECK: %cst = arith.constant dense<32> : tensor<2x2xi64>
    //CHECK: %cst_0 = arith.constant dense<65536> : tensor<2x2xi64>
    //CHECK: %cst_1 = arith.constant dense<256> : tensor<2x2xi64>
    //CHECK: %cst_2 = arith.constant dense<3> : tensor<2x2xi64>
    //CHECK: %cst_3 = arith.constant dense<16> : tensor<2x2xi64>
    //CHECK: %cst_4 = arith.constant dense<4> : tensor<2x2xi64>
    //CHECK: %cst_5 = arith.constant dense<2> : tensor<2x2xi64>
    //CHECK: %cst_6 = arith.constant dense<0> : tensor<2x2xi64>
    //CHECK: %cst_7 = arith.constant dense<0.0013327304> : tensor<2x2xf32>
    //CHECK: %0 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_7) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %cst_8 = arith.constant dense<0.00961834099> : tensor<2x2xf32>
    //CHECK: %1 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_8) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %cst_9 = arith.constant dense<0.055504065> : tensor<2x2xf32>
    //CHECK: %2 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_9) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %cst_10 = arith.constant dense<0.240226507> : tensor<2x2xf32>
    //CHECK: %3 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_10) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %cst_11 = arith.constant dense<1.00000012> : tensor<2x2xf32>
    //CHECK: %4 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_11) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %cst_12 = arith.constant dense<63> : tensor<2x2xi64>
    //CHECK: %cst_13 = arith.constant dense<1.44269502> : tensor<2x2xf32>
    //CHECK: %5 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_13) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %cst_14 = arith.constant dense<0.693147182> : tensor<2x2xf32>
    //CHECK: %6 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_14) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %cst_15 = arith.constant dense<18> : tensor<2x2xi64>
    //CHECK: %cst_16 = arith.constant dense<19> : tensor<2x2xi64>
    //CHECK: %cst_17 = arith.constant dense<-0.00645354437> : tensor<2x2xf32>
    //CHECK: %7 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_17) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %cst_18 = arith.constant dense<0.0360884927> : tensor<2x2xf32>
    //CHECK: %8 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_18) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %cst_19 = arith.constant dense<-0.095329389> : tensor<2x2xf32>
    //CHECK: %9 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_19) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %cst_20 = arith.constant dense<0.167654067> : tensor<2x2xf32>
    //CHECK: %10 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_20) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %cst_21 = arith.constant dense<-0.240733802> : tensor<2x2xf32>
    //CHECK: %11 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_21) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %cst_22 = arith.constant dense<0.33179903> : tensor<2x2xf32>
    //CHECK: %12 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_22) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %cst_23 = arith.constant dense<-0.499874115> : tensor<2x2xf32>
    //CHECK: %13 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_23) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %cst_24 = arith.constant dense<0.999996423> : tensor<2x2xf32>
    //CHECK: %14 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_24) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %cst_25 = arith.constant dense<1.000000e+00> : tensor<2x2xf32>
    //CHECK: %15 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_25) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %cst_26 = arith.constant dense<1> : tensor<2x2xi64>
    //CHECK: %cst_27 = arith.constant dense<0.000000e+00> : tensor<2x2xf32>
    //CHECK: %16 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_27) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %17 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%arg1) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %18 = pphlo.less %arg0, %16 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.secret<i1>>
    //CHECK: %19 = pphlo.negate %arg0 : tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %20 = pphlo.select %18, %19, %arg0 : (tensor<2x2x!pphlo.secret<i1>>, tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %21 = pphlo.bitcast_convert %20 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %22 = pphlo.prefix_or %21 : tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %23 = pphlo.popcnt %22 {bits = 38 : i64} : tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %24 = pphlo.bitcast_convert %23 : (tensor<2x2x!pphlo.secret<ui64>>) -> tensor<2x2x!pphlo.secret<i64>>
    //CHECK: %25 = pphlo.bitcast_convert %cst_26 : (tensor<2x2xi64>) -> tensor<2x2xui64>
    //CHECK: %26 = pphlo.shift_right_logical %22, %25 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %27 = pphlo.xor %22, %26 : tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %28 = pphlo.bitrev %27 {end = 37 : i64, start = 0 : i64} : tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %29 = pphlo.bitcast_convert %28 : (tensor<2x2x!pphlo.secret<ui64>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %30 = pphlo.multiply %20, %29 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %31 = pphlo.truncate %30 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %32 = pphlo.subtract %31, %15 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %33 = pphlo.multiply %32, %14 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %34 = pphlo.add %33, %16 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %35 = pphlo.multiply %32, %32 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %36 = pphlo.truncate %35 {sign = #pphlo<sign_type Positive>} : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %37 = pphlo.multiply %36, %13 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %38 = pphlo.add %34, %37 : tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %39 = pphlo.multiply %36, %32 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %40 = pphlo.truncate %39 {sign = #pphlo<sign_type Positive>} : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %41 = pphlo.multiply %40, %12 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %42 = pphlo.add %38, %41 : tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %43 = pphlo.multiply %40, %32 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %44 = pphlo.truncate %43 {sign = #pphlo<sign_type Positive>} : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %45 = pphlo.multiply %44, %11 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %46 = pphlo.add %42, %45 : tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %47 = pphlo.multiply %44, %32 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %48 = pphlo.truncate %47 {sign = #pphlo<sign_type Positive>} : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %49 = pphlo.multiply %48, %10 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %50 = pphlo.add %46, %49 : tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %51 = pphlo.multiply %48, %32 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %52 = pphlo.truncate %51 {sign = #pphlo<sign_type Positive>} : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %53 = pphlo.multiply %52, %9 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %54 = pphlo.add %50, %53 : tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %55 = pphlo.multiply %52, %32 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %56 = pphlo.truncate %55 {sign = #pphlo<sign_type Positive>} : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %57 = pphlo.multiply %56, %8 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %58 = pphlo.add %54, %57 : tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %59 = pphlo.multiply %56, %32 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %60 = pphlo.truncate %59 {sign = #pphlo<sign_type Positive>} : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %61 = pphlo.multiply %60, %7 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %62 = pphlo.add %58, %61 : tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %63 = pphlo.truncate %62 {sign = #pphlo<sign_type Positive>} : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %64 = pphlo.subtract %24, %cst_16 : (tensor<2x2x!pphlo.secret<i64>>, tensor<2x2xi64>) -> tensor<2x2x!pphlo.secret<i64>>
    //CHECK: %65 = pphlo.bitcast_convert %cst_15 : tensor<2x2xi64>
    //CHECK: %66 = pphlo.shift_left %64, %65 : (tensor<2x2x!pphlo.secret<i64>>, tensor<2x2xi64>) -> tensor<2x2x!pphlo.secret<i64>>
    //CHECK: %67 = pphlo.bitcast_convert %66 : (tensor<2x2x!pphlo.secret<i64>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %68 = pphlo.multiply %67, %6 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %69 = pphlo.truncate %68 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %70 = pphlo.add %63, %69 : tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %71 = pphlo.multiply %17, %70 : (tensor<2x2x!pphlo.fxp<64, 18>>, tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %72 = pphlo.truncate %71 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %73 = pphlo.multiply %72, %5 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %74 = pphlo.truncate %73 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %75 = pphlo.bitcast_convert %cst_26 : (tensor<2x2xi64>) -> tensor<2x2xui64>
    //CHECK: %76 = pphlo.bitcast_convert %74 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %77 = pphlo.bitcast_convert %cst_12 : (tensor<2x2xi64>) -> tensor<2x2xui64>
    //CHECK: %78 = pphlo.shift_right_logical %76, %77 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %79 = pphlo.bitcast_convert %74 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %80 = pphlo.bitcast_convert %cst_15 : (tensor<2x2xi64>) -> tensor<2x2xui64>
    //CHECK: %81 = pphlo.shift_right_logical %79, %80 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %82 = pphlo.bitcast_convert %cst_15 : (tensor<2x2xi64>) -> tensor<2x2xui64>
    //CHECK: %83 = pphlo.shift_left %81, %82 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %84 = pphlo.bitcast_convert %83 : (tensor<2x2x!pphlo.secret<ui64>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %85 = pphlo.subtract %74, %84 : tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %86 = pphlo.multiply %85, %85 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %87 = pphlo.truncate %86 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %88 = pphlo.multiply %85, %87 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %89 = pphlo.truncate %88 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %90 = pphlo.multiply %85, %89 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %91 = pphlo.truncate %90 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %92 = pphlo.multiply %85, %91 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %93 = pphlo.truncate %92 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %94 = pphlo.multiply %85, %6 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %95 = pphlo.multiply %87, %3 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %96 = pphlo.add %94, %95 : tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %97 = pphlo.multiply %89, %2 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %98 = pphlo.add %96, %97 : tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %99 = pphlo.multiply %91, %1 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %100 = pphlo.add %98, %99 : tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %101 = pphlo.multiply %93, %0 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %102 = pphlo.add %100, %101 : tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %103 = pphlo.truncate %102 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %104 = pphlo.add %103, %4 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %105 = pphlo.bitcast_convert %cst_6 : (tensor<2x2xi64>) -> tensor<2x2xui64>
    //CHECK: %106 = pphlo.shift_right_logical %81, %105 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %107 = pphlo.and %106, %75 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %108 = pphlo.bitcast_convert %cst_5 : (tensor<2x2xi64>) -> tensor<2x2xui64>
    //CHECK: %109 = pphlo.subtract %75, %107 : (tensor<2x2xui64>, tensor<2x2x!pphlo.secret<ui64>>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %110 = pphlo.multiply %107, %108 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %111 = pphlo.add %110, %109 : tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %112 = pphlo.multiply %104, %111 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.secret<ui64>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %113 = pphlo.bitcast_convert %cst_26 : (tensor<2x2xi64>) -> tensor<2x2xui64>
    //CHECK: %114 = pphlo.shift_right_logical %81, %113 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %115 = pphlo.and %114, %75 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %116 = pphlo.bitcast_convert %cst_4 : (tensor<2x2xi64>) -> tensor<2x2xui64>
    //CHECK: %117 = pphlo.subtract %75, %115 : (tensor<2x2xui64>, tensor<2x2x!pphlo.secret<ui64>>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %118 = pphlo.multiply %115, %116 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %119 = pphlo.add %118, %117 : tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %120 = pphlo.multiply %112, %119 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.secret<ui64>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %121 = pphlo.bitcast_convert %cst_5 : (tensor<2x2xi64>) -> tensor<2x2xui64>
    //CHECK: %122 = pphlo.shift_right_logical %81, %121 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %123 = pphlo.and %122, %75 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %124 = pphlo.bitcast_convert %cst_3 : (tensor<2x2xi64>) -> tensor<2x2xui64>
    //CHECK: %125 = pphlo.subtract %75, %123 : (tensor<2x2xui64>, tensor<2x2x!pphlo.secret<ui64>>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %126 = pphlo.multiply %123, %124 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %127 = pphlo.add %126, %125 : tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %128 = pphlo.multiply %120, %127 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.secret<ui64>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %129 = pphlo.bitcast_convert %cst_2 : (tensor<2x2xi64>) -> tensor<2x2xui64>
    //CHECK: %130 = pphlo.shift_right_logical %81, %129 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %131 = pphlo.and %130, %75 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %132 = pphlo.bitcast_convert %cst_1 : (tensor<2x2xi64>) -> tensor<2x2xui64>
    //CHECK: %133 = pphlo.subtract %75, %131 : (tensor<2x2xui64>, tensor<2x2x!pphlo.secret<ui64>>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %134 = pphlo.multiply %131, %132 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %135 = pphlo.add %134, %133 : tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %136 = pphlo.multiply %128, %135 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.secret<ui64>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %137 = pphlo.bitcast_convert %cst_4 : (tensor<2x2xi64>) -> tensor<2x2xui64>
    //CHECK: %138 = pphlo.shift_right_logical %81, %137 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %139 = pphlo.and %138, %75 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %140 = pphlo.bitcast_convert %cst_0 : (tensor<2x2xi64>) -> tensor<2x2xui64>
    //CHECK: %141 = pphlo.subtract %75, %139 : (tensor<2x2xui64>, tensor<2x2x!pphlo.secret<ui64>>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %142 = pphlo.multiply %139, %140 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %143 = pphlo.add %142, %141 : tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %144 = pphlo.multiply %136, %143 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.secret<ui64>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %145 = pphlo.bitcast_convert %144 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %146 = pphlo.bitcast_convert %cst : (tensor<2x2xi64>) -> tensor<2x2xui64>
    //CHECK: %147 = pphlo.shift_right_logical %145, %146 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %148 = pphlo.bitcast_convert %147 : (tensor<2x2x!pphlo.secret<ui64>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %149 = pphlo.subtract %148, %144 : tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %150 = pphlo.multiply %78, %149 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %151 = pphlo.add %144, %150 : tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %152 = pphlo.bitcast_convert %17 : (tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2xi64>
    //CHECK: %153 = pphlo.bitcast_convert %cst_15 : tensor<2x2xi64>
    //CHECK: %154 = pphlo.shift_right_logical %152, %153 : tensor<2x2xi64>
    //CHECK: %155 = pphlo.and %154, %cst_26 : tensor<2x2xi64>
    //CHECK: %156 = pphlo.convert %155 : (tensor<2x2xi64>) -> tensor<2x2x!pphlo.secret<i1>>
    //CHECK: %157 = pphlo.and %18, %156 : tensor<2x2x!pphlo.secret<i1>>
    //CHECK: %158 = pphlo.negate %151 : tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %159 = pphlo.select %157, %158, %151 : (tensor<2x2x!pphlo.secret<i1>>, tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: return %159 : tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
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
