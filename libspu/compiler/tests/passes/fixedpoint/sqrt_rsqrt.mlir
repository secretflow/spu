// RUN: spu-opt --lower-sfloat-to-fxp --expand-fixedpoint-approx --lower-pphlo-float-inputs --split-input-file %s | FileCheck %s

func.func @main(%arg0: tensor<2x2x!pphlo.secret<f32>>) -> (tensor<2x2x!pphlo.secret<f32>>) {
    //CHECK: %cst = arith.constant dense<512> : tensor<2x2xui64>
    //CHECK: %cst_0 = arith.constant dense<724> : tensor<2x2xui64>
    //CHECK: %cst_1 = arith.constant dense<32> : tensor<2x2xui64>
    //CHECK: %cst_2 = arith.constant dense<4294967295> : tensor<2x2xui64>
    //CHECK: %cst_3 = arith.constant dense<4.142850e+00> : tensor<2x2xf32>
    //CHECK: %0 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_3) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %cst_4 = arith.constant dense<26.0294228> : tensor<2x2xf32>
    //CHECK: %1 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_4) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %cst_5 = arith.constant dense<-49.8660583> : tensor<2x2xf32>
    //CHECK: %2 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_5) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %cst_6 = arith.constant dense<38.4714813> : tensor<2x2xf32>
    //CHECK: %3 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_6) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %cst_7 = arith.constant dense<-15.4799442> : tensor<2x2xf32>
    //CHECK: %4 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_7) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %cst_8 = arith.constant dense<0.000000e+00> : tensor<2x2xf32>
    //CHECK: %5 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_8) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %cst_9 = arith.constant dense<1> : tensor<2x2xui64>
    //CHECK: %6 = pphlo.bitcast_convert %arg0 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %7 = pphlo.bitcast_convert %6 : tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %8 = pphlo.prefix_or %7 : tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %9 = pphlo.shift_right_logical %8, %cst_9 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %10 = pphlo.xor %8, %9 : tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %11 = pphlo.shift_left %10, %cst_9 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %12 = pphlo.bitrev %11 {end = 36 : i64, start = 0 : i64} : tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %13 = pphlo.bitcast_convert %12 : (tensor<2x2x!pphlo.secret<ui64>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %14 = pphlo.multiply %arg0, %13 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %15 = pphlo.truncate %14 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %16 = pphlo.multiply %15, %4 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %17 = pphlo.add %16, %5 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %18 = pphlo.multiply %15, %15 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %19 = pphlo.truncate %18 {sign = #pphlo<sign_type Positive>} : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %20 = pphlo.multiply %19, %3 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %21 = pphlo.add %17, %20 : tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %22 = pphlo.multiply %19, %15 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %23 = pphlo.truncate %22 {sign = #pphlo<sign_type Positive>} : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %24 = pphlo.multiply %23, %2 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %25 = pphlo.add %21, %24 : tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %26 = pphlo.multiply %23, %15 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %27 = pphlo.truncate %26 {sign = #pphlo<sign_type Positive>} : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %28 = pphlo.multiply %27, %1 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %29 = pphlo.add %25, %28 : tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %30 = pphlo.truncate %29 {sign = #pphlo<sign_type Positive>} : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %31 = pphlo.add %30, %0 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %32 = pphlo.bitdeintl %11 : tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %33 = pphlo.and %32, %cst_2 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %34 = pphlo.shift_right_logical %32, %cst_1 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %35 = pphlo.and %34, %cst_2 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %36 = pphlo.xor %35, %33 : tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %37 = pphlo.bitparity %33 {bits = 32 : i64} : (tensor<2x2x!pphlo.secret<ui64>>) -> tensor<2x2x!pphlo.secret<i1>>
    //CHECK: %38 = pphlo.bitrev %36 {end = 18 : i64, start = 0 : i64} : tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %39 = pphlo.select %37, %cst_0, %cst : (tensor<2x2x!pphlo.secret<i1>>, tensor<2x2xui64>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %40 = pphlo.multiply %39, %38 : tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %41 = pphlo.bitcast_convert %40 : (tensor<2x2x!pphlo.secret<ui64>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %42 = pphlo.multiply %31, %41 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %43 = pphlo.truncate %42 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: return %43 : tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    %0 = pphlo.rsqrt %arg0 : tensor<2x2x!pphlo.secret<f32>>
    return %0 : tensor<2x2x!pphlo.secret<f32>>
}

// -----

func.func @main(%arg0: tensor<2x2x!pphlo.secret<f32>>) -> (tensor<2x2x!pphlo.secret<f32>>) {
    //CHECK: %cst = arith.constant dense<512> : tensor<2x2xui64>
    //CHECK: %cst_0 = arith.constant dense<724> : tensor<2x2xui64>
    //CHECK: %cst_1 = arith.constant dense<32> : tensor<2x2xui64>
    //CHECK: %cst_2 = arith.constant dense<4294967295> : tensor<2x2xui64>
    //CHECK: %cst_3 = arith.constant dense<4.142850e+00> : tensor<2x2xf32>
    //CHECK: %0 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_3) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %cst_4 = arith.constant dense<26.0294228> : tensor<2x2xf32>
    //CHECK: %1 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_4) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %cst_5 = arith.constant dense<-49.8660583> : tensor<2x2xf32>
    //CHECK: %2 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_5) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %cst_6 = arith.constant dense<38.4714813> : tensor<2x2xf32>
    //CHECK: %3 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_6) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %cst_7 = arith.constant dense<-15.4799442> : tensor<2x2xf32>
    //CHECK: %4 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_7) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %cst_8 = arith.constant dense<0.000000e+00> : tensor<2x2xf32>
    //CHECK: %5 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_8) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %cst_9 = arith.constant dense<1> : tensor<2x2xui64>
    //CHECK: %cst_10 = arith.constant dense<5.000000e-01> : tensor<2x2xf32>
    //CHECK: %6 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_10) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %cst_11 = arith.constant dense<1.500000e+00> : tensor<2x2xf32>
    //CHECK: %7 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_11) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %8 = pphlo.bitcast_convert %arg0 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %9 = pphlo.bitcast_convert %8 : tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %10 = pphlo.prefix_or %9 : tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %11 = pphlo.shift_right_logical %10, %cst_9 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %12 = pphlo.xor %10, %11 : tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %13 = pphlo.shift_left %12, %cst_9 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %14 = pphlo.bitrev %13 {end = 36 : i64, start = 0 : i64} : tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %15 = pphlo.bitcast_convert %14 : (tensor<2x2x!pphlo.secret<ui64>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %16 = pphlo.multiply %arg0, %15 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %17 = pphlo.truncate %16 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %18 = pphlo.multiply %17, %4 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %19 = pphlo.add %18, %5 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %20 = pphlo.multiply %17, %17 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %21 = pphlo.truncate %20 {sign = #pphlo<sign_type Positive>} : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %22 = pphlo.multiply %21, %3 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %23 = pphlo.add %19, %22 : tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %24 = pphlo.multiply %21, %17 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %25 = pphlo.truncate %24 {sign = #pphlo<sign_type Positive>} : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %26 = pphlo.multiply %25, %2 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %27 = pphlo.add %23, %26 : tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %28 = pphlo.multiply %25, %17 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %29 = pphlo.truncate %28 {sign = #pphlo<sign_type Positive>} : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %30 = pphlo.multiply %29, %1 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %31 = pphlo.add %27, %30 : tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %32 = pphlo.truncate %31 {sign = #pphlo<sign_type Positive>} : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %33 = pphlo.add %32, %0 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %34 = pphlo.bitdeintl %13 : tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %35 = pphlo.and %34, %cst_2 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %36 = pphlo.shift_right_logical %34, %cst_1 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %37 = pphlo.and %36, %cst_2 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %38 = pphlo.xor %37, %35 : tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %39 = pphlo.bitparity %35 {bits = 32 : i64} : (tensor<2x2x!pphlo.secret<ui64>>) -> tensor<2x2x!pphlo.secret<i1>>
    //CHECK: %40 = pphlo.bitrev %38 {end = 18 : i64, start = 0 : i64} : tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %41 = pphlo.select %39, %cst_0, %cst : (tensor<2x2x!pphlo.secret<i1>>, tensor<2x2xui64>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %42 = pphlo.multiply %41, %40 : tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %43 = pphlo.bitcast_convert %42 : (tensor<2x2x!pphlo.secret<ui64>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %44 = pphlo.multiply %33, %43 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %45 = pphlo.truncate %44 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %46 = pphlo.multiply %arg0, %45 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %47 = pphlo.truncate %46 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %48 = pphlo.multiply %45, %6 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %49 = pphlo.truncate %48 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %50 = pphlo.multiply %47, %49 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %51 = pphlo.truncate %50 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %52 = pphlo.subtract %7, %51 : (tensor<2x2x!pphlo.fxp<64, 18>>, tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %53 = pphlo.multiply %47, %52 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %54 = pphlo.truncate %53 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: return %54 : tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    %0 = pphlo.sqrt %arg0 : tensor<2x2x!pphlo.secret<f32>>
    return %0 : tensor<2x2x!pphlo.secret<f32>>
}
