// RUN: spu-opt --lower-sfloat-to-fxp --expand-fixedpoint-approx="exp_mode=pade" --lower-pphlo-float-inputs --split-input-file %s | FileCheck %s

func.func @main(%arg0: tensor<2x2x!pphlo.secret<f32>>) -> (tensor<2x2x!pphlo.secret<f32>>) {
    //CHECK: %cst = arith.constant dense<32> : tensor<2x2xui64>
    //CHECK: %cst_0 = arith.constant dense<65536> : tensor<2x2xui64>
    //CHECK: %cst_1 = arith.constant dense<256> : tensor<2x2xui64>
    //CHECK: %cst_2 = arith.constant dense<3> : tensor<2x2xui64>
    //CHECK: %cst_3 = arith.constant dense<16> : tensor<2x2xui64>
    //CHECK: %cst_4 = arith.constant dense<4> : tensor<2x2xui64>
    //CHECK: %cst_5 = arith.constant dense<2> : tensor<2x2xui64>
    //CHECK: %cst_6 = arith.constant dense<0> : tensor<2x2xui64>
    //CHECK: %cst_7 = arith.constant dense<0.0013327304> : tensor<2x2xf32>
    //CHECK: %0 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_7) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %cst_8 = arith.constant dense<0.00961834099> : tensor<2x2xf32>
    //CHECK: %1 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_8) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %cst_9 = arith.constant dense<0.055504065> : tensor<2x2xf32>
    //CHECK: %2 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_9) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %cst_10 = arith.constant dense<0.240226507> : tensor<2x2xf32>
    //CHECK: %3 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_10) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %cst_11 = arith.constant dense<0.693147182> : tensor<2x2xf32>
    //CHECK: %4 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_11) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %cst_12 = arith.constant dense<1.00000012> : tensor<2x2xf32>
    //CHECK: %5 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_12) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %cst_13 = arith.constant dense<18> : tensor<2x2xui64>
    //CHECK: %cst_14 = arith.constant dense<63> : tensor<2x2xui64>
    //CHECK: %cst_15 = arith.constant dense<1> : tensor<2x2xui64>
    //CHECK: %cst_16 = arith.constant dense<1.44269502> : tensor<2x2xf32>
    //CHECK: %6 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_16) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %7 = pphlo.multiply %arg0, %6 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %8 = pphlo.truncate %7 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %9 = pphlo.bitcast_convert %8 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %10 = pphlo.shift_right_logical %9, %cst_14 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %11 = pphlo.bitcast_convert %8 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %12 = pphlo.shift_right_logical %11, %cst_13 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %13 = pphlo.shift_left %12, %cst_13 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %14 = pphlo.bitcast_convert %13 : (tensor<2x2x!pphlo.secret<ui64>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %15 = pphlo.subtract %8, %14 : tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %16 = pphlo.multiply %15, %15 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %17 = pphlo.truncate %16 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %18 = pphlo.multiply %15, %17 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %19 = pphlo.truncate %18 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %20 = pphlo.multiply %15, %19 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %21 = pphlo.truncate %20 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %22 = pphlo.multiply %15, %21 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %23 = pphlo.truncate %22 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %24 = pphlo.multiply %15, %4 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %25 = pphlo.multiply %17, %3 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %26 = pphlo.add %24, %25 : tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %27 = pphlo.multiply %19, %2 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %28 = pphlo.add %26, %27 : tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %29 = pphlo.multiply %21, %1 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %30 = pphlo.add %28, %29 : tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %31 = pphlo.multiply %23, %0 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %32 = pphlo.add %30, %31 : tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %33 = pphlo.truncate %32 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %34 = pphlo.add %33, %5 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %35 = pphlo.shift_right_logical %12, %cst_6 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %36 = pphlo.and %35, %cst_15 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %37 = pphlo.subtract %cst_15, %36 : (tensor<2x2xui64>, tensor<2x2x!pphlo.secret<ui64>>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %38 = pphlo.multiply %36, %cst_5 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %39 = pphlo.add %38, %37 : tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %40 = pphlo.multiply %34, %39 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.secret<ui64>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %41 = pphlo.shift_right_logical %12, %cst_15 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %42 = pphlo.and %41, %cst_15 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %43 = pphlo.subtract %cst_15, %42 : (tensor<2x2xui64>, tensor<2x2x!pphlo.secret<ui64>>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %44 = pphlo.multiply %42, %cst_4 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %45 = pphlo.add %44, %43 : tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %46 = pphlo.multiply %40, %45 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.secret<ui64>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %47 = pphlo.shift_right_logical %12, %cst_5 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %48 = pphlo.and %47, %cst_15 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %49 = pphlo.subtract %cst_15, %48 : (tensor<2x2xui64>, tensor<2x2x!pphlo.secret<ui64>>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %50 = pphlo.multiply %48, %cst_3 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %51 = pphlo.add %50, %49 : tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %52 = pphlo.multiply %46, %51 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.secret<ui64>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %53 = pphlo.shift_right_logical %12, %cst_2 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %54 = pphlo.and %53, %cst_15 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %55 = pphlo.subtract %cst_15, %54 : (tensor<2x2xui64>, tensor<2x2x!pphlo.secret<ui64>>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %56 = pphlo.multiply %54, %cst_1 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %57 = pphlo.add %56, %55 : tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %58 = pphlo.multiply %52, %57 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.secret<ui64>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %59 = pphlo.shift_right_logical %12, %cst_4 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %60 = pphlo.and %59, %cst_15 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %61 = pphlo.subtract %cst_15, %60 : (tensor<2x2xui64>, tensor<2x2x!pphlo.secret<ui64>>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %62 = pphlo.multiply %60, %cst_0 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %63 = pphlo.add %62, %61 : tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %64 = pphlo.multiply %58, %63 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.secret<ui64>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %65 = pphlo.bitcast_convert %64 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %66 = pphlo.shift_right_logical %65, %cst : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %67 = pphlo.bitcast_convert %66 : (tensor<2x2x!pphlo.secret<ui64>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %68 = pphlo.subtract %67, %64 : tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %69 = pphlo.multiply %10, %68 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %70 = pphlo.add %64, %69 : tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: return %70 : tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    %0 = pphlo.exponential %arg0 : tensor<2x2x!pphlo.secret<f32>>
    return %0 : tensor<2x2x!pphlo.secret<f32>>
}
