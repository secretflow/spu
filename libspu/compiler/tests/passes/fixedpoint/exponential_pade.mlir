// RUN: spu-opt --lower-sfloat-to-fxp --expand-fixedpoint-approx="exp_mode=pade" --lower-pphlo-float-inputs --split-input-file %s | FileCheck %s

func.func @main(%arg0: tensor<2x2x!pphlo.secret<f32>>) -> (tensor<2x2x!pphlo.secret<f32>>) {
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
    //CHECK: %cst_11 = arith.constant dense<0.693147182> : tensor<2x2xf32>
    //CHECK: %4 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_11) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %cst_12 = arith.constant dense<1.00000012> : tensor<2x2xf32>
    //CHECK: %5 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_12) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %cst_13 = arith.constant dense<18> : tensor<2x2xi64>
    //CHECK: %cst_14 = arith.constant dense<63> : tensor<2x2xi64>
    //CHECK: %cst_15 = arith.constant dense<1> : tensor<2x2xi64>
    //CHECK: %cst_16 = arith.constant dense<1.44269502> : tensor<2x2xf32>
    //CHECK: %6 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_16) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %7 = pphlo.multiply %arg0, %6 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %8 = pphlo.truncate %7 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %9 = pphlo.bitcast_convert %cst_15 : (tensor<2x2xi64>) -> tensor<2x2xui64>
    //CHECK: %10 = pphlo.bitcast_convert %8 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %11 = pphlo.bitcast_convert %cst_14 : (tensor<2x2xi64>) -> tensor<2x2xui64>
    //CHECK: %12 = pphlo.shift_right_logical %10, %11 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %13 = pphlo.bitcast_convert %8 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %14 = pphlo.bitcast_convert %cst_13 : (tensor<2x2xi64>) -> tensor<2x2xui64>
    //CHECK: %15 = pphlo.shift_right_logical %13, %14 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %16 = pphlo.bitcast_convert %cst_13 : (tensor<2x2xi64>) -> tensor<2x2xui64>
    //CHECK: %17 = pphlo.shift_left %15, %16 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %18 = pphlo.bitcast_convert %17 : (tensor<2x2x!pphlo.secret<ui64>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %19 = pphlo.subtract %8, %18 : tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %20 = pphlo.multiply %19, %19 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %21 = pphlo.truncate %20 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %22 = pphlo.multiply %19, %21 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %23 = pphlo.truncate %22 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %24 = pphlo.multiply %19, %23 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %25 = pphlo.truncate %24 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %26 = pphlo.multiply %19, %25 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %27 = pphlo.truncate %26 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %28 = pphlo.multiply %19, %4 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %29 = pphlo.multiply %21, %3 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %30 = pphlo.add %28, %29 : tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %31 = pphlo.multiply %23, %2 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %32 = pphlo.add %30, %31 : tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %33 = pphlo.multiply %25, %1 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %34 = pphlo.add %32, %33 : tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %35 = pphlo.multiply %27, %0 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %36 = pphlo.add %34, %35 : tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %37 = pphlo.truncate %36 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %38 = pphlo.add %37, %5 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %39 = pphlo.bitcast_convert %cst_6 : (tensor<2x2xi64>) -> tensor<2x2xui64>
    //CHECK: %40 = pphlo.shift_right_logical %15, %39 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %41 = pphlo.and %40, %9 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %42 = pphlo.bitcast_convert %cst_5 : (tensor<2x2xi64>) -> tensor<2x2xui64>
    //CHECK: %43 = pphlo.subtract %9, %41 : (tensor<2x2xui64>, tensor<2x2x!pphlo.secret<ui64>>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %44 = pphlo.multiply %41, %42 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %45 = pphlo.add %44, %43 : tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %46 = pphlo.multiply %38, %45 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.secret<ui64>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %47 = pphlo.bitcast_convert %cst_15 : (tensor<2x2xi64>) -> tensor<2x2xui64>
    //CHECK: %48 = pphlo.shift_right_logical %15, %47 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %49 = pphlo.and %48, %9 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %50 = pphlo.bitcast_convert %cst_4 : (tensor<2x2xi64>) -> tensor<2x2xui64>
    //CHECK: %51 = pphlo.subtract %9, %49 : (tensor<2x2xui64>, tensor<2x2x!pphlo.secret<ui64>>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %52 = pphlo.multiply %49, %50 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %53 = pphlo.add %52, %51 : tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %54 = pphlo.multiply %46, %53 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.secret<ui64>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %55 = pphlo.bitcast_convert %cst_5 : (tensor<2x2xi64>) -> tensor<2x2xui64>
    //CHECK: %56 = pphlo.shift_right_logical %15, %55 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %57 = pphlo.and %56, %9 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %58 = pphlo.bitcast_convert %cst_3 : (tensor<2x2xi64>) -> tensor<2x2xui64>
    //CHECK: %59 = pphlo.subtract %9, %57 : (tensor<2x2xui64>, tensor<2x2x!pphlo.secret<ui64>>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %60 = pphlo.multiply %57, %58 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %61 = pphlo.add %60, %59 : tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %62 = pphlo.multiply %54, %61 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.secret<ui64>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %63 = pphlo.bitcast_convert %cst_2 : (tensor<2x2xi64>) -> tensor<2x2xui64>
    //CHECK: %64 = pphlo.shift_right_logical %15, %63 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %65 = pphlo.and %64, %9 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %66 = pphlo.bitcast_convert %cst_1 : (tensor<2x2xi64>) -> tensor<2x2xui64>
    //CHECK: %67 = pphlo.subtract %9, %65 : (tensor<2x2xui64>, tensor<2x2x!pphlo.secret<ui64>>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %68 = pphlo.multiply %65, %66 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %69 = pphlo.add %68, %67 : tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %70 = pphlo.multiply %62, %69 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.secret<ui64>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %71 = pphlo.bitcast_convert %cst_4 : (tensor<2x2xi64>) -> tensor<2x2xui64>
    //CHECK: %72 = pphlo.shift_right_logical %15, %71 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %73 = pphlo.and %72, %9 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %74 = pphlo.bitcast_convert %cst_0 : (tensor<2x2xi64>) -> tensor<2x2xui64>
    //CHECK: %75 = pphlo.subtract %9, %73 : (tensor<2x2xui64>, tensor<2x2x!pphlo.secret<ui64>>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %76 = pphlo.multiply %73, %74 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %77 = pphlo.add %76, %75 : tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %78 = pphlo.multiply %70, %77 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.secret<ui64>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %79 = pphlo.bitcast_convert %78 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %80 = pphlo.bitcast_convert %cst : (tensor<2x2xi64>) -> tensor<2x2xui64>
    //CHECK: %81 = pphlo.shift_right_logical %79, %80 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2xui64>) -> tensor<2x2x!pphlo.secret<ui64>>
    //CHECK: %82 = pphlo.bitcast_convert %81 : (tensor<2x2x!pphlo.secret<ui64>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %83 = pphlo.subtract %82, %78 : tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %84 = pphlo.multiply %12, %83 : (tensor<2x2x!pphlo.secret<ui64>>, tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %85 = pphlo.add %78, %84 : tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: return %85 : tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    %0 = pphlo.exponential %arg0 : tensor<2x2x!pphlo.secret<f32>>
    return %0 : tensor<2x2x!pphlo.secret<f32>>
}
