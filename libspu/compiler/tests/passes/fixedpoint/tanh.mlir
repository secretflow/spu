// RUN: spu-opt --lower-sfloat-to-fxp --expand-fixedpoint-approx --lower-pphlo-float-inputs --split-input-file %s | FileCheck %s

func.func @main(%arg0: tensor<2x2x!pphlo.secret<f32>>) -> (tensor<2x2x!pphlo.secret<f32>>) {
    //CHECK: %cst = arith.constant dense<1.000000e+00> : tensor<1x4xf32>
    //CHECK: %0 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst) {allow_float = true} : (tensor<1x4xf32>) -> tensor<1x4x!pphlo.fxp<64, 18>>
    //CHECK: %cst_0 = arith.constant dense<2.000000e+00> : tensor<1x4xf32>
    //CHECK: %1 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_0) {allow_float = true} : (tensor<1x4xf32>) -> tensor<1x4x!pphlo.fxp<64, 18>>
    //CHECK: %cst_1 = arith.constant dense<4> : tensor<1x4xi64>
    //CHECK: %cst_2 = arith.constant dense<2.000000e-01> : tensor<1x4xf32>
    //CHECK: %2 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_2) {allow_float = true} : (tensor<1x4xf32>) -> tensor<1x4x!pphlo.fxp<64, 18>>
    //CHECK: %cst_3 = arith.constant dense<5.000000e+00> : tensor<1x4xf32>
    //CHECK: %3 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_3) {allow_float = true} : (tensor<1x4xf32>) -> tensor<1x4x!pphlo.fxp<64, 18>>
    //CHECK: %cst_4 = arith.constant dense<-5.000000e+00> : tensor<1x4xf32>
    //CHECK: %4 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_4) {allow_float = true} : (tensor<1x4xf32>) -> tensor<1x4x!pphlo.fxp<64, 18>>
    //CHECK{LITERAL}: %cst_5 = arith.constant dense<[[1.25140464, -0.365598768, 0.172531411, -0.0894344598, 0.0477030165, -0.0258302912, 0.0143388016, -0.00854173116, 0.00612306874]]> : tensor<1x9xf32>
    //CHECK: %5 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_5) {allow_float = true} : (tensor<1x9xf32>) -> tensor<1x9x!pphlo.fxp<64, 18>>
    //CHECK: %6 = pphlo.reshape %arg0 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %7 = pphlo.clamp %4, %6, %3 : (tensor<1x4x!pphlo.fxp<64, 18>>, tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<1x4x!pphlo.fxp<64, 18>>) -> tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %8 = pphlo.multiply %7, %2 : (tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<1x4x!pphlo.fxp<64, 18>>) -> tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %9 = pphlo.truncate %8 : (tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %10 = pphlo.multiply %9, %9 : (tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %11 = pphlo.truncate %10 : (tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %12 = pphlo.multiply %cst_1, %11 : (tensor<1x4xi64>, tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %13 = pphlo.subtract %12, %1 : (tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<1x4x!pphlo.fxp<64, 18>>) -> tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %14 = pphlo.subtract %13, %0 : (tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<1x4x!pphlo.fxp<64, 18>>) -> tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %15 = pphlo.multiply %9, %14 : (tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %16 = pphlo.truncate %15 : (tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %17 = pphlo.multiply %13, %16 : (tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %18 = pphlo.truncate %17 : (tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %19 = pphlo.subtract %18, %9 : tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %20 = pphlo.multiply %13, %19 : (tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %21 = pphlo.truncate %20 : (tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %22 = pphlo.subtract %21, %16 : tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %23 = pphlo.multiply %13, %22 : (tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %24 = pphlo.truncate %23 : (tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %25 = pphlo.subtract %24, %19 : tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %26 = pphlo.multiply %13, %25 : (tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %27 = pphlo.truncate %26 : (tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %28 = pphlo.subtract %27, %22 : tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %29 = pphlo.multiply %13, %28 : (tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %30 = pphlo.truncate %29 : (tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %31 = pphlo.subtract %30, %25 : tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %32 = pphlo.multiply %13, %31 : (tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %33 = pphlo.truncate %32 : (tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %34 = pphlo.subtract %33, %28 : tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %35 = pphlo.multiply %13, %34 : (tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %36 = pphlo.truncate %35 : (tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %37 = pphlo.subtract %36, %31 : tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %38 = pphlo.concatenate %9, %16, %19, %22, %25, %28, %31, %34, %37 dim = 0 : (tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<9x4x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %39 = pphlo.dot %5, %38 : (tensor<1x9x!pphlo.fxp<64, 18>>, tensor<9x4x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %40 = pphlo.truncate %39 : (tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %41 = pphlo.reshape %40 : (tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: return %41 : tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    %0 = pphlo.tanh %arg0 : tensor<2x2x!pphlo.secret<f32>>
    return %0 : tensor<2x2x!pphlo.secret<f32>>
}
