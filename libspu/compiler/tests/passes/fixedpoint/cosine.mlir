// RUN: spu-opt --lower-sfloat-to-fxp --expand-fixedpoint-approx --lower-pphlo-float-inputs --split-input-file %s | FileCheck %s

func.func @main(%arg0: tensor<2x2x!pphlo.secret<f32>>) -> (tensor<2x2x!pphlo.secret<f32>>) {
    //CHECK: %cst = arith.constant dense<1.000000e+00> : tensor<1x4xf32>
    //CHECK: %0 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst) {allow_float = true} : (tensor<1x4xf32>) -> tensor<1x4x!pphlo.fxp<64, 18>>
    //CHECK: %cst_0 = arith.constant dense<2.000000e+00> : tensor<1x4xf32>
    //CHECK: %1 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_0) {allow_float = true} : (tensor<1x4xf32>) -> tensor<1x4x!pphlo.fxp<64, 18>>
    //CHECK: %cst_1 = arith.constant dense<4> : tensor<1x4xi64>
    //CHECK: %cst_2 = arith.constant dense<0.254647911> : tensor<1x4xf32>
    //CHECK: %2 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_2) {allow_float = true} : (tensor<1x4xf32>) -> tensor<1x4x!pphlo.fxp<64, 18>>
    //CHECK: %cst_3 = arith.constant dense<0.159154937> : tensor<2x2xf32>
    //CHECK: %3 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_3) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %cst_4 = arith.constant dense<6.28318548> : tensor<2x2xf32>
    //CHECK: %4 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_4) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %cst_5 = arith.constant dense<3.14159274> : tensor<2x2xf32>
    //CHECK: %5 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_5) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK{LITERAL}: %cst_6 = arith.constant dense<[[-0.0757078752, -0.853236377, 0.247478902, -0.0271984488, 0.00167500577]]> : tensor<1x5xf32>
    //CHECK: %6 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_6) {allow_float = true} : (tensor<1x5xf32>) -> tensor<1x5x!pphlo.fxp<64, 18>>
    //CHECK: %cst_7 = arith.constant dense<1.57079637> : tensor<2x2xf32>
    //CHECK: %7 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_7) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %8 = pphlo.subtract %7, %arg0 : (tensor<2x2x!pphlo.fxp<64, 18>>, tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %9 = pphlo.add %8, %5 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %10 = pphlo.multiply %9, %3 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %11 = pphlo.truncate %10 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %12 = pphlo.floor %11 : tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %13 = pphlo.multiply %12, %4 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %14 = pphlo.truncate %13 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %15 = pphlo.subtract %8, %14 : tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %16 = pphlo.reshape %15 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %17 = pphlo.multiply %16, %2 : (tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<1x4x!pphlo.fxp<64, 18>>) -> tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %18 = pphlo.truncate %17 : (tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %19 = pphlo.multiply %18, %18 : (tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %20 = pphlo.truncate %19 : (tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %21 = pphlo.multiply %cst_1, %20 : (tensor<1x4xi64>, tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %22 = pphlo.subtract %21, %1 : (tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<1x4x!pphlo.fxp<64, 18>>) -> tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %23 = pphlo.subtract %22, %0 : (tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<1x4x!pphlo.fxp<64, 18>>) -> tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %24 = pphlo.multiply %18, %23 : (tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %25 = pphlo.truncate %24 : (tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %26 = pphlo.multiply %22, %25 : (tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %27 = pphlo.truncate %26 : (tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %28 = pphlo.subtract %27, %18 : tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %29 = pphlo.multiply %22, %28 : (tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %30 = pphlo.truncate %29 : (tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %31 = pphlo.subtract %30, %25 : tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %32 = pphlo.multiply %22, %31 : (tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %33 = pphlo.truncate %32 : (tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %34 = pphlo.subtract %33, %28 : tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %35 = pphlo.concatenate %18, %25, %28, %31, %34 dim = 0 : (tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<5x4x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %36 = pphlo.dot %6, %35 : (tensor<1x5x!pphlo.fxp<64, 18>>, tensor<5x4x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %37 = pphlo.truncate %36 : (tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %38 = pphlo.reshape %37 : (tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: return %38 : tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    %0 = pphlo.cosine %arg0 : tensor<2x2x!pphlo.secret<f32>>
    return %0 : tensor<2x2x!pphlo.secret<f32>>
}
