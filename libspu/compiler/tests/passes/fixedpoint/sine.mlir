// RUN: spu-opt --lower-sfloat-to-fxp --expand-fixedpoint-approx --lower-pphlo-float-inputs --split-input-file %s | FileCheck %s

func.func @main(%arg0: tensor<2x2x!pphlo.secret<f32>>) -> (tensor<2x2x!pphlo.secret<f32>>) {
    //CHECK: %cst = arith.constant dense<1.000000e+00> : tensor<1x4xf32>
    //CHECK: %0 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst) {allow_float = true} : (tensor<1x4xf32>) -> tensor<1x4x!pphlo.fxp<64, 18>>
    //CHECK: %cst_0 = arith.constant dense<2.000000e+00> : tensor<1x4xf32>
    //CHECK: %1 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_0) {allow_float = true} : (tensor<1x4xf32>) -> tensor<1x4x!pphlo.fxp<64, 18>>
    //CHECK: %cst_1 = arith.constant dense<4> : tensor<1x4xi64>
    //CHECK: %cst_2 = arith.constant dense<0.254647911> : tensor<1x4xf32>
    //CHECK: %2 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_2) {allow_float = true} : (tensor<1x4xf32>) -> tensor<1x4x!pphlo.fxp<64, 18>>
    //CHECK{LITERAL}: %cst_3 = arith.constant dense<[[-0.0757078752, -0.853236377, 0.247478902, -0.0271984488, 0.00167500577]]> : tensor<1x5xf32>
    //CHECK: %3 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_3) {allow_float = true} : (tensor<1x5xf32>) -> tensor<1x5x!pphlo.fxp<64, 18>>
    //CHECK: %cst_4 = arith.constant dense<3.14159274> : tensor<2x2xf32>
    //CHECK: %4 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_4) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %cst_5 = arith.constant dense<6.28318548> : tensor<2x2xf32>
    //CHECK: %5 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_5) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %cst_6 = arith.constant dense<0.159154937> : tensor<2x2xf32>
    //CHECK: %6 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%cst_6) {allow_float = true} : (tensor<2x2xf32>) -> tensor<2x2x!pphlo.fxp<64, 18>>
    //CHECK: %7 = pphlo.add %arg0, %4 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %8 = pphlo.multiply %7, %6 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %9 = pphlo.truncate %8 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %10 = pphlo.floor %9 : tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %11 = pphlo.multiply %10, %5 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.fxp<64, 18>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %12 = pphlo.truncate %11 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %13 = pphlo.subtract %arg0, %12 : tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %14 = pphlo.reshape %13 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %15 = pphlo.multiply %14, %2 : (tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<1x4x!pphlo.fxp<64, 18>>) -> tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %16 = pphlo.truncate %15 : (tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %17 = pphlo.multiply %16, %16 : (tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %18 = pphlo.truncate %17 : (tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %19 = pphlo.multiply %cst_1, %18 : (tensor<1x4xi64>, tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %20 = pphlo.subtract %19, %1 : (tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<1x4x!pphlo.fxp<64, 18>>) -> tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %21 = pphlo.subtract %20, %0 : (tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<1x4x!pphlo.fxp<64, 18>>) -> tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %22 = pphlo.multiply %16, %21 : (tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %23 = pphlo.truncate %22 : (tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %24 = pphlo.multiply %20, %23 : (tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %25 = pphlo.truncate %24 : (tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %26 = pphlo.subtract %25, %16 : tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %27 = pphlo.multiply %20, %26 : (tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %28 = pphlo.truncate %27 : (tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %29 = pphlo.subtract %28, %23 : tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %30 = pphlo.multiply %20, %29 : (tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %31 = pphlo.truncate %30 : (tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %32 = pphlo.subtract %31, %26 : tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %33 = pphlo.concatenate %16, %23, %26, %29, %32 dim = 0 : (tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<5x4x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %34 = pphlo.dot %3, %33 : (tensor<1x5x!pphlo.fxp<64, 18>>, tensor<5x4x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 36>>>
    //CHECK: %35 = pphlo.truncate %34 : (tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: %36 = pphlo.reshape %35 : (tensor<1x4x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    //CHECK: return %36 : tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    %0 = pphlo.sine %arg0 : tensor<2x2x!pphlo.secret<f32>>
    return %0 : tensor<2x2x!pphlo.secret<f32>>
}
