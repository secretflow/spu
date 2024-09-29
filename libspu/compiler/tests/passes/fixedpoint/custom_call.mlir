// RUN: spu-opt --lower-sfloat-to-fxp --expand-fixedpoint-approx --lower-pphlo-float-inputs --split-input-file %s | FileCheck %s

func.func @main(%arg0: tensor<2xf32>) -> tensor<2xf32> {
    //CHECK: %0 = pphlo.custom_call @spu.compiler.internal.encode_to_fxp(%arg0) {allow_float = true} : (tensor<2xf32>) -> tensor<2x!pphlo.fxp<64, 18>>
    //CHECK: %1 = pphlo.custom_call @magic_fun(%0) : (tensor<2x!pphlo.fxp<64, 18>>) -> tensor<2x!pphlo.fxp<64, 18>>
    //CHECK: %2 = pphlo.custom_call @spu.compiler.internal.decode_from_fxp(%1) {allow_float = true} : (tensor<2x!pphlo.fxp<64, 18>>) -> tensor<2xf32>
    //CHECK: return %2 : tensor<2xf32>
    %0 = pphlo.custom_call @magic_fun(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
    return %0 : tensor<2xf32>
}

// -----

func.func @main(%arg0: tensor<2x!pphlo.secret<f32>>) -> tensor<2x!pphlo.secret<f32>> {
    //CHECK: %0 = pphlo.custom_call @magic_fun(%arg0) : (tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    %0 = pphlo.custom_call @magic_fun(%arg0) : (tensor<2x!pphlo.secret<f32>>) -> tensor<2x!pphlo.secret<f32>>
    return %0 : tensor<2x!pphlo.secret<f32>>
}

// -----

func.func @main(%arg0: tensor<2xf32>) -> tensor<2xf32> {
    //CHECK: %0 = pphlo.custom_call @magic_fun(%arg0) {allow_float = true} : (tensor<2xf32>) -> tensor<2xf32>
    //CHECK: return %0 : tensor<2xf32>
    %0 = pphlo.custom_call @magic_fun(%arg0) {allow_float = true}: (tensor<2xf32>) -> tensor<2xf32>
    return %0 : tensor<2xf32>
}
