// RUN: spu-opt --legalize-to-arith --lower-sfloat-to-fxp --lower-pphlo-float-inputs --split-input-file %s | FileCheck %s

func.func @main(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>,
                %arg2: tensor<2x2x!pphlo.secret<f32>>, %arg3: tensor<2x2x!pphlo.secret<f32>>) -> (tensor<2x2xf32>, tensor<2x2x!pphlo.secret<f32>>) {
    //CHECK: %0 = arith.addf %arg0, %arg1 : tensor<2x2xf32>
    //CHECK: %1 = pphlo.add %arg2, %arg3 : tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    %0 = pphlo.add %arg0, %arg1 : tensor<2x2xf32>
    %1 = pphlo.add %arg2, %arg3 : tensor<2x2x!pphlo.secret<f32>>
    return %0,%1 : tensor<2x2xf32>,tensor<2x2x!pphlo.secret<f32>>
}

// -----

func.func @main(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>,
                %arg2: tensor<2x2x!pphlo.secret<f32>>, %arg3: tensor<2x2x!pphlo.secret<f32>>) -> (tensor<2x2xf32>, tensor<2x2x!pphlo.secret<f32>>) {
    //CHECK: %0 = arith.mulf %arg0, %arg1 : tensor<2x2xf32>
    %0 = "pphlo.multiply"(%arg0, %arg1) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    // CHECK: %[[MUL2:.+]] = pphlo.multiply %arg2, %arg3 : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>, tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>
    // CHECK: pphlo.truncate %[[MUL2]] : (tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 36>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    %1 = "pphlo.multiply"(%arg2, %arg3) : (tensor<2x2x!pphlo.secret<f32>>, tensor<2x2x!pphlo.secret<f32>>) -> tensor<2x2x!pphlo.secret<f32>>
    return %0,%1 : tensor<2x2xf32>,tensor<2x2x!pphlo.secret<f32>>
}

// -----

func.func @main(%arg0: tensor<2x2x!pphlo.secret<i32>>, %arg1: tensor<2x2x!pphlo.secret<f32>>) -> (tensor<2x2x!pphlo.secret<f32>>) {
    // CHECK: %0 = pphlo.multiply %arg0, %arg1 : (tensor<2x2x!pphlo.secret<i32>>, tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>) -> tensor<2x2x!pphlo.secret<!pphlo.fxp<64, 18>>>
    %0 = "pphlo.multiply"(%arg0, %arg1) : (tensor<2x2x!pphlo.secret<i32>>, tensor<2x2x!pphlo.secret<f32>>) -> tensor<2x2x!pphlo.secret<f32>>
    return %0 : tensor<2x2x!pphlo.secret<f32>>
}
