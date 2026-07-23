// RUN: spu-opt --hlo-legalize-to-pphlo=input_vis_list=VIS_SECRET,VIS_PUBLIC,VIS_PUBLIC --lower-conversion-cast --split-input-file %s | FileCheck %s

func.func @main(%arg0: tensor<1024x1xi1>, %arg1: tensor<1024x1xf32>, %arg2: tensor<1024x1xf32>) -> (tensor<1024x1xf32>) {
    // CHECK:pphlo.select %arg0, %arg1, %arg2 : (tensor<1024x1x!pphlo.secret<i1>>, tensor<1024x1xf32>, tensor<1024x1xf32>) -> tensor<1024x1x!pphlo.secret<f32>>
    %0 = "stablehlo.select"(%arg0, %arg1, %arg2) : (tensor<1024x1xi1>, tensor<1024x1xf32>, tensor<1024x1xf32>) -> tensor<1024x1xf32>
    return %0 : tensor<1024x1xf32>
}

// -----

func.func @main(%arg0: tensor<1024x1xf32>, %arg1: tensor<1024x1xf32>, %arg2: tensor<1024x1xf32>) -> (tensor<1024x1xf32>) {
    // CHECK: %0 = pphlo.clamp %arg0, %arg1, %arg2 : (tensor<1024x1x!pphlo.secret<f32>>, tensor<1024x1xf32>, tensor<1024x1xf32>) -> tensor<1024x1x!pphlo.secret<f32>>
    %0 = "stablehlo.clamp"(%arg0, %arg1, %arg2) : (tensor<1024x1xf32>, tensor<1024x1xf32>, tensor<1024x1xf32>) -> tensor<1024x1xf32>
    return %0 : tensor<1024x1xf32>
}
