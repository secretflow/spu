// RUN: mlir-pphlo-opt --hlo-legalize-to-pphlo=input_vis_list=VIS_PUBLIC,VIS_PUBLIC,VIS_PUBLIC --lower-conversion-cast %s | FileCheck %s

func.func @main(%arg0: tensor<1024x1xi1>, %arg1: tensor<1024x1xf32>, %arg2: tensor<1024x1xf32>) -> (tensor<1024x1xf32>) {
    // CHECK: %0 = "pphlo.select"(%arg0, %arg1, %arg2) : (tensor<1024x1xi1>, tensor<1024x1xf32>, tensor<1024x1xf32>) -> tensor<1024x1xf32>
    %0 = "stablehlo.select"(%arg0, %arg1, %arg2) : (tensor<1024x1xi1>, tensor<1024x1xf32>, tensor<1024x1xf32>) -> tensor<1024x1xf32>
    return %0 : tensor<1024x1xf32>
}
