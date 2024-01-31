// RUN: mlir-pphlo-opt --hlo-legalize-to-pphlo=input_vis_list=VIS_SECRET,VIS_PUBLIC --lower-conversion-cast %s --split-input-file  | FileCheck %s

func.func @main(%arg0: tensor<1024x1xf32>, %arg1: tensor<1024x1xf32>) -> (tensor<1024xf32>, tensor<1024xf32>) {
    %0 = "stablehlo.constant"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<f32>
    // CHECK: %2:2 = "pphlo.reduce"(%arg0, %arg1, %1, %0) ({
    // CHECK:    ^bb0(%arg2: tensor<!pphlo.secret<f32>>, %arg3: tensor<f32>, %arg4: tensor<!pphlo.secret<f32>>, %arg5: tensor<f32>):
    // CHECK:        %3 = "pphlo.add"(%arg2, %arg4) : (tensor<!pphlo.secret<f32>>, tensor<!pphlo.secret<f32>>) -> tensor<!pphlo.secret<f32>>
    // CHECK:        %4 = "pphlo.add"(%arg3, %arg5) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    // CHECK:        "pphlo.return"(%3, %4) : (tensor<!pphlo.secret<f32>>, tensor<f32>) -> ()
    // CHECK: }) {dimensions = array<i64: 1>} : (tensor<1024x1x!pphlo.secret<f32>>, tensor<1024x1xf32>, tensor<!pphlo.secret<f32>>, tensor<f32>) -> (tensor<1024x!pphlo.secret<f32>>, tensor<1024xf32>)
    %1:2 = "stablehlo.reduce"(%arg0, %arg1, %0, %0) ( {
        ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>, %arg4: tensor<f32>, %arg5: tensor<f32>):  // no predecessors
        %2 = "stablehlo.add"(%arg2, %arg4) : (tensor<f32>, tensor<f32>) -> tensor<f32>
        %3 = "stablehlo.add"(%arg3, %arg5) : (tensor<f32>, tensor<f32>) -> tensor<f32>
        "stablehlo.return"(%2, %3) : (tensor<f32>, tensor<f32>) -> ()
    }) {dimensions = dense<[1]> : tensor<1xi64>} : (tensor<1024x1xf32>, tensor<1024x1xf32>, tensor<f32>, tensor<f32>) -> (tensor<1024xf32>, tensor<1024xf32>)
    return %1#0, %1#1 :  tensor<1024xf32>, tensor<1024xf32>
}
