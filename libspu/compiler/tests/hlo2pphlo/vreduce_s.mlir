// RUN: mlir-pphlo-opt --hlo-legalize-to-pphlo=input_vis_list=VIS_SECRET,VIS_SECRET --lower-conversion-cast %s --split-input-file  | FileCheck %s

func.func @main(%arg0: tensor<1024x1xf32>, %arg1: tensor<1024x1xf32>) -> (tensor<1024xf32>, tensor<1024xf32>) {
    %0 = "stablehlo.constant"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<f32>
    // CHECK: %1 = pphlo.convert %0 : (tensor<f32>) -> tensor<!pphlo.secret<f32>>
    // CHECK: %2 = pphlo.convert %0 : (tensor<f32>) -> tensor<!pphlo.secret<f32>>
    // CHECK: %3:2 = pphlo.reduce(%arg0 init: %1), (%arg1 init: %2) across dimensions = [1] : (tensor<1024x1x!pphlo.secret<f32>>, tensor<1024x1x!pphlo.secret<f32>>, tensor<!pphlo.secret<f32>>, tensor<!pphlo.secret<f32>>) -> (tensor<1024x!pphlo.secret<f32>>, tensor<1024x!pphlo.secret<f32>>)
    // CHECK:    reducer(%arg2: tensor<!pphlo.secret<f32>>, %arg4: tensor<!pphlo.secret<f32>>) (%arg3: tensor<!pphlo.secret<f32>>, %arg5: tensor<!pphlo.secret<f32>>) {
    // CHECK:        %4 = pphlo.add %arg2, %arg4 : tensor<!pphlo.secret<f32>>
    // CHECK:        %5 = pphlo.add %arg3, %arg5 : tensor<!pphlo.secret<f32>>
    // CHECK:        pphlo.return %4, %5 : tensor<!pphlo.secret<f32>>, tensor<!pphlo.secret<f32>>
    // CHECK: }
    %1:2 = "stablehlo.reduce"(%arg0, %arg1, %0, %0) ( {
        ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>, %arg4: tensor<f32>, %arg5: tensor<f32>):  // no predecessors
        %2 = "stablehlo.add"(%arg2, %arg4) : (tensor<f32>, tensor<f32>) -> tensor<f32>
        %3 = "stablehlo.add"(%arg3, %arg5) : (tensor<f32>, tensor<f32>) -> tensor<f32>
        "stablehlo.return"(%2, %3) : (tensor<f32>, tensor<f32>) -> ()
    }) {dimensions = array<i64: 1>} : (tensor<1024x1xf32>, tensor<1024x1xf32>, tensor<f32>, tensor<f32>) -> (tensor<1024xf32>, tensor<1024xf32>)
    return %1#0, %1#1 :  tensor<1024xf32>, tensor<1024xf32>
}
