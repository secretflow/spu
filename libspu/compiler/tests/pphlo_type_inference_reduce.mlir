// RUN: mlir-pphlo-opt --hlo-legalize-to-pphlo=input_vis_list=VIS_SECRET %s --split-input-file  | FileCheck %s

// CHECK: func @main(%arg0: tensor<1024x1x!pphlo.sec<f32>>) -> tensor<1024x!pphlo.sec<f32>> {
func.func @main(%arg1: tensor<1024x1xf32>) -> (tensor<1024xf32>) {
    // CHECK: %0 = "pphlo.constant"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<!pphlo.pub<f32>>
    %0 = "stablehlo.constant"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<f32>
    // CHECK: %2 = "pphlo.reduce"(%arg0, %1) ({
    // CHECK:       ^bb0(%arg1: tensor<!pphlo.sec<f32>>, %arg2: tensor<!pphlo.sec<f32>>):
    // CHECK:        %3 = "pphlo.add"(%arg1, %arg2) : (tensor<!pphlo.sec<f32>>, tensor<!pphlo.sec<f32>>) -> tensor<!pphlo.sec<f32>>
    // CHECK:        "pphlo.return"(%3) : (tensor<!pphlo.sec<f32>>) -> ()
    // CHECK: }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<1024x1x!pphlo.sec<f32>>, tensor<!pphlo.sec<f32>>) -> tensor<1024x!pphlo.sec<f32>>
    %1 = "stablehlo.reduce"(%arg1, %0) ( {
        ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):  // no predecessors
        %2 = "stablehlo.add"(%arg2, %arg3) : (tensor<f32>, tensor<f32>) -> tensor<f32>
        "stablehlo.return"(%2) : (tensor<f32>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<1024x1xf32>, tensor<f32>) -> tensor<1024xf32>
    return %1 :  tensor<1024xf32>
}
