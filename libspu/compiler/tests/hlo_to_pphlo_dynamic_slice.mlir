// RUN: mlir-pphlo-opt --hlo-legalize-to-pphlo=input_vis_list=VIS_PUBLIC,VIS_SECRET --split-input-file %s | FileCheck %s

func.func @main(%arg0: tensor<15xi32>,%arg1: tensor<i32>) -> (tensor<1xi32>) {
    // CHECK:  %0 = "pphlo.dynamic-slice"(%arg0, %arg1) {slice_sizes = array<i64: 1>} : (tensor<15x!pphlo.pub<i32>>, tensor<!pphlo.sec<i32>>) -> tensor<1x!pphlo.sec<i32>>
    %0 = "stablehlo.dynamic_slice"(%arg0, %arg1) {slice_sizes = array<i64: 1>} : (tensor<15xi32>, tensor<i32>) -> tensor<1xi32>
    return %0 : tensor<1xi32>
}
