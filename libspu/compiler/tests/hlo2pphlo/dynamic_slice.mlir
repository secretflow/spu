// RUN: mlir-pphlo-opt --hlo-legalize-to-pphlo=input_vis_list=VIS_PUBLIC,VIS_SECRET --lower-conversion-cast --split-input-file %s | FileCheck %s

func.func @main(%arg0: tensor<15xi32>,%arg1: tensor<i32>) -> (tensor<1xi32>) {
    // CHECK:  %0 = "pphlo.dynamic-slice"(%arg0, %arg1) {slice_sizes = array<i64: 1>} : (tensor<15xi32>, tensor<!pphlo.secret<i32>>) -> tensor<1x!pphlo.secret<i32>>
    %0 = "stablehlo.dynamic_slice"(%arg0, %arg1) {slice_sizes = array<i64: 1>} : (tensor<15xi32>, tensor<i32>) -> tensor<1xi32>
    return %0 : tensor<1xi32>
}
