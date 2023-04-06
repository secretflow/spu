// RUN: mlir-pphlo-opt --expand-secret-gather --split-input-file %s | FileCheck %s

func.func @main(%arg0: tensor<2x!pphlo.pub<i32>>, %arg1: tensor<1x!pphlo.pub<i32>>) -> (tensor<!pphlo.pub<i32>>) {
    //CHECK-NOT: pphlo.while
    //CHECK : pphlo.gather
    %0 = "pphlo.gather"(%arg0, %arg1) {dimension_numbers = #pphlo.gather<collapsed_slice_dims = [0], start_index_map = [0]>, indices_are_sorted = true, slice_sizes = dense<1> : tensor<1xi64>} : (tensor<2x!pphlo.pub<i32>>, tensor<1x!pphlo.pub<i32>>) -> tensor<!pphlo.pub<i32>>
    return %0: tensor<!pphlo.pub<i32>>
}
