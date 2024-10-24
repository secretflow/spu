// RUN: mlir-pphlo-opt --expand-secret-gather --split-input-file %s | FileCheck %s

func.func @main(%arg0: tensor<2x!pphlo.pub<i32>>, %arg1: tensor<1x!pphlo.sec<i32>>) -> (tensor<!pphlo.sec<i32>>) {
    //CHECK-NOT: pphlo.gather
    //CHECK : pphlo.while
    %0 = "pphlo.gather"(%arg0, %arg1) {dimension_numbers = #pphlo.gather<collapsed_slice_dims = [0], start_index_map = [0]>, indices_are_sorted = true, slice_sizes = dense<1> : tensor<1xi64>} : (tensor<2x!pphlo.pub<i32>>, tensor<1x!pphlo.sec<i32>>) -> tensor<!pphlo.sec<i32>>
    return %0: tensor<!pphlo.sec<i32>>
}

// -----
func.func @main(%arg0: tensor<3x3x!pphlo.pub<i32>>, %arg1: tensor<2x!pphlo.sec<i32>>) -> (tensor<2x3x!pphlo.sec<i32>>) {
    %0 = "pphlo.gather"(%arg0, %arg1) {dimension_numbers = #pphlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 3]> : tensor<2xi64>} : (tensor<3x3x!pphlo.pub<i32>>, tensor<2x!pphlo.sec<i32>>) -> tensor<2x3x!pphlo.sec<i32>>
    return %0 : tensor<2x3x!pphlo.sec<i32>>
}
