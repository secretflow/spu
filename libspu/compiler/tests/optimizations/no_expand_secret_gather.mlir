// RUN: pphlo-opt --expand-secret-gather --split-input-file %s | FileCheck %s
func.func @main(%arg0: tensor<2x!pphlo.secret<i32>>, %arg1: tensor<1xi32>) -> (tensor<!pphlo.secret<i32>>) {
    //CHECK-NOT: pphlo.while
    //CHECK : pphlo.gather
    %0 = pphlo.custom_call @pphlo.gather(%arg0, %arg1) {pphlo.attributes = {offset_dims = array<i64: 0>, collapsed_slice_dims = array<i64: 0>, start_index_map = array<i64: 0>, index_vector_dim = 0 : i64, slice_sizes = array<i64: 1>}} : (tensor<2x!pphlo.secret<i32>>, tensor<1xi32>) -> tensor<!pphlo.secret<i32>>
    return %0: tensor<!pphlo.secret<i32>>
}

// -----
func.func @main(%arg0: tensor<3x3xi32>, %arg1: tensor<2xi32>) -> (tensor<2x3xi32>) {
    //CHECK-NOT: pphlo.while
    //CHECK : pphlo.gather
   %0 = pphlo.custom_call @pphlo.gather(%arg0, %arg1) {pphlo.attributes = {offset_dims = array<i64: 1>, collapsed_slice_dims = array<i64: 0>, start_index_map = array<i64: 0>, index_vector_dim = 1 : i64, slice_sizes = array<i64: 1, 3>}} : (tensor<3x3xi32>, tensor<2xi32>) -> tensor<2x3xi32>
    return %0 : tensor<2x3xi32>
}