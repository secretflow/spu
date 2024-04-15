// RUN: pphlo-opt --hlo-legalize-to-pphlo=input_vis_list=VIS_PUBLIC --lower-conversion-cast --split-input-file %s | FileCheck %s

func.func @main() -> tensor<2x3x2x2xi64> {
  // CHECK: pphlo.custom_call @pphlo.gather(%0, %1) {pphlo.attributes = {collapsed_slice_dims = array<i64: 0>, index_vector_dim = 2 : i64, offset_dims = array<i64: 2, 3>, slice_sizes = array<i64: 1, 2, 2>, start_index_map = array<i64: 1, 0>}} : (tensor<3x4x2xi64>, tensor<2x3x2xi64>) -> tensor<2x3x2x2xi64>
  %operand = stablehlo.constant dense<[[[1, 2], [3, 4], [5, 6], [7, 8]],
                                       [[9, 10], [11, 12], [13, 14], [15, 16]],
                                       [[17, 18], [19, 20], [21, 22], [23, 24]]]> : tensor<3x4x2xi64>
  %start_indices = stablehlo.constant dense<[[[0, 0], [1, 0], [2, 1]],
                                             [[0, 1], [1, 1], [0, 9]]]> : tensor<2x3x2xi64>
  %result = "stablehlo.gather"(%operand, %start_indices) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [2, 3],
      collapsed_slice_dims = [0],
      start_index_map = [1, 0],
      index_vector_dim = 2>,
    slice_sizes = array<i64: 1, 2, 2>,
    indices_are_sorted = false
  } : (tensor<3x4x2xi64>, tensor<2x3x2xi64>) -> tensor<2x3x2x2xi64>
  return %result : tensor<2x3x2x2xi64>
}
