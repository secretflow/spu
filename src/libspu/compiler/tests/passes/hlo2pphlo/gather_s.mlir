// RUN: spu-opt --hlo-legalize-to-pphlo=input_vis_list=VIS_SECRET,VIS_SECRET --lower-conversion-cast --split-input-file %s | FileCheck %s

func.func @main(%arg0 : tensor<3x4x2xi64>, %arg1 : tensor<2x3x2xi64>) -> tensor<2x3x2x2xi64> {
  // CHECK: pphlo.custom_call @spu.gather(%arg0, %arg1) {pphlo.attributes = {collapsed_slice_dims = array<i64: 0>, index_vector_dim = 2 : i64, offset_dims = array<i64: 2, 3>, slice_sizes = array<i64: 1, 2, 2>, start_index_map = array<i64: 1, 0>}} : (tensor<3x4x2x!pphlo.secret<i64>>, tensor<2x3x2x!pphlo.secret<i64>>) -> tensor<2x3x2x2x!pphlo.secret<i64>>
  %result = "stablehlo.gather"(%arg0, %arg1) {
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
