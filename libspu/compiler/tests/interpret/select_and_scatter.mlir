// RUN: spu-translate --protocol_kind=1 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=2 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=3 --interpret -split-input-file %s

// FIXME
func.func @select_and_scatter_op_test() {
  %operand = pphlo.constant dense<[[1, 5],
                                       [2, 5],
                                       [3, 6],
                                       [4, 4]]> : tensor<4x2xi64>
  %source = pphlo.constant dense<[[5, 6],
                                      [7, 8]]> : tensor<2x2xi64>
  %init_value = pphlo.constant dense<0> : tensor<i64>
//   %result = "pphlo.select_and_scatter"(%operand, %source, %init_value) ({
//     ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
//       %0 = pphlo.greater_equal %arg0, %arg1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
//       pphlo.return %0 : tensor<i1>
//   }, {
//     ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
//       %0 = pphlo.add %arg0, %arg1 : tensor<i64>
//       pphlo.return %0 : tensor<i64>
//   }) {
//     window_dimensions = array<i64: 3, 1>,
//     window_strides = array<i64: 2, 1>,
//     padding = dense<[[0, 1], [0, 0]]> : tensor<2x2xi64>
//   } : (tensor<4x2xi64>, tensor<2x2xi64>, tensor<i64>) -> tensor<4x2xi64>
//   %expected = pphlo.constant dense<[[0, 0],
//                                     [0, 0],
//                                     [5, 14],
//                                     [7, 0]]> : tensor<4x2xi64>
//   pphlo.custom_call @expect_eq %result, %expected : tensor<4x2xi64>
  func.return
}
