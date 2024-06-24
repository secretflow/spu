// RUN: spu-translate --interpret -split-input-file %s

func.func @dynamic_slice() {
  %operand = pphlo.constant dense<[[1, 1, 1],
                                   [1, 1, 1],
                                   [1, 1, 1]]> : tensor<3x3xi64>
  %start_indices0 = pphlo.constant dense<3> : tensor<i64>
  %start_indices1 = pphlo.constant dense<3> : tensor<i64>
  %result = "pphlo.dynamic_slice"(%operand, %start_indices0, %start_indices1) {
    slice_sizes = array<i64: 3, 3>
  } : (tensor<3x3xi64>, tensor<i64>, tensor<i64>) -> tensor<3x3xi64>
  pphlo.custom_call @expect_eq (%result, %operand) : (tensor<3x3xi64>,tensor<3x3xi64>)->()
  func.return
}
