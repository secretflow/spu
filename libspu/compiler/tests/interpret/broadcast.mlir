// RUN: spu-translate --interpret -split-input-file %s

func.func @broadcast_in_dim() {
  %operand = pphlo.constant dense<[[1], [2], [3]]> : tensor<3x1xi64>
  %result = pphlo.broadcast %operand, dims = [0, 2] : (tensor<3x1xi64>) -> tensor<3x2x2xi64>
  %expected = pphlo.constant dense<[[[1, 1], [1, 1]], [[2, 2], [2, 2]], [[3, 3], [3, 3]]]> : tensor<3x2x2xi64>
  pphlo.custom_call @expect_eq(%result, %expected) : (tensor<3x2x2xi64>,tensor<3x2x2xi64>)->()
  func.return
}
