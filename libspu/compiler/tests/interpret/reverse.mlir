// RUN: spu-translate --protocol_kind=1 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=2 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=3 --interpret -split-input-file %s

func.func @reverse() {
  %operand = pphlo.constant dense<[[1, 2], [3, 4], [5, 6]]> : tensor<3x2xi64>
  %result = "pphlo.reverse"(%operand) {
    dimensions = array<i64: 1, 0>
  } : (tensor<3x2xi64>) -> tensor<3x2xi64>
  %expected = pphlo.constant dense<[[6, 5], [4, 3], [2, 1]]> : tensor<3x2xi64>
  pphlo.custom_call @expect_eq (%result, %expected) : (tensor<3x2xi64>,tensor<3x2xi64>)->()
  func.return
}
