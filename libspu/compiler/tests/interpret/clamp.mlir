// RUN: spu-translate --protocol_kind=1 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=2 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=3 --interpret -split-input-file %s

func.func @clamp_op_test_si64() {
  %min = pphlo.constant dense<[1, 5, -5]> : tensor<3xi64>
  %operand = pphlo.constant dense<[2, 3, -1]> : tensor<3xi64>
  %max = pphlo.constant dense<[3, 7, -3]> : tensor<3xi64>
  %result = pphlo.clamp %min, %operand, %max : (tensor<3xi64>, tensor<3xi64>, tensor<3xi64>) -> tensor<3xi64>
  %expected = pphlo.constant dense<[2, 5, -3]> : tensor<3xi64>
  pphlo.custom_call @expect_eq(%result, %expected) : (tensor<3xi64>,tensor<3xi64>)->()
  func.return
}
