// RUN: spu-translate --protocol_kind=1 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=2 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=3 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=4 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=5 --interpret -split-input-file %s

func.func @clamp_op_test_si64() {
  %min = arith.constant dense<[1, 5, -5]> : tensor<3xi64>
  %operand = arith.constant dense<[2, 3, -1]> : tensor<3xi64>
  %max = arith.constant dense<[3, 7, -3]> : tensor<3xi64>
  %result = pphlo.clamp %min, %operand, %max : (tensor<3xi64>, tensor<3xi64>, tensor<3xi64>) -> tensor<3xi64>
  %expected = arith.constant dense<[2, 5, -3]> : tensor<3xi64>
  pphlo.custom_call @expect_eq(%result, %expected) : (tensor<3xi64>, tensor<3xi64>)->()
  func.return
}

// -----

func.func @clamp_op_test_si64_s() {
  %0 = arith.constant dense<[1, 5, -5]> : tensor<3xi64>
  %1 = arith.constant dense<[2, 3, -1]> : tensor<3xi64>
  %2 = arith.constant dense<[3, 7, -3]> : tensor<3xi64>
  %min = pphlo.convert %0 : (tensor<3xi64>) -> tensor<3x!pphlo.secret<i64>>
  %operand = pphlo.convert %1 : (tensor<3xi64>) -> tensor<3x!pphlo.secret<i64>>
  %max = pphlo.convert %2 : (tensor<3xi64>) -> tensor<3x!pphlo.secret<i64>>
  %3 = pphlo.clamp %min, %operand, %max : (tensor<3x!pphlo.secret<i64>>, tensor<3x!pphlo.secret<i64>>, tensor<3x!pphlo.secret<i64>>) -> tensor<3x!pphlo.secret<i64>>
  %expected = arith.constant dense<[2, 5, -3]> : tensor<3xi64>
  pphlo.custom_call @expect_eq(%expected, %3) : (tensor<3xi64>, tensor<3x!pphlo.secret<i64>>)->()
  func.return
}
