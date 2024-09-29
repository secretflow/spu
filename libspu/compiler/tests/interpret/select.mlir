// RUN: spu-translate --protocol_kind=1 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=2 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=3 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=4 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=5 --interpret -split-input-file %s

func.func @select_op_test_si64() {
  %pred = arith.constant dense<[true, false, true]> : tensor<3xi1>
  %on_true = arith.constant dense<[2, 3, -1]> : tensor<3xi64>
  %on_false = arith.constant dense<[3, 7, -3]> : tensor<3xi64>
  %result = pphlo.select %pred, %on_true, %on_false : (tensor<3xi1>, tensor<3xi64>, tensor<3xi64>) -> tensor<3xi64>
  %expected = arith.constant dense<[2, 7, -1]> : tensor<3xi64>
  pphlo.custom_call @expect_eq (%result, %expected) : (tensor<3xi64>,tensor<3xi64>)->()
  func.return
}

// -----
// FIXME
func.func @select_op_test_si64_scalar() {
  %pred = arith.constant dense<false> : tensor<i1>
  %on_true = arith.constant dense<[2, 3, -1]> : tensor<3xi64>
  %on_false = arith.constant dense<[3, 7, -3]> : tensor<3xi64>
//   %result = pphlo.select %pred, %on_true, %on_false : (tensor<i1>, tensor<3xi64>, tensor<3xi64>) -> tensor<3xi64>
//   %expected = arith.constant dense<[3, 7, -3]> : tensor<3xi64>
//   pphlo.custom_call @expect_eq %result, %expected : tensor<3xi64>
  func.return
}
