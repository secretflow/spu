// RUN: spu-translate --protocol_kind=1 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=2 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=3 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=4 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=5 --interpret -split-input-file %s

func.func @cast_1() {
  %c0 = arith.constant dense<[[1, 2], [3, 4], [5, 6]]> : tensor<3x2xi32>
  %operand = pphlo.convert %c0 : (tensor<3x2xi32>)->tensor<3x2x!pphlo.secret<i32>>
  %result = pphlo.convert %operand : (tensor<3x2x!pphlo.secret<i32>>) -> tensor<3x2x!pphlo.secret<i64>>
  %expected = arith.constant dense<[[1, 2], [3, 4], [5, 6]]> : tensor<3x2xi64>
  pphlo.custom_call @expect_eq (%expected, %result) : (tensor<3x2xi64>,tensor<3x2x!pphlo.secret<i64>>)->()
  func.return
}
