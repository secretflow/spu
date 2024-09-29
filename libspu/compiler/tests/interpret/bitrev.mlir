// RUN: spu-translate --protocol_kind=1 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=2 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=3 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=4 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=5 --interpret -split-input-file %s
// AUTO GENERATED, DO NOT EDIT

func.func @bitrev_op_test_i64_i64_p() {
   %0 = arith.constant dense<[262143, 1, 34359738368, 536870911]> : tensor<4xi64>
   %1 = pphlo.bitrev %0 { start = 0, end = 36 }: (tensor<4xi64>)->tensor<4xi64>
   %2 = arith.constant dense<[68719214592, 34359738368, 1, 68719476608]> : tensor<4xi64>
   pphlo.custom_call @expect_eq(%1, %2) : (tensor<4xi64>, tensor<4xi64>)->()
   func.return
}

// -----

func.func @bitrev_op_test_i64_i64_s() {
   %0 = arith.constant dense<[262143, 1, 34359738368, 536870911]> : tensor<4xi64>
   %1 = pphlo.convert %0 : (tensor<4xi64>)->tensor<4x!pphlo.secret<i64>>
   %2 = pphlo.bitrev %1 { start = 0, end = 36 }: (tensor<4x!pphlo.secret<i64>>)->tensor<4x!pphlo.secret<i64>>
   %3 = arith.constant dense<[68719214592, 34359738368, 1, 68719476608]> : tensor<4xi64>
   pphlo.custom_call @expect_eq(%3, %2) : (tensor<4xi64>, tensor<4x!pphlo.secret<i64>>)->()
   func.return
}
