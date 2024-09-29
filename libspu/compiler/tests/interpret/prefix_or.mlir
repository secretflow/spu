// RUN: spu-translate --protocol_kind=1 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=2 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=3 --interpret -split-input-file %s

func.func @prefix_or_op_test_i64_i64_s() {
   %0 = arith.constant dense<[-1, 0, 8]> : tensor<3xi64>
   %1 = pphlo.convert %0 : (tensor<3xi64>)->tensor<3x!pphlo.secret<i64>>
   %2 = pphlo.prefix_or %1 : (tensor<3x!pphlo.secret<i64>>)->tensor<3x!pphlo.secret<i64>>
   %3 = arith.constant dense<[-1, 0, 15]> : tensor<3xi64>
   pphlo.custom_call @expect_eq(%3, %2) : (tensor<3xi64>, tensor<3x!pphlo.secret<i64>>)->()
   func.return
}
