// RUN: spu-translate --protocol_kind=1 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=2 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=3 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=4 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=5 --interpret -split-input-file %s
// AUTO GENERATED, DO NOT EDIT

func.func @shift_left_op_test_i64_i64_pp() {
   %0 = arith.constant dense<[-1, 0, 1]> : tensor<3xi64>
   %1 = arith.constant dense<[1, 2, 3]> : tensor<3xi64>
   %2 = pphlo.shift_left %0,%1 : (tensor<3xi64>,tensor<3xi64>)->tensor<3xi64>
   %3 = arith.constant dense<[-2, 0, 8]> : tensor<3xi64>
   pphlo.custom_call @expect_eq(%2, %3) : (tensor<3xi64>, tensor<3xi64>)->()
   func.return
}

// -----

func.func @shift_left_op_test_i64_i64_ss() {
   %0 = arith.constant dense<[-1, 0, 1]> : tensor<3xi64>
   %1 = arith.constant dense<[1, 2, 3]> : tensor<3xi64>
   %2 = pphlo.convert %0 : (tensor<3xi64>)->tensor<3x!pphlo.secret<i64>>
   %3 = pphlo.convert %1 : (tensor<3xi64>)->tensor<3x!pphlo.secret<i64>>
   %4 = pphlo.shift_left %2, %3 : (tensor<3x!pphlo.secret<i64>>,tensor<3x!pphlo.secret<i64>>)->tensor<3x!pphlo.secret<i64>>
   %5 = arith.constant dense<[-2, 0, 8]> : tensor<3xi64>
   pphlo.custom_call @expect_eq(%5, %4) : (tensor<3xi64>, tensor<3x!pphlo.secret<i64>>)->()
   func.return
}
