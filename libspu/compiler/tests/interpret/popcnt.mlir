// RUN: spu-translate --interpret -split-input-file %s
// AUTO GENERATED, DO NOT EDIT

func.func @popcnt_op_test_i64_i64_p() {
   %0 = pphlo.constant dense<[0, 1, 2, 127]> : tensor<4xi64>
   %1 = pphlo.popcnt %0 : (tensor<4xi64>)->tensor<4xi64>
   %2 = pphlo.constant dense<[0, 1, 1, 7]> : tensor<4xi64>
   pphlo.custom_call @expect_eq(%1, %2) : (tensor<4xi64>, tensor<4xi64>)->()
   func.return
}

// -----

func.func @popcnt_op_test_i64_i64_s() {
   %0 = pphlo.constant dense<[0, 1, 2, 127]> : tensor<4xi64>
   %1 = pphlo.convert %0 : (tensor<4xi64>)->tensor<4x!pphlo.secret<i64>>
   %2 = pphlo.popcnt %1 : (tensor<4x!pphlo.secret<i64>>)->tensor<4x!pphlo.secret<i64>>
   %3 = pphlo.constant dense<[0, 1, 1, 7]> : tensor<4xi64>
   %4 = pphlo.convert %2 : (tensor<4x!pphlo.secret<i64>>)->tensor<4xi64>
   // pphlo.custom_call @expect_eq(%3, %4) : (tensor<4xi64>, tensor<4xi64>)->()
   func.return
}
