// RUN: spu-translate --protocol_kind=1 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=2 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=3 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=4 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=5 --interpret -split-input-file %s
// AUTO GENERATED, DO NOT EDIT

func.func @sign_op_test_i64_i64_p() {
   %0 = arith.constant dense<[-1, 0, 1]> : tensor<3xi64>
   %1 = pphlo.sign %0 : (tensor<3xi64>)->tensor<3xi64>
   %2 = arith.constant dense<[-1, 0, 1]> : tensor<3xi64>
   pphlo.custom_call @expect_eq(%1, %2) : (tensor<3xi64>, tensor<3xi64>)->()
   func.return
}

// -----

func.func @sign_op_test_i64_i64_s() {
   %0 = arith.constant dense<[-1, 0, 1]> : tensor<3xi64>
   %1 = pphlo.convert %0 : (tensor<3xi64>)->tensor<3x!pphlo.secret<i64>>
   %2 = pphlo.sign %1 : (tensor<3x!pphlo.secret<i64>>)->tensor<3x!pphlo.secret<i64>>
   %3 = arith.constant dense<[-1, 0, 1]> : tensor<3xi64>
   pphlo.custom_call @expect_eq(%3, %2) : (tensor<3xi64>, tensor<3x!pphlo.secret<i64>>)->()
   func.return
}

// -----

func.func @sign_op_test_f64_f64_p() {
   %0 = arith.constant dense<[-1.0, 0.0, 1.0]> : tensor<3xf64>
   %1 = pphlo.sign %0 : (tensor<3xf64>)->tensor<3xf64>
   %2 = arith.constant dense<[-1.0, 0.0, 1.0]> : tensor<3xf64>
   pphlo.custom_call @expect_almost_eq(%1, %2) : (tensor<3xf64>, tensor<3xf64>)->()
   func.return
}

// -----

func.func @sign_op_test_f64_f64_s() {
   %0 = arith.constant dense<[-1.0, 0.0, 1.0]> : tensor<3xf64>
   %1 = pphlo.convert %0 : (tensor<3xf64>)->tensor<3x!pphlo.secret<f64>>
   %2 = pphlo.sign %1 : (tensor<3x!pphlo.secret<f64>>)->tensor<3x!pphlo.secret<f64>>
   %3 = arith.constant dense<[-1.0, 0.0, 1.0]> : tensor<3xf64>
   pphlo.custom_call @expect_almost_eq(%3, %2) : (tensor<3xf64>, tensor<3x!pphlo.secret<f64>>)->()
   func.return
}
