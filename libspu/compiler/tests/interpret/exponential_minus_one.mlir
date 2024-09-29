// RUN: spu-translate --protocol_kind=1 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=2 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=3 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=4 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=5 --interpret -split-input-file %s
// AUTO GENERATED, DO NOT EDIT

func.func @exponential_minus_one_op_test_f64_f64_p() {
   %0 = arith.constant dense<[0.0, 1.0]> : tensor<2xf64>
   %1 = pphlo.exponential_minus_one %0 : (tensor<2xf64>)->tensor<2xf64>
   %2 = arith.constant dense<[0.0, 1.7182818284590451]> : tensor<2xf64>
   pphlo.custom_call @expect_almost_eq(%1, %2) : (tensor<2xf64>, tensor<2xf64>)->()
   func.return
}

// -----

func.func @exponential_minus_one_op_test_f64_f64_s() {
   %0 = arith.constant dense<[0.0, 1.0]> : tensor<2xf64>
   %1 = pphlo.convert %0 : (tensor<2xf64>)->tensor<2x!pphlo.secret<f64>>
   %2 = pphlo.exponential_minus_one %1 : (tensor<2x!pphlo.secret<f64>>)->tensor<2x!pphlo.secret<f64>>
   %3 = arith.constant dense<[0.0, 1.7182818284590451]> : tensor<2xf64>
   pphlo.custom_call @expect_almost_eq(%3, %2) : (tensor<2xf64>, tensor<2x!pphlo.secret<f64>>)->()
   func.return
}
