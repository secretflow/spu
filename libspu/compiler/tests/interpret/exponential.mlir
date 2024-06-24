// RUN: spu-translate --interpret -split-input-file %s
// AUTO GENERATED, DO NOT EDIT

func.func @exponential_op_test_f64_f64_p() {
   %0 = pphlo.constant dense<[[0.0, 1.0], [2.0, 3.0]]> : tensor<2x2xf64>
   %1 = pphlo.exponential %0 : (tensor<2x2xf64>)->tensor<2x2xf64>
   %2 = pphlo.constant dense<[[1.000000e+00, 2.7182818284590451], [7.3890560989306504, 20.085536923187668]]> : tensor<2x2xf64>
   pphlo.custom_call @expect_almost_eq(%1, %2) { tol = 0.4 }: (tensor<2x2xf64>, tensor<2x2xf64>)->()
   func.return
}

// -----

func.func @exponential_op_test_f64_f64_s() {
   %0 = pphlo.constant dense<[[0.0, 1.0], [2.0, 3.0]]> : tensor<2x2xf64>
   %1 = pphlo.convert %0 : (tensor<2x2xf64>)->tensor<2x2x!pphlo.secret<f64>>
   %2 = pphlo.exponential %1 : (tensor<2x2x!pphlo.secret<f64>>)->tensor<2x2x!pphlo.secret<f64>>
   %3 = pphlo.constant dense<[[1.000000e+00, 2.7182818284590451], [7.3890560989306504, 20.085536923187668]]> : tensor<2x2xf64>
   %4 = pphlo.convert %2 : (tensor<2x2x!pphlo.secret<f64>>)->tensor<2x2xf64>
   pphlo.custom_call @expect_almost_eq(%3, %4) { tol = 0.4 }: (tensor<2x2xf64>, tensor<2x2xf64>)->()
   func.return
}
