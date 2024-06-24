// RUN: spu-translate --interpret -split-input-file %s
// AUTO GENERATED, DO NOT EDIT

func.func @log_op_test_f64_f64_p() {
   %0 = pphlo.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf64>
   %1 = pphlo.log %0 : (tensor<2x2xf64>)->tensor<2x2xf64>
   %2 = pphlo.constant dense<[[0.000000e+00, 0.69314718055994529], [1.0986122886681098, 1.3862943611198906]]> : tensor<2x2xf64>
   pphlo.custom_call @expect_almost_eq(%1, %2) : (tensor<2x2xf64>, tensor<2x2xf64>)->()
   func.return
}

// -----

func.func @log_op_test_f64_f64_s() {
   %0 = pphlo.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf64>
   %1 = pphlo.convert %0 : (tensor<2x2xf64>)->tensor<2x2x!pphlo.secret<f64>>
   %2 = pphlo.log %1 : (tensor<2x2x!pphlo.secret<f64>>)->tensor<2x2x!pphlo.secret<f64>>
   %3 = pphlo.constant dense<[[0.000000e+00, 0.69314718055994529], [1.0986122886681098, 1.3862943611198906]]> : tensor<2x2xf64>
   %4 = pphlo.convert %2 : (tensor<2x2x!pphlo.secret<f64>>)->tensor<2x2xf64>
   pphlo.custom_call @expect_almost_eq(%3, %4) : (tensor<2x2xf64>, tensor<2x2xf64>)->()
   func.return
}
