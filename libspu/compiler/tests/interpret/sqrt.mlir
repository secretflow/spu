// RUN: spu-translate --interpret -split-input-file %s
// AUTO GENERATED, DO NOT EDIT

func.func @sqrt_op_test_f64_f64_p() {
   %0 = pphlo.constant dense<[[0.0, 1.0], [4.0, 9.0]]> : tensor<2x2xf64>
   %1 = pphlo.sqrt %0 : (tensor<2x2xf64>)->tensor<2x2xf64>
   %2 = pphlo.constant dense<[[0.000000e+00, 1.000000e+00], [2.000000e+00, 3.000000e+00]]> : tensor<2x2xf64>
   pphlo.custom_call @expect_almost_eq(%1, %2) : (tensor<2x2xf64>, tensor<2x2xf64>)->()
   func.return
}

// -----

func.func @sqrt_op_test_f64_f64_s() {
   %0 = pphlo.constant dense<[[0.0, 1.0], [4.0, 9.0]]> : tensor<2x2xf64>
   %1 = pphlo.convert %0 : (tensor<2x2xf64>)->tensor<2x2x!pphlo.secret<f64>>
   %2 = pphlo.sqrt %1 : (tensor<2x2x!pphlo.secret<f64>>)->tensor<2x2x!pphlo.secret<f64>>
   %3 = pphlo.constant dense<[[0.000000e+00, 1.000000e+00], [2.000000e+00, 3.000000e+00]]> : tensor<2x2xf64>
   %4 = pphlo.convert %2 : (tensor<2x2x!pphlo.secret<f64>>)->tensor<2x2xf64>
   pphlo.custom_call @expect_almost_eq(%3, %4) : (tensor<2x2xf64>, tensor<2x2xf64>)->()
   func.return
}
