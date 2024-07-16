// RUN: spu-translate --protocol_kind=1 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=2 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=3 --interpret -split-input-file %s
// AUTO GENERATED, DO NOT EDIT

func.func @rsqrt_op_test_f64_f64_p() {
   %0 = pphlo.constant dense<[[1.0, 4.0], [9.0, 25.0]]> : tensor<2x2xf64>
   %1 = pphlo.rsqrt %0 : (tensor<2x2xf64>)->tensor<2x2xf64>
   %2 = pphlo.constant dense<[[1.000000e+00, 5.000000e-01], [0.33333333333333331, 2.000000e-01]]> : tensor<2x2xf64>
   pphlo.custom_call @expect_almost_eq(%1, %2) : (tensor<2x2xf64>, tensor<2x2xf64>)->()
   func.return
}

// -----

func.func @rsqrt_op_test_f64_f64_s() {
   %0 = pphlo.constant dense<[[1.0, 4.0], [9.0, 25.0]]> : tensor<2x2xf64>
   %1 = pphlo.convert %0 : (tensor<2x2xf64>)->tensor<2x2x!pphlo.secret<f64>>
   %2 = pphlo.rsqrt %1 : (tensor<2x2x!pphlo.secret<f64>>)->tensor<2x2x!pphlo.secret<f64>>
   %3 = pphlo.constant dense<[[1.000000e+00, 5.000000e-01], [0.33333333333333331, 2.000000e-01]]> : tensor<2x2xf64>
   %4 = pphlo.convert %2 : (tensor<2x2x!pphlo.secret<f64>>)->tensor<2x2xf64>
   pphlo.custom_call @expect_almost_eq(%3, %4) : (tensor<2x2xf64>, tensor<2x2xf64>)->()
   func.return
}
