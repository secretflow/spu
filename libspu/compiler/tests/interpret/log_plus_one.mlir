// RUN: spu-translate --protocol_kind=1 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=2 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=3 --interpret -split-input-file %s
// AUTO GENERATED, DO NOT EDIT

func.func @log_plus_one_op_test_f64_f64_p() {
   %0 = pphlo.constant dense<[0.0, -0.999, 7.0, 6.38905621, 15.0]> : tensor<5xf64>
   %1 = pphlo.log_plus_one %0 : (tensor<5xf64>)->tensor<5xf64>
   %2 = pphlo.constant dense<[0.0, -6.90776825, 2.07944155, 2.0, 2.77258873]> : tensor<5xf64>
   pphlo.custom_call @expect_almost_eq(%1, %2) : (tensor<5xf64>, tensor<5xf64>)->()
   func.return
}

// -----

func.func @log_plus_one_op_test_f64_f64_s() {
   %0 = pphlo.constant dense<[0.0, -0.999, 7.0, 6.38905621, 15.0]> : tensor<5xf64>
   %1 = pphlo.convert %0 : (tensor<5xf64>)->tensor<5x!pphlo.secret<f64>>
   %2 = pphlo.log_plus_one %1 : (tensor<5x!pphlo.secret<f64>>)->tensor<5x!pphlo.secret<f64>>
   %3 = pphlo.constant dense<[0.0, -6.90776825, 2.07944155, 2.0, 2.77258873]> : tensor<5xf64>
   %4 = pphlo.convert %2 : (tensor<5x!pphlo.secret<f64>>)->tensor<5xf64>
   pphlo.custom_call @expect_almost_eq(%3, %4) : (tensor<5xf64>, tensor<5xf64>)->()
   func.return
}
