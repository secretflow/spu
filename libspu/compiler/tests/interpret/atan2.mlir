// RUN: spu-translate --protocol_kind=1 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=2 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=3 --interpret -split-input-file %s
// AUTO GENERATED, DO NOT EDIT

func.func @atan2_op_test_f64_f64_pp() {
   %0 = pphlo.constant dense<[0.0, 1.0, -1.0]> : tensor<3xf64>
   %1 = pphlo.constant dense<[0.0, 0.0, 0.0]> : tensor<3xf64>
   %2 = pphlo.atan2 %0,%1 : (tensor<3xf64>,tensor<3xf64>)->tensor<3xf64>
   %3 = pphlo.constant dense<[0.0, 1.5707963267948966, -1.5707963267948966]> : tensor<3xf64>
   pphlo.custom_call @expect_almost_eq(%2, %3) : (tensor<3xf64>, tensor<3xf64>)->()
   func.return
}

// -----

func.func @atan2_op_test_f64_f64_ss() {
   %0 = pphlo.constant dense<[0.0, 1.0, -1.0]> : tensor<3xf64>
   %1 = pphlo.constant dense<[0.0, 0.0, 0.0]> : tensor<3xf64>
   %2 = pphlo.convert %0 : (tensor<3xf64>)->tensor<3x!pphlo.secret<f64>>
   %3 = pphlo.convert %1 : (tensor<3xf64>)->tensor<3x!pphlo.secret<f64>>
   %4 = pphlo.atan2 %2, %3 : (tensor<3x!pphlo.secret<f64>>,tensor<3x!pphlo.secret<f64>>)->tensor<3x!pphlo.secret<f64>>
   %5 = pphlo.constant dense<[0.0, 1.5707963267948966, -1.5707963267948966]> : tensor<3xf64>
   %6 = pphlo.convert %4 : (tensor<3x!pphlo.secret<f64>>)->tensor<3xf64>
   pphlo.custom_call @expect_almost_eq(%5, %6) : (tensor<3xf64>, tensor<3xf64>)->()
   func.return
}
