// RUN: spu-translate --protocol_kind=1 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=2 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=3 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=4 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=5 --interpret -split-input-file %s
// AUTO GENERATED, DO NOT EDIT

func.func @round_nearest_afz_op_test_f64_f64_p() {
   %0 = arith.constant dense<[-2.5, 0.4, 0.5, 0.6, 2.5]> : tensor<5xf64>
   %1 = pphlo.round_nearest_afz %0 : (tensor<5xf64>)->tensor<5xf64>
   %2 = arith.constant dense<[-3.0, 0.0, 1.0, 1.0, 3.0]> : tensor<5xf64>
   pphlo.custom_call @expect_almost_eq(%1, %2) : (tensor<5xf64>, tensor<5xf64>)->()
   func.return
}

// -----

func.func @round_nearest_afz_op_test_f64_f64_s() {
   %0 = arith.constant dense<[-2.5, 0.4, 0.5, 0.6, 2.5]> : tensor<5xf64>
   %1 = pphlo.convert %0 : (tensor<5xf64>)->tensor<5x!pphlo.secret<f64>>
   %2 = pphlo.round_nearest_afz %1 : (tensor<5x!pphlo.secret<f64>>)->tensor<5x!pphlo.secret<f64>>
   %3 = arith.constant dense<[-3.0, 0.0, 1.0, 1.0, 3.0]> : tensor<5xf64>
   pphlo.custom_call @expect_almost_eq(%3, %2) : (tensor<5xf64>, tensor<5x!pphlo.secret<f64>>)->()
   func.return
}
