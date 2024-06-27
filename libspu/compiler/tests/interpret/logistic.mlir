// RUN: spu-translate --protocol_kind=1 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=2 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=3 --interpret -split-input-file %s
// AUTO GENERATED, DO NOT EDIT

func.func @logistic_op_test_f64_f64_p() {
   %0 = pphlo.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf64>
   %1 = pphlo.logistic %0 : (tensor<2x2xf64>)->tensor<2x2xf64>
   %2 = pphlo.constant dense<[[0.73105857863000488, 0.88079707797788244],[0.95257412682243322, 0.98201379003790844]]> : tensor<2x2xf64>
   pphlo.custom_call @expect_almost_eq(%1, %2) : (tensor<2x2xf64>, tensor<2x2xf64>)->()
   func.return
}

// -----

func.func @logistic_op_test_f64_f64_s() {
   %0 = pphlo.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf64>
   %1 = pphlo.convert %0 : (tensor<2x2xf64>)->tensor<2x2x!pphlo.secret<f64>>
   %2 = pphlo.logistic %1 : (tensor<2x2x!pphlo.secret<f64>>)->tensor<2x2x!pphlo.secret<f64>>
   %3 = pphlo.constant dense<[[0.73105857863000488, 0.88079707797788244],[0.95257412682243322, 0.98201379003790844]]> : tensor<2x2xf64>
   %4 = pphlo.convert %2 : (tensor<2x2x!pphlo.secret<f64>>)->tensor<2x2xf64>
   pphlo.custom_call @expect_almost_eq(%3, %4) : (tensor<2x2xf64>, tensor<2x2xf64>)->()
   func.return
}
