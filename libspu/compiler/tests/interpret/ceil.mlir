// RUN: spu-translate --interpret -split-input-file %s
// AUTO GENERATED, DO NOT EDIT

func.func @ceil_op_test_f16_f16_p() {
   %0 = pphlo.constant dense<[-2.5, 0.0, 2.5]> : tensor<3xf16>
   %1 = pphlo.ceil %0 : (tensor<3xf16>)->tensor<3xf16>
   %2 = pphlo.constant dense<[-2.000000e+00, 0.000000e+00, 3.000000e+00]> : tensor<3xf16>
   pphlo.custom_call @expect_almost_eq(%1, %2) : (tensor<3xf16>, tensor<3xf16>)->()
   func.return
}

// -----

func.func @ceil_op_test_f16_f16_s() {
   %0 = pphlo.constant dense<[-2.5, 0.0, 2.5]> : tensor<3xf16>
   %1 = pphlo.convert %0 : (tensor<3xf16>)->tensor<3x!pphlo.secret<f16>>
   %2 = pphlo.ceil %1 : (tensor<3x!pphlo.secret<f16>>)->tensor<3x!pphlo.secret<f16>>
   %3 = pphlo.constant dense<[-2.000000e+00, 0.000000e+00, 3.000000e+00]> : tensor<3xf16>
   %4 = pphlo.convert %2 : (tensor<3x!pphlo.secret<f16>>)->tensor<3xf16>
   pphlo.custom_call @expect_almost_eq(%3, %4) : (tensor<3xf16>, tensor<3xf16>)->()
   func.return
}

// -----

func.func @ceil_op_test_f32_f32_p() {
   %0 = pphlo.constant dense<[-2.5, 0.0, 2.5]> : tensor<3xf32>
   %1 = pphlo.ceil %0 : (tensor<3xf32>)->tensor<3xf32>
   %2 = pphlo.constant dense<[-2.000000e+00, 0.000000e+00, 3.000000e+00]> : tensor<3xf32>
   pphlo.custom_call @expect_almost_eq(%1, %2) : (tensor<3xf32>, tensor<3xf32>)->()
   func.return
}

// -----

func.func @ceil_op_test_f32_f32_s() {
   %0 = pphlo.constant dense<[-2.5, 0.0, 2.5]> : tensor<3xf32>
   %1 = pphlo.convert %0 : (tensor<3xf32>)->tensor<3x!pphlo.secret<f32>>
   %2 = pphlo.ceil %1 : (tensor<3x!pphlo.secret<f32>>)->tensor<3x!pphlo.secret<f32>>
   %3 = pphlo.constant dense<[-2.000000e+00, 0.000000e+00, 3.000000e+00]> : tensor<3xf32>
   %4 = pphlo.convert %2 : (tensor<3x!pphlo.secret<f32>>)->tensor<3xf32>
   pphlo.custom_call @expect_almost_eq(%3, %4) : (tensor<3xf32>, tensor<3xf32>)->()
   func.return
}

// -----

func.func @ceil_op_test_f64_f64_p() {
   %0 = pphlo.constant dense<[-2.5, 0.0, 2.5]> : tensor<3xf64>
   %1 = pphlo.ceil %0 : (tensor<3xf64>)->tensor<3xf64>
   %2 = pphlo.constant dense<[-2.000000e+00, 0.000000e+00, 3.000000e+00]> : tensor<3xf64>
   pphlo.custom_call @expect_almost_eq(%1, %2) : (tensor<3xf64>, tensor<3xf64>)->()
   func.return
}

// -----

func.func @ceil_op_test_f64_f64_s() {
   %0 = pphlo.constant dense<[-2.5, 0.0, 2.5]> : tensor<3xf64>
   %1 = pphlo.convert %0 : (tensor<3xf64>)->tensor<3x!pphlo.secret<f64>>
   %2 = pphlo.ceil %1 : (tensor<3x!pphlo.secret<f64>>)->tensor<3x!pphlo.secret<f64>>
   %3 = pphlo.constant dense<[-2.000000e+00, 0.000000e+00, 3.000000e+00]> : tensor<3xf64>
   %4 = pphlo.convert %2 : (tensor<3x!pphlo.secret<f64>>)->tensor<3xf64>
   pphlo.custom_call @expect_almost_eq(%3, %4) : (tensor<3xf64>, tensor<3xf64>)->()
   func.return
}
