// RUN: spu-translate --protocol_kind=1 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=2 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=3 --interpret -split-input-file %s
// AUTO GENERATED, DO NOT EDIT

func.func @cosine_op_test_f16_f16_p() {
   %0 = pphlo.constant dense<[0.0, 1.0, 0.125, 0.1, 3.140630]> : tensor<5xf16>
   %1 = pphlo.cosine %0 : (tensor<5xf16>)->tensor<5xf16>
   %2 = pphlo.constant dense<[1.000000e+00, 0.540302277, 0.992197692, 0.995004177, -1.000000e+00]> : tensor<5xf16>
   pphlo.custom_call @expect_almost_eq(%1, %2) : (tensor<5xf16>, tensor<5xf16>)->()
   func.return
}

// -----

func.func @cosine_op_test_f16_f16_s() {
   %0 = pphlo.constant dense<[0.0, 1.0, 0.125, 0.1, 3.140630]> : tensor<5xf16>
   %1 = pphlo.convert %0 : (tensor<5xf16>)->tensor<5x!pphlo.secret<f16>>
   %2 = pphlo.cosine %1 : (tensor<5x!pphlo.secret<f16>>)->tensor<5x!pphlo.secret<f16>>
   %3 = pphlo.constant dense<[1.000000e+00, 0.540302277, 0.992197692, 0.995004177, -1.000000e+00]> : tensor<5xf16>
   %4 = pphlo.convert %2 : (tensor<5x!pphlo.secret<f16>>)->tensor<5xf16>
   pphlo.custom_call @expect_almost_eq(%3, %4) : (tensor<5xf16>, tensor<5xf16>)->()
   func.return
}

// -----

func.func @cosine_op_test_f32_f32_p() {
   %0 = pphlo.constant dense<[0.0, 1.0, 0.125, 0.1, 3.14159274]> : tensor<5xf32>
   %1 = pphlo.cosine %0 : (tensor<5xf32>)->tensor<5xf32>
   %2 = pphlo.constant dense<[1.000000e+00, 0.540302277, 0.992197692, 0.995004177, -1.000000e+00]> : tensor<5xf32>
   pphlo.custom_call @expect_almost_eq(%1, %2) : (tensor<5xf32>, tensor<5xf32>)->()
   func.return
}

// -----

func.func @cosine_op_test_f32_f32_s() {
   %0 = pphlo.constant dense<[0.0, 1.0, 0.125, 0.1, 3.14159274]> : tensor<5xf32>
   %1 = pphlo.convert %0 : (tensor<5xf32>)->tensor<5x!pphlo.secret<f32>>
   %2 = pphlo.cosine %1 : (tensor<5x!pphlo.secret<f32>>)->tensor<5x!pphlo.secret<f32>>
   %3 = pphlo.constant dense<[1.000000e+00, 0.540302277, 0.992197692, 0.995004177, -1.000000e+00]> : tensor<5xf32>
   %4 = pphlo.convert %2 : (tensor<5x!pphlo.secret<f32>>)->tensor<5xf32>
   pphlo.custom_call @expect_almost_eq(%3, %4) : (tensor<5xf32>, tensor<5xf32>)->()
   func.return
}

// -----

func.func @cosine_op_test_f64_f64_p() {
   %0 = pphlo.constant dense<[0.0, 1.0, 0.125, 0.1, 3.1415926535897931]> : tensor<5xf64>
   %1 = pphlo.cosine %0 : (tensor<5xf64>)->tensor<5xf64>
   %2 = pphlo.constant dense<[1.000000e+00, 0.54030230586813977, 0.992197667229329, 0.99500416527802582, -1.000000e+00]> : tensor<5xf64>
   pphlo.custom_call @expect_almost_eq(%1, %2) : (tensor<5xf64>, tensor<5xf64>)->()
   func.return
}

// -----

func.func @cosine_op_test_f64_f64_s() {
   %0 = pphlo.constant dense<[0.0, 1.0, 0.125, 0.1, 3.1415926535897931]> : tensor<5xf64>
   %1 = pphlo.convert %0 : (tensor<5xf64>)->tensor<5x!pphlo.secret<f64>>
   %2 = pphlo.cosine %1 : (tensor<5x!pphlo.secret<f64>>)->tensor<5x!pphlo.secret<f64>>
   %3 = pphlo.constant dense<[1.000000e+00, 0.54030230586813977, 0.992197667229329, 0.99500416527802582, -1.000000e+00]> : tensor<5xf64>
   %4 = pphlo.convert %2 : (tensor<5x!pphlo.secret<f64>>)->tensor<5xf64>
   pphlo.custom_call @expect_almost_eq(%3, %4) : (tensor<5xf64>, tensor<5xf64>)->()
   func.return
}
