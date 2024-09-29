// RUN: spu-translate --protocol_kind=1 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=2 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=3 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=4 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=5 --interpret -split-input-file %s
// AUTO GENERATED, DO NOT EDIT

func.func @divide_op_test_i64_i64_pp() {
   %0 = arith.constant dense<[17, -17, 17, -17]> : tensor<4xi64>
   %1 = arith.constant dense<[3, 3, -3, -3]> : tensor<4xi64>
   %2 = pphlo.divide %0,%1 : (tensor<4xi64>,tensor<4xi64>)->tensor<4xi64>
   %3 = arith.constant dense<[5, -5, -5, 5]> : tensor<4xi64>
   pphlo.custom_call @expect_eq(%2, %3) : (tensor<4xi64>, tensor<4xi64>)->()
   func.return
}

// -----

func.func @divide_op_test_i64_i64_ss() {
   %0 = arith.constant dense<[17, -17, 17, -17]> : tensor<4xi64>
   %1 = arith.constant dense<[3, 3, -3, -3]> : tensor<4xi64>
   %2 = pphlo.convert %0 : (tensor<4xi64>)->tensor<4x!pphlo.secret<i64>>
   %3 = pphlo.convert %1 : (tensor<4xi64>)->tensor<4x!pphlo.secret<i64>>
   %4 = pphlo.divide %2, %3 : (tensor<4x!pphlo.secret<i64>>,tensor<4x!pphlo.secret<i64>>)->tensor<4x!pphlo.secret<i64>>
   %5 = arith.constant dense<[5, -5, -5, 5]> : tensor<4xi64>
   pphlo.custom_call @expect_eq(%5, %4) : (tensor<4xi64>, tensor<4x!pphlo.secret<i64>>)->()
   func.return
}

// -----

func.func @divide_op_test_ui64_ui64_pp() {
   %0 = arith.constant dense<[17, 18, 19, 20]> : tensor<4xui64>
   %1 = arith.constant dense<[3, 4, 5, 7]> : tensor<4xui64>
   %2 = pphlo.divide %0,%1 : (tensor<4xui64>,tensor<4xui64>)->tensor<4xui64>
   %3 = arith.constant dense<[5, 4, 3, 2]> : tensor<4xui64>
   pphlo.custom_call @expect_eq(%2, %3) : (tensor<4xui64>, tensor<4xui64>)->()
   func.return
}

// -----

func.func @divide_op_test_ui64_ui64_ss() {
   %0 = arith.constant dense<[17, 18, 19, 20]> : tensor<4xui64>
   %1 = arith.constant dense<[3, 4, 5, 7]> : tensor<4xui64>
   %2 = pphlo.convert %0 : (tensor<4xui64>)->tensor<4x!pphlo.secret<ui64>>
   %3 = pphlo.convert %1 : (tensor<4xui64>)->tensor<4x!pphlo.secret<ui64>>
   %4 = pphlo.divide %2, %3 : (tensor<4x!pphlo.secret<ui64>>,tensor<4x!pphlo.secret<ui64>>)->tensor<4x!pphlo.secret<ui64>>
   %5 = arith.constant dense<[5, 4, 3, 2]> : tensor<4xui64>
   pphlo.custom_call @expect_eq(%5, %4) : (tensor<4xui64>, tensor<4x!pphlo.secret<ui64>>)->()
   func.return
}

// -----

func.func @divide_op_test_f64_f64_pp() {
   %0 = arith.constant dense<[17.1, -17.1, 17.1, -17.1]> : tensor<4xf64>
   %1 = arith.constant dense<[3.0, 3.0, -3.0, -3.0]> : tensor<4xf64>
   %2 = pphlo.divide %0,%1 : (tensor<4xf64>,tensor<4xf64>)->tensor<4xf64>
   %3 = arith.constant dense<[5.700000e+00, -5.700000e+00, -5.700000e+00, 5.700000e+00]> : tensor<4xf64>
   pphlo.custom_call @expect_almost_eq(%2, %3) : (tensor<4xf64>, tensor<4xf64>)->()
   func.return
}

// -----

func.func @divide_op_test_f64_f64_ss() {
   %0 = arith.constant dense<[17.1, -17.1, 17.1, -17.1]> : tensor<4xf64>
   %1 = arith.constant dense<[3.0, 3.0, -3.0, -3.0]> : tensor<4xf64>
   %2 = pphlo.convert %0 : (tensor<4xf64>)->tensor<4x!pphlo.secret<f64>>
   %3 = pphlo.convert %1 : (tensor<4xf64>)->tensor<4x!pphlo.secret<f64>>
   %4 = pphlo.divide %2, %3 : (tensor<4x!pphlo.secret<f64>>,tensor<4x!pphlo.secret<f64>>)->tensor<4x!pphlo.secret<f64>>
   %5 = arith.constant dense<[5.700000e+00, -5.700000e+00, -5.700000e+00, 5.700000e+00]> : tensor<4xf64>
   pphlo.custom_call @expect_almost_eq(%5, %4) : (tensor<4xf64>, tensor<4x!pphlo.secret<f64>>)->()
   func.return
}
