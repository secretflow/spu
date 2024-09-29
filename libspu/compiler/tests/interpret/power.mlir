// RUN: spu-translate --protocol_kind=1 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=2 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=3 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=4 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=5 --interpret -split-input-file %s
// AUTO GENERATED, DO NOT EDIT

func.func @power_op_test_i64_i64_pp() {
   %0 = arith.constant dense<[-1, -1, -3, 1, -3, 0]> : tensor<6xi64>
   %1 = arith.constant dense<[1, 0, -3, -3, 3, 2]> : tensor<6xi64>
   %2 = pphlo.power %0,%1 : (tensor<6xi64>,tensor<6xi64>)->tensor<6xi64>
   %3 = arith.constant dense<[-1, 1, 0, 1, -27, 0]> : tensor<6xi64>
   pphlo.custom_call @expect_eq(%2, %3) : (tensor<6xi64>, tensor<6xi64>)->()
   func.return
}

// -----

func.func @power_op_test_i64_i64_ss() {
   %0 = arith.constant dense<[-1, -1, -3, 1, -3, 0]> : tensor<6xi64>
   %1 = arith.constant dense<[1, 0, -3, -3, 3, 2]> : tensor<6xi64>
   %2 = pphlo.convert %0 : (tensor<6xi64>)->tensor<6x!pphlo.secret<i64>>
   %3 = pphlo.convert %1 : (tensor<6xi64>)->tensor<6x!pphlo.secret<i64>>
   %4 = pphlo.power %2, %3 : (tensor<6x!pphlo.secret<i64>>,tensor<6x!pphlo.secret<i64>>)->tensor<6x!pphlo.secret<i64>>
   %5 = arith.constant dense<[-1, 1, 0, 1, -27, 0]> : tensor<6xi64>
   pphlo.custom_call @expect_eq(%5, %4) : (tensor<6xi64>, tensor<6x!pphlo.secret<i64>>)->()
   func.return
}

// -----

func.func @power_op_test_ui64_ui64_pp() {
   %0 = arith.constant dense<[0, 0, 1, 1, 5]> : tensor<5xui64>
   %1 = arith.constant dense<[0, 1, 0, 2, 5]> : tensor<5xui64>
   %2 = pphlo.power %0,%1 : (tensor<5xui64>,tensor<5xui64>)->tensor<5xui64>
   %3 = arith.constant dense<[1, 0, 1, 1, 3125]> : tensor<5xui64>
   pphlo.custom_call @expect_eq(%2, %3) : (tensor<5xui64>, tensor<5xui64>)->()
   func.return
}

// -----

func.func @power_op_test_ui64_ui64_ss() {
   %0 = arith.constant dense<[0, 0, 1, 1, 5]> : tensor<5xui64>
   %1 = arith.constant dense<[0, 1, 0, 2, 5]> : tensor<5xui64>
   %2 = pphlo.convert %0 : (tensor<5xui64>)->tensor<5x!pphlo.secret<ui64>>
   %3 = pphlo.convert %1 : (tensor<5xui64>)->tensor<5x!pphlo.secret<ui64>>
   %4 = pphlo.power %2, %3 : (tensor<5x!pphlo.secret<ui64>>,tensor<5x!pphlo.secret<ui64>>)->tensor<5x!pphlo.secret<ui64>>
   %5 = arith.constant dense<[1, 0, 1, 1, 3125]> : tensor<5xui64>
   pphlo.custom_call @expect_eq(%5, %4) : (tensor<5xui64>, tensor<5x!pphlo.secret<ui64>>)->()
   func.return
}

// -----

func.func @power_op_test_f64_f64_pp() {
   %0 = arith.constant dense<[-2.0, -0.0, 5.0, 3.0, 10000.0]> : tensor<5xf64>
   %1 = arith.constant dense<[2.0, 2.0, 2.0, -1.0, 1.0]> : tensor<5xf64>
   %2 = pphlo.power %0,%1 : (tensor<5xf64>,tensor<5xf64>)->tensor<5xf64>
   %3 = arith.constant dense<[4.000000e+00, 0.000000e+00, 2.500000e+01, 0.33333333333333331, 10000.0]> : tensor<5xf64>
   pphlo.custom_call @expect_almost_eq(%2, %3) { tol = 0.6 }: (tensor<5xf64>, tensor<5xf64>)->()
   func.return
}

// -----

func.func @power_op_test_f64_f64_ss() {
   %0 = arith.constant dense<[-2.0, -0.0, 5.0, 3.0, 10000.0]> : tensor<5xf64>
   %1 = arith.constant dense<[2.0, 2.0, 2.0, -1.0, 1.0]> : tensor<5xf64>
   %2 = pphlo.convert %0 : (tensor<5xf64>)->tensor<5x!pphlo.secret<f64>>
   %3 = pphlo.convert %1 : (tensor<5xf64>)->tensor<5x!pphlo.secret<f64>>
   %4 = pphlo.power %2, %3 : (tensor<5x!pphlo.secret<f64>>,tensor<5x!pphlo.secret<f64>>)->tensor<5x!pphlo.secret<f64>>
   %5 = arith.constant dense<[4.000000e+00, 0.000000e+00, 2.500000e+01, 0.33333333333333331, 10000.0]> : tensor<5xf64>
   pphlo.custom_call @expect_almost_eq(%5, %4) { tol = 0.6 }: (tensor<5xf64>, tensor<5x!pphlo.secret<f64>>)->()
   func.return
}
