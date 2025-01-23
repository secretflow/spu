// RUN: spu-translate --protocol_kind=1 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=2 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=3 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=4 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=5 --interpret -split-input-file %s
// AUTO GENERATED, DO NOT EDIT

func.func @greater_op_test_i64_i1_pp() {
   %0 = pphlo.constant dense<[-2, -1, 0, 2, 2]> : tensor<5xi64>
   %1 = pphlo.constant dense<[-2, -2, 0, 1, 2]> : tensor<5xi64>
   %2 = pphlo.greater %0,%1 : (tensor<5xi64>,tensor<5xi64>)->tensor<5xi1>
   %3 = pphlo.constant dense<[false, true, false, true, false]> : tensor<5xi1>
   pphlo.custom_call @expect_eq(%2, %3) : (tensor<5xi1>, tensor<5xi1>)->()
   func.return
}

// -----

func.func @greater_op_test_i64_i1_ss() {
   %0 = pphlo.constant dense<[-2, -1, 0, 2, 2]> : tensor<5xi64>
   %1 = pphlo.constant dense<[-2, -2, 0, 1, 2]> : tensor<5xi64>
   %2 = pphlo.convert %0 : (tensor<5xi64>)->tensor<5x!pphlo.secret<i64>>
   %3 = pphlo.convert %1 : (tensor<5xi64>)->tensor<5x!pphlo.secret<i64>>
   %4 = pphlo.greater %2, %3 : (tensor<5x!pphlo.secret<i64>>,tensor<5x!pphlo.secret<i64>>)->tensor<5x!pphlo.secret<i1>>
   %5 = pphlo.constant dense<[false, true, false, true, false]> : tensor<5xi1>
   %6 = pphlo.convert %4 : (tensor<5x!pphlo.secret<i1>>)->tensor<5xi1>
   pphlo.custom_call @expect_eq(%5, %6) : (tensor<5xi1>, tensor<5xi1>)->()
   func.return
}

// -----

func.func @greater_op_test_ui64_i1_pp() {
   %0 = pphlo.constant dense<[0, 1]> : tensor<2xui64>
   %1 = pphlo.constant dense<[0, 0]> : tensor<2xui64>
   %2 = pphlo.greater %0,%1 : (tensor<2xui64>,tensor<2xui64>)->tensor<2xi1>
   %3 = pphlo.constant dense<[false, true]> : tensor<2xi1>
   pphlo.custom_call @expect_eq(%2, %3) : (tensor<2xi1>, tensor<2xi1>)->()
   func.return
}

// -----

func.func @greater_op_test_ui64_i1_ss() {
   %0 = pphlo.constant dense<[0, 1]> : tensor<2xui64>
   %1 = pphlo.constant dense<[0, 0]> : tensor<2xui64>
   %2 = pphlo.convert %0 : (tensor<2xui64>)->tensor<2x!pphlo.secret<ui64>>
   %3 = pphlo.convert %1 : (tensor<2xui64>)->tensor<2x!pphlo.secret<ui64>>
   %4 = pphlo.greater %2, %3 : (tensor<2x!pphlo.secret<ui64>>,tensor<2x!pphlo.secret<ui64>>)->tensor<2x!pphlo.secret<i1>>
   %5 = pphlo.constant dense<[false, true]> : tensor<2xi1>
   %6 = pphlo.convert %4 : (tensor<2x!pphlo.secret<i1>>)->tensor<2xi1>
   pphlo.custom_call @expect_eq(%5, %6) : (tensor<2xi1>, tensor<2xi1>)->()
   func.return
}

// -----

func.func @greater_op_test_i1_i1_pp() {
   %0 = pphlo.constant dense<[true, true, false, false]> : tensor<4xi1>
   %1 = pphlo.constant dense<[true, false, true, false]> : tensor<4xi1>
   %2 = pphlo.greater %0,%1 : (tensor<4xi1>,tensor<4xi1>)->tensor<4xi1>
   %3 = pphlo.constant dense<[false, true, false, false]> : tensor<4xi1>
   pphlo.custom_call @expect_eq(%2, %3) : (tensor<4xi1>, tensor<4xi1>)->()
   func.return
}

// -----

func.func @greater_op_test_i1_i1_ss() {
   %0 = pphlo.constant dense<[true, true, false, false]> : tensor<4xi1>
   %1 = pphlo.constant dense<[true, false, true, false]> : tensor<4xi1>
   %2 = pphlo.convert %0 : (tensor<4xi1>)->tensor<4x!pphlo.secret<i1>>
   %3 = pphlo.convert %1 : (tensor<4xi1>)->tensor<4x!pphlo.secret<i1>>
   %4 = pphlo.greater %2, %3 : (tensor<4x!pphlo.secret<i1>>,tensor<4x!pphlo.secret<i1>>)->tensor<4x!pphlo.secret<i1>>
   %5 = pphlo.constant dense<[false, true, false, false]> : tensor<4xi1>
   %6 = pphlo.convert %4 : (tensor<4x!pphlo.secret<i1>>)->tensor<4xi1>
   pphlo.custom_call @expect_eq(%5, %6) : (tensor<4xi1>, tensor<4xi1>)->()
   func.return
}

// -----

func.func @greater_op_test_f64_i1_pp() {
   %0 = pphlo.constant dense<[-2.0, -2.0, 0.0, 1.0, 2.0]> : tensor<5xf64>
   %1 = pphlo.constant dense<[-2.0, -1.0, 0.0, 2.0, 2.0]> : tensor<5xf64>
   %2 = pphlo.greater %0,%1 : (tensor<5xf64>,tensor<5xf64>)->tensor<5xi1>
   %3 = pphlo.constant dense<[false, false, false, false, false]> : tensor<5xi1>
   pphlo.custom_call @expect_eq(%2, %3) : (tensor<5xi1>, tensor<5xi1>)->()
   func.return
}

// -----

func.func @greater_op_test_f64_i1_ss() {
   %0 = pphlo.constant dense<[-2.0, -2.0, 0.0, 1.0, 2.0]> : tensor<5xf64>
   %1 = pphlo.constant dense<[-2.0, -1.0, 0.0, 2.0, 2.0]> : tensor<5xf64>
   %2 = pphlo.convert %0 : (tensor<5xf64>)->tensor<5x!pphlo.secret<f64>>
   %3 = pphlo.convert %1 : (tensor<5xf64>)->tensor<5x!pphlo.secret<f64>>
   %4 = pphlo.greater %2, %3 : (tensor<5x!pphlo.secret<f64>>,tensor<5x!pphlo.secret<f64>>)->tensor<5x!pphlo.secret<i1>>
   %5 = pphlo.constant dense<[false, false, false, false, false]> : tensor<5xi1>
   %6 = pphlo.convert %4 : (tensor<5x!pphlo.secret<i1>>)->tensor<5xi1>
   pphlo.custom_call @expect_eq(%5, %6) : (tensor<5xi1>, tensor<5xi1>)->()
   func.return
}
