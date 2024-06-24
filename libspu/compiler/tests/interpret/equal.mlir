// RUN: spu-translate --interpret -split-input-file %s
// AUTO GENERATED, DO NOT EDIT

func.func @equal_op_test_i64_i1_pp() {
   %0 = pphlo.constant dense<-2> : tensor<i64>
   %1 = pphlo.constant dense<-2> : tensor<i64>
   %2 = pphlo.equal %0,%1 : (tensor<i64>,tensor<i64>)->tensor<i1>
   %3 = pphlo.constant dense<true> : tensor<i1>
   pphlo.custom_call @expect_eq(%2, %3) : (tensor<i1>, tensor<i1>)->()
   func.return
}

// -----

func.func @equal_op_test_i64_i1_ss() {
   %0 = pphlo.constant dense<-2> : tensor<i64>
   %1 = pphlo.constant dense<-2> : tensor<i64>
   %2 = pphlo.convert %0 : (tensor<i64>)->tensor<!pphlo.secret<i64>>
   %3 = pphlo.convert %1 : (tensor<i64>)->tensor<!pphlo.secret<i64>>
   %4 = pphlo.equal %2, %3 : (tensor<!pphlo.secret<i64>>,tensor<!pphlo.secret<i64>>)->tensor<!pphlo.secret<i1>>
   %5 = pphlo.constant dense<true> : tensor<i1>
   %6 = pphlo.convert %4 : (tensor<!pphlo.secret<i1>>)->tensor<i1>
   pphlo.custom_call @expect_eq(%5, %6) : (tensor<i1>, tensor<i1>)->()
   func.return
}

// -----

func.func @equal_op_test_i64_i1_pp() {
   %0 = pphlo.constant dense<[-2, -1, 0, 2, 2]> : tensor<5xi64>
   %1 = pphlo.constant dense<[-2, -2, 0, 1, 2]> : tensor<5xi64>
   %2 = pphlo.equal %0,%1 : (tensor<5xi64>,tensor<5xi64>)->tensor<5xi1>
   %3 = pphlo.constant dense<[true, false, true, false, true]> : tensor<5xi1>
   pphlo.custom_call @expect_eq(%2, %3) : (tensor<5xi1>, tensor<5xi1>)->()
   func.return
}

// -----

func.func @equal_op_test_i64_i1_ss() {
   %0 = pphlo.constant dense<[-2, -1, 0, 2, 2]> : tensor<5xi64>
   %1 = pphlo.constant dense<[-2, -2, 0, 1, 2]> : tensor<5xi64>
   %2 = pphlo.convert %0 : (tensor<5xi64>)->tensor<5x!pphlo.secret<i64>>
   %3 = pphlo.convert %1 : (tensor<5xi64>)->tensor<5x!pphlo.secret<i64>>
   %4 = pphlo.equal %2, %3 : (tensor<5x!pphlo.secret<i64>>,tensor<5x!pphlo.secret<i64>>)->tensor<5x!pphlo.secret<i1>>
   %5 = pphlo.constant dense<[true, false, true, false, true]> : tensor<5xi1>
   %6 = pphlo.convert %4 : (tensor<5x!pphlo.secret<i1>>)->tensor<5xi1>
   pphlo.custom_call @expect_eq(%5, %6) : (tensor<5xi1>, tensor<5xi1>)->()
   func.return
}

// -----

func.func @equal_op_test_ui64_i1_pp() {
   %0 = pphlo.constant dense<0> : tensor<ui64>
   %1 = pphlo.constant dense<0> : tensor<ui64>
   %2 = pphlo.equal %0,%1 : (tensor<ui64>,tensor<ui64>)->tensor<i1>
   %3 = pphlo.constant dense<true> : tensor<i1>
   pphlo.custom_call @expect_eq(%2, %3) : (tensor<i1>, tensor<i1>)->()
   func.return
}

// -----

func.func @equal_op_test_ui64_i1_ss() {
   %0 = pphlo.constant dense<0> : tensor<ui64>
   %1 = pphlo.constant dense<0> : tensor<ui64>
   %2 = pphlo.convert %0 : (tensor<ui64>)->tensor<!pphlo.secret<ui64>>
   %3 = pphlo.convert %1 : (tensor<ui64>)->tensor<!pphlo.secret<ui64>>
   %4 = pphlo.equal %2, %3 : (tensor<!pphlo.secret<ui64>>,tensor<!pphlo.secret<ui64>>)->tensor<!pphlo.secret<i1>>
   %5 = pphlo.constant dense<true> : tensor<i1>
   %6 = pphlo.convert %4 : (tensor<!pphlo.secret<i1>>)->tensor<i1>
   pphlo.custom_call @expect_eq(%5, %6) : (tensor<i1>, tensor<i1>)->()
   func.return
}

// -----

func.func @equal_op_test_ui64_i1_pp() {
   %0 = pphlo.constant dense<[0, 1]> : tensor<2xui64>
   %1 = pphlo.constant dense<[0, 0]> : tensor<2xui64>
   %2 = pphlo.equal %0,%1 : (tensor<2xui64>,tensor<2xui64>)->tensor<2xi1>
   %3 = pphlo.constant dense<[true, false]> : tensor<2xi1>
   pphlo.custom_call @expect_eq(%2, %3) : (tensor<2xi1>, tensor<2xi1>)->()
   func.return
}

// -----

func.func @equal_op_test_ui64_i1_ss() {
   %0 = pphlo.constant dense<[0, 1]> : tensor<2xui64>
   %1 = pphlo.constant dense<[0, 0]> : tensor<2xui64>
   %2 = pphlo.convert %0 : (tensor<2xui64>)->tensor<2x!pphlo.secret<ui64>>
   %3 = pphlo.convert %1 : (tensor<2xui64>)->tensor<2x!pphlo.secret<ui64>>
   %4 = pphlo.equal %2, %3 : (tensor<2x!pphlo.secret<ui64>>,tensor<2x!pphlo.secret<ui64>>)->tensor<2x!pphlo.secret<i1>>
   %5 = pphlo.constant dense<[true, false]> : tensor<2xi1>
   %6 = pphlo.convert %4 : (tensor<2x!pphlo.secret<i1>>)->tensor<2xi1>
   pphlo.custom_call @expect_eq(%5, %6) : (tensor<2xi1>, tensor<2xi1>)->()
   func.return
}

// -----

func.func @equal_op_test_i1_i1_pp() {
   %0 = pphlo.constant dense<true> : tensor<i1>
   %1 = pphlo.constant dense<true> : tensor<i1>
   %2 = pphlo.equal %0,%1 : (tensor<i1>,tensor<i1>)->tensor<i1>
   %3 = pphlo.constant dense<true> : tensor<i1>
   pphlo.custom_call @expect_eq(%2, %3) : (tensor<i1>, tensor<i1>)->()
   func.return
}

// -----

func.func @equal_op_test_i1_i1_ss() {
   %0 = pphlo.constant dense<true> : tensor<i1>
   %1 = pphlo.constant dense<true> : tensor<i1>
   %2 = pphlo.convert %0 : (tensor<i1>)->tensor<!pphlo.secret<i1>>
   %3 = pphlo.convert %1 : (tensor<i1>)->tensor<!pphlo.secret<i1>>
   %4 = pphlo.equal %2, %3 : (tensor<!pphlo.secret<i1>>,tensor<!pphlo.secret<i1>>)->tensor<!pphlo.secret<i1>>
   %5 = pphlo.constant dense<true> : tensor<i1>
   %6 = pphlo.convert %4 : (tensor<!pphlo.secret<i1>>)->tensor<i1>
   pphlo.custom_call @expect_eq(%5, %6) : (tensor<i1>, tensor<i1>)->()
   func.return
}

// -----

func.func @equal_op_test_i1_i1_pp() {
   %0 = pphlo.constant dense<[true, true, false, false]> : tensor<4xi1>
   %1 = pphlo.constant dense<[true, false, true, false]> : tensor<4xi1>
   %2 = pphlo.equal %0,%1 : (tensor<4xi1>,tensor<4xi1>)->tensor<4xi1>
   %3 = pphlo.constant dense<[true, false, false, true]> : tensor<4xi1>
   pphlo.custom_call @expect_eq(%2, %3) : (tensor<4xi1>, tensor<4xi1>)->()
   func.return
}

// -----

func.func @equal_op_test_i1_i1_ss() {
   %0 = pphlo.constant dense<[true, true, false, false]> : tensor<4xi1>
   %1 = pphlo.constant dense<[true, false, true, false]> : tensor<4xi1>
   %2 = pphlo.convert %0 : (tensor<4xi1>)->tensor<4x!pphlo.secret<i1>>
   %3 = pphlo.convert %1 : (tensor<4xi1>)->tensor<4x!pphlo.secret<i1>>
   %4 = pphlo.equal %2, %3 : (tensor<4x!pphlo.secret<i1>>,tensor<4x!pphlo.secret<i1>>)->tensor<4x!pphlo.secret<i1>>
   %5 = pphlo.constant dense<[true, false, false, true]> : tensor<4xi1>
   %6 = pphlo.convert %4 : (tensor<4x!pphlo.secret<i1>>)->tensor<4xi1>
   pphlo.custom_call @expect_eq(%5, %6) : (tensor<4xi1>, tensor<4xi1>)->()
   func.return
}

// -----

func.func @equal_op_test_f64_i1_pp() {
   %0 = pphlo.constant dense<[-2.0, -2.0, 0.0, 1.0, 2.0]> : tensor<5xf64>
   %1 = pphlo.constant dense<[-2.0, -1.0, 0.0, 2.0, 2.0]> : tensor<5xf64>
   %2 = pphlo.equal %0,%1 : (tensor<5xf64>,tensor<5xf64>)->tensor<5xi1>
   %3 = pphlo.constant dense<[true, false, true, false, true]> : tensor<5xi1>
   pphlo.custom_call @expect_eq(%2, %3) : (tensor<5xi1>, tensor<5xi1>)->()
   func.return
}

// -----

func.func @equal_op_test_f64_i1_ss() {
   %0 = pphlo.constant dense<[-2.0, -2.0, 0.0, 1.0, 2.0]> : tensor<5xf64>
   %1 = pphlo.constant dense<[-2.0, -1.0, 0.0, 2.0, 2.0]> : tensor<5xf64>
   %2 = pphlo.convert %0 : (tensor<5xf64>)->tensor<5x!pphlo.secret<f64>>
   %3 = pphlo.convert %1 : (tensor<5xf64>)->tensor<5x!pphlo.secret<f64>>
   %4 = pphlo.equal %2, %3 : (tensor<5x!pphlo.secret<f64>>,tensor<5x!pphlo.secret<f64>>)->tensor<5x!pphlo.secret<i1>>
   %5 = pphlo.constant dense<[true, false, true, false, true]> : tensor<5xi1>
   %6 = pphlo.convert %4 : (tensor<5x!pphlo.secret<i1>>)->tensor<5xi1>
   pphlo.custom_call @expect_eq(%5, %6) : (tensor<5xi1>, tensor<5xi1>)->()
   func.return
}
