// RUN: spu-opt --hlo-legalize-to-pphlo --split-input-file %s | FileCheck %s

func.func @main() {
    // CHECK: arith.constant dense<true> : tensor<2x2xi1>
    %0 = stablehlo.constant dense<true> : tensor<2x2xi1>
    // CHECK: arith.constant dense<1> : tensor<2x2xi8>
    %1 = stablehlo.constant dense<1> : tensor<2x2xi8>
    // CHECK: arith.constant dense<1> : tensor<2x2xi16>
    %2 = stablehlo.constant dense<1> : tensor<2x2xi16>
    // CHECK: arith.constant dense<1> : tensor<2x2xi32>
    %3 = stablehlo.constant dense<1> : tensor<2x2xi32>
    // CHECK: arith.constant dense<1> : tensor<2x2xi64>
    %4 = stablehlo.constant dense<1> : tensor<2x2xi64>
    // CHECK: arith.constant dense<1> : tensor<2x2xui8>
    %5 = stablehlo.constant dense<1> : tensor<2x2xui8>
    // CHECK: arith.constant dense<1> : tensor<2x2xui16>
    %6 = stablehlo.constant dense<1> : tensor<2x2xui16>
    // CHECK: arith.constant dense<1> : tensor<2x2xui32>
    %7 = stablehlo.constant dense<1> : tensor<2x2xui32>
    // CHECK: arith.constant dense<1> : tensor<2x2xui64>
    %8 = stablehlo.constant dense<1> : tensor<2x2xui64>
    // CHECK: arith.constant dense<1.000000e+00> : tensor<2x2xf16>
    %9 = stablehlo.constant dense<1.0> : tensor<2x2xf16>
    // CHECK: arith.constant dense<1.000000e+00> : tensor<2x2xf32>
    %10 = stablehlo.constant dense<1.0> : tensor<2x2xf32>
    // CHECK: arith.constant dense<1.000000e+00> : tensor<2x2xf64>
    %11 = stablehlo.constant dense<1.0> : tensor<2x2xf64>
    func.return
}

// -----

func.func @iota_op_test_si64_dim_0() {
  //CHECK{LITERAL}: arith.constant dense<[[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]]> : tensor<3x4xi64>
  %0 = stablehlo.iota dim = 0 : tensor<3x4xi64>
  func.return
}
// -----


func.func @iota_op_test_si64_dim_1() {
  //CHECK{LITERAL}: arith.constant dense<[[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]> : tensor<3x4xi64>
  %0 = stablehlo.iota dim = 1 : tensor<3x4xi64>
  func.return
}

// -----

func.func @iota_op_test_ui64_dim_0() {
  //CHECK{LITERAL}: arith.constant dense<[[[0, 0], [0, 0], [0, 0]], [[1, 1], [1, 1], [1, 1]]]> : tensor<2x3x2xi64>
  //CHECK: %0 = pphlo.convert %cst : (tensor<2x3x2xi64>) -> tensor<2x3x2xui64>
  %0 = stablehlo.iota dim = 0 : tensor<2x3x2xui64>
  func.return
}

// -----

func.func @iota_op_test_ui64_dim_1() {
  //CHECK{LITERAL}: arith.constant dense<[[[0, 0], [1, 1], [2, 2]], [[0, 0], [1, 1], [2, 2]]]> : tensor<2x3x2xi64>
  //CHECK: %0 = pphlo.convert %cst : (tensor<2x3x2xi64>) -> tensor<2x3x2xui64>
  %0 = stablehlo.iota dim = 1 : tensor<2x3x2xui64>
  func.return
}

// -----

func.func @iota_op_test_ui64_dim_2() {
  //CHECK{LITERAL}: arith.constant dense<[[[0, 1], [0, 1], [0, 1]], [[0, 1], [0, 1], [0, 1]]]> : tensor<2x3x2xi64>
  //CHECK: %0 = pphlo.convert %cst : (tensor<2x3x2xi64>) -> tensor<2x3x2xui64>
  %0 = stablehlo.iota dim = 2 : tensor<2x3x2xui64>
  func.return
}

// -----

func.func @iota_op_test_f64_dim_0() {
  //CHECK{LITERAL}: arith.constant dense<[[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]]> : tensor<3x4xi64>
  //CHECK: %0 = pphlo.convert %cst : (tensor<3x4xi64>) -> tensor<3x4xf64>
  %0 = stablehlo.iota dim = 0 : tensor<3x4xf64>
  func.return
}

// -----

func.func @iota_op_test_f64_dim_1() {
  //CHECK{LITERAL}: arith.constant dense<[[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]> : tensor<3x4xi64>
  //CHECK: %0 = pphlo.convert %cst : (tensor<3x4xi64>) -> tensor<3x4xf64>
  %0 = stablehlo.iota dim = 1 : tensor<3x4xf64>
  func.return
}

