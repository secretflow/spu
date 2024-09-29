// RUN: spu-translate --protocol_kind=1 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=2 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=3 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=4 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=5 --interpret -split-input-file %s

func.func @reduce_window() {
  %input = arith.constant dense<[[1, 2], [3, 4], [5, 6]]> : tensor<3x2xi64>
  %init_value = arith.constant dense<0> : tensor<i64>
  %input_pad = pphlo.pad %input, %init_value, low = [2, 0], high = [1, 0], interior = [1, 0] : (tensor<3x2xi64>, tensor<i64>) -> tensor<8x2xi64>
  %result = "pphlo.reduce_window"(%input_pad, %init_value) ({
    ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
      %0 = pphlo.add %arg0, %arg1 : tensor<i64>
      pphlo.return %0 : tensor<i64>
  }) {
    window_dilations = array<i64: 3, 1>,
    window_dimensions = array<i64: 2, 1>,
    window_strides = array<i64: 4, 1>
  } :  (tensor<8x2xi64>, tensor<i64>) -> tensor<2x2xi64>
  %expected = arith.constant dense<[[0, 0], [3, 4]]> : tensor<2x2xi64>
  pphlo.custom_call @expect_eq(%result, %expected) : (tensor<2x2xi64>, tensor<2x2xi64>)->()
  func.return
}

// -----

func.func @reduce_window_f64() {
  %input = arith.constant dense<[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]> : tensor<3x2xf64>
  %init_value = arith.constant dense<0.0> : tensor<f64>
  %input_pad = pphlo.pad %input, %init_value, low = [2, 0], high = [1, 0], interior = [1, 0] : (tensor<3x2xf64>, tensor<f64>) -> tensor<8x2xf64>
  %result = "pphlo.reduce_window"(%input_pad, %init_value) ({
    ^bb0(%arg0: tensor<f64>, %arg1: tensor<f64>):
      %0 = pphlo.add %arg0, %arg1 : tensor<f64>
      pphlo.return %0 : tensor<f64>
  }) {
    window_dilations = array<i64: 3, 1>,
    window_dimensions = array<i64: 2, 1>,
    window_strides = array<i64: 4, 1>
  } :  (tensor<8x2xf64>, tensor<f64>) -> tensor<2x2xf64>
  %expected = arith.constant dense<[[0.0, 0.0], [3.0, 4.0]]> : tensor<2x2xf64>
  pphlo.custom_call @expect_eq(%result, %expected) : (tensor<2x2xf64>, tensor<2x2xf64>)->()
  func.return
}

// -----

func.func @reduce_window_issue_1662() {
  %input = arith.constant dense<[[1, 2], [3, 4], [5, 6]]> : tensor<3x2xi64>
  %init_value = arith.constant dense<0> : tensor<i64>
  %input_pad = pphlo.pad %input, %init_value, low = [2, 0], high = [1, 0], interior = [1, 0] : (tensor<3x2xi64>, tensor<i64>) -> tensor<8x2xi64>
  %result = "pphlo.reduce_window"(%input_pad, %init_value) ({
    ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
      %0 = pphlo.add %arg0, %arg1 : tensor<i64>
      pphlo.return %0 : tensor<i64>
  }) {
    window_dilations = array<i64: 3, 1>,
    window_dimensions = array<i64: 3, 1>,
    window_strides = array<i64: 4, 1>
  } :  (tensor<8x2xi64>, tensor<i64>) -> tensor<1x2xi64>
  %expected = arith.constant dense<[[5, 6]]> : tensor<1x2xi64>
  pphlo.custom_call  @expect_eq(%result, %expected) : (tensor<1x2xi64>, tensor<1x2xi64>)->()
  func.return
}
