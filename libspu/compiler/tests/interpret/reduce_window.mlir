// RUN: spu-translate --protocol_kind=1 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=2 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=3 --interpret -split-input-file %s

func.func @reduce_window() {
  %input = pphlo.constant dense<[[1, 2], [3, 4], [5, 6]]> : tensor<3x2xi64>
  %init_value = pphlo.constant dense<0> : tensor<i64>
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
  %expected = pphlo.constant dense<[[0, 0], [3, 4]]> : tensor<2x2xi64>
  pphlo.custom_call @expect_eq(%result, %expected) : (tensor<2x2xi64>, tensor<2x2xi64>)->()
  func.return
}

// -----

func.func @reduce_window_issue_1662() {
  %input = pphlo.constant dense<[[1, 2], [3, 4], [5, 6]]> : tensor<3x2xi64>
  %init_value = pphlo.constant dense<0> : tensor<i64>
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
  %expected = pphlo.constant dense<[[5, 6]]> : tensor<1x2xi64>
  pphlo.custom_call  @expect_eq(%result, %expected) : (tensor<1x2xi64>, tensor<1x2xi64>)->()
  func.return
}
