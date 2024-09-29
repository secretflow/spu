// RUN: spu-translate --protocol_kind=1 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=2 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=3 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=4 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=5 --interpret -split-input-file %s

func.func @reduce() {
  %input = arith.constant dense<[[0, 1, 2, 3, 4, 5]]> : tensor<1x6xi64>
  %init_value = arith.constant dense<0> : tensor<i64>
  %result = "pphlo.reduce"(%input, %init_value) ({
    ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
      %0 = pphlo.add %arg0, %arg1 : tensor<i64>
      pphlo.return %0 : tensor<i64>
  }) {
    dimensions = array<i64: 1>
  } : (tensor<1x6xi64>, tensor<i64>) -> tensor<1xi64>
  %expected = arith.constant dense<[15]> : tensor<1xi64>
  pphlo.custom_call @expect_eq (%result, %expected) : (tensor<1xi64>,tensor<1xi64>)->()
  func.return
}

// -----

func.func @reduce_f64() {
  %input = arith.constant dense<[[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]]> : tensor<1x6xf64>
  %init_value = arith.constant dense<0.0> : tensor<f64>
  %result = "pphlo.reduce"(%input, %init_value) ({
    ^bb0(%arg0: tensor<f64>, %arg1: tensor<f64>):
      %0 = pphlo.add %arg0, %arg1 : tensor<f64>
      pphlo.return %0 : tensor<f64>
  }) {
    dimensions = array<i64: 1>
  } : (tensor<1x6xf64>, tensor<f64>) -> tensor<1xf64>
  %expected = arith.constant dense<[15.0]> : tensor<1xf64>
  pphlo.custom_call @expect_eq (%result, %expected) : (tensor<1xf64>,tensor<1xf64>)->()
  func.return
}
