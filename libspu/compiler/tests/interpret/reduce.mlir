// RUN: spu-translate --interpret -split-input-file %s

func.func @reduce() {
  %input = pphlo.constant dense<[[0, 1, 2, 3, 4, 5]]> : tensor<1x6xi64>
  %init_value = pphlo.constant dense<0> : tensor<i64>
  %result = "pphlo.reduce"(%input, %init_value) ({
    ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
      %0 = pphlo.add %arg0, %arg1 : tensor<i64>
      pphlo.return %0 : tensor<i64>
  }) {
    dimensions = array<i64: 1>
  } : (tensor<1x6xi64>, tensor<i64>) -> tensor<1xi64>
  %expected = pphlo.constant dense<[15]> : tensor<1xi64>
  pphlo.custom_call @expect_eq (%result, %expected) : (tensor<1xi64>,tensor<1xi64>)->()
  func.return
}
