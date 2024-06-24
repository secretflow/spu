// RUN: spu-translate --interpret -split-input-file %s

func.func @sort_stable() {
  %input0 = pphlo.constant dense<[[1, 2, 3], [3, 2, 1]]> : tensor<2x3xi64>
  %input1 = pphlo.constant dense<[[3, 2, 1], [1, 2, 3]]> : tensor<2x3xi64>
  %result0, %result1 = "pphlo.sort"(%input0, %input1) ({
    ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>, %arg2: tensor<i64>, %arg3: tensor<i64>):
      %predicate = pphlo.greater %arg0, %arg1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      pphlo.return %predicate : tensor<i1>
  }) {
    dimension = 0 : i64,
    is_stable = true
  } : (tensor<2x3xi64>, tensor<2x3xi64>) -> (tensor<2x3xi64>, tensor<2x3xi64>)
  %expected0 = pphlo.constant dense<[[3, 2, 3], [1, 2, 1]]> : tensor<2x3xi64>
  %expected1 = pphlo.constant dense<[[1, 2, 1], [3, 2, 3]]> : tensor<2x3xi64>
  pphlo.custom_call @expect_eq (%result0, %expected0) : (tensor<2x3xi64>,tensor<2x3xi64>)->()
  pphlo.custom_call @expect_eq (%result1, %expected1) : (tensor<2x3xi64>,tensor<2x3xi64>)->()
  func.return
}
