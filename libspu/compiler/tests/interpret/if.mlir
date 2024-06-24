// RUN: spu-translate --interpret -split-input-file %s

func.func @if_ops_true_branch() {
  %pred = pphlo.constant dense<true> : tensor<i1>
  %result0, %result1 = "pphlo.if"(%pred) ({
    %0 = pphlo.constant dense<0> : tensor<2xi64>
    pphlo.return %0, %0 : tensor<2xi64>, tensor<2xi64>
  }, {
    %1 = pphlo.constant dense<1> : tensor<2xi64>
    pphlo.return %1, %1 : tensor<2xi64>, tensor<2xi64>
  }) : (tensor<i1>) -> (tensor<2xi64>, tensor<2xi64>)
  %expected = pphlo.constant dense<[0,0]> : tensor<2xi64>
  pphlo.custom_call @expect_eq (%result0, %expected) : (tensor<2xi64>,tensor<2xi64>)->()
  pphlo.custom_call @expect_eq (%result1, %expected) : (tensor<2xi64>,tensor<2xi64>)->()
  func.return
}

// -----

func.func @if_ops_false_branch() {
  %pred = pphlo.constant dense<false> : tensor<i1>
  %result0, %result1 = "pphlo.if"(%pred) ({
    %0 = pphlo.constant dense<0> : tensor<2xi64>
    pphlo.return %0, %0 : tensor<2xi64>, tensor<2xi64>
  }, {
    %1 = pphlo.constant dense<1> : tensor<2xi64>
    pphlo.return %1, %1 : tensor<2xi64>, tensor<2xi64>
  }) : (tensor<i1>) -> (tensor<2xi64>, tensor<2xi64>)
  %expected = pphlo.constant dense<[1, 1]> : tensor<2xi64>
  pphlo.custom_call @expect_eq (%result0, %expected) : (tensor<2xi64>,tensor<2xi64>)->()
  pphlo.custom_call @expect_eq (%result1, %expected) : (tensor<2xi64>,tensor<2xi64>)->()
  func.return
}
