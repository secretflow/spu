// RUN: spu-translate --protocol_kind=1 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=2 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=3 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=4 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=5 --interpret -split-input-file %s

func.func @if_ops_true_branch() {
  %pred = arith.constant dense<true> : tensor<i1>
  %result0, %result1 = "pphlo.if"(%pred) ({
    %0 = arith.constant dense<0> : tensor<2xi64>
    %2 = pphlo.add %0, %0 : tensor<2xi64>
    %3 = pphlo.add %0, %2 : tensor<2xi64>
    pphlo.return %3, %3 : tensor<2xi64>, tensor<2xi64>
  }, {
    %1 = arith.constant dense<1> : tensor<2xi64>
    pphlo.return %1, %1 : tensor<2xi64>, tensor<2xi64>
  }) : (tensor<i1>) -> (tensor<2xi64>, tensor<2xi64>)
  %expected = arith.constant dense<[0,0]> : tensor<2xi64>
  pphlo.custom_call @expect_eq (%result0, %expected) : (tensor<2xi64>,tensor<2xi64>)->()
  pphlo.custom_call @expect_eq (%result1, %expected) : (tensor<2xi64>,tensor<2xi64>)->()
  func.return
}

// -----

func.func @if_ops_false_branch() {
  %pred = arith.constant dense<false> : tensor<i1>
  %result0, %result1 = "pphlo.if"(%pred) ({
    %0 = arith.constant dense<0> : tensor<2xi64>
    pphlo.return %0, %0 : tensor<2xi64>, tensor<2xi64>
  }, {
    %1 = arith.constant dense<1> : tensor<2xi64>
    %2 = pphlo.add %1, %1 : tensor<2xi64>
    pphlo.return %2, %2 : tensor<2xi64>, tensor<2xi64>
  }) : (tensor<i1>) -> (tensor<2xi64>, tensor<2xi64>)
  %expected = arith.constant dense<[2, 2]> : tensor<2xi64>
  pphlo.custom_call @expect_eq (%result0, %expected) : (tensor<2xi64>,tensor<2xi64>)->()
  pphlo.custom_call @expect_eq (%result1, %expected) : (tensor<2xi64>,tensor<2xi64>)->()
  func.return
}
