// RUN: spu-translate --protocol_kind=1 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=2 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=3 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=4 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=5 --interpret -split-input-file %s

func.func @concatenate() {
  %input0 = pphlo.constant dense<[[1, 2], [3, 4], [5, 6]]> : tensor<3x2xi64>
  %input1 = pphlo.constant dense<[[7, 8]]> : tensor<1x2xi64>
  %result = pphlo.concatenate %input0, %input1 dim = 0 : (tensor<3x2xi64>, tensor<1x2xi64>) -> tensor<4x2xi64>
  %expected = pphlo.constant dense<[[1, 2], [3, 4] , [5, 6], [7, 8]]> : tensor<4x2xi64>
  pphlo.custom_call @expect_eq (%result, %expected) : (tensor<4x2xi64>,tensor<4x2xi64>)->()
  func.return
}