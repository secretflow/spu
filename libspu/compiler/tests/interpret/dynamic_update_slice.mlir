// RUN: spu-translate --protocol_kind=1 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=2 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=3 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=4 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=5 --interpret -split-input-file %s

func.func @dynamic_update_slice() {
  %operand = pphlo.constant dense<[[1, 1, 1, 1],
                                   [1, 1, 1, 1],
                                   [1, 2, 2, 2],
                                   [1, 2, 2, 2]]> : tensor<4x4xi64>
  %update = pphlo.constant dense<[[1, 1, 1],
                                  [1, 1, 1]]> : tensor<2x3xi64>
  %start_indices0 = pphlo.constant dense<4> : tensor<i64>
  %start_indices1 = pphlo.constant dense<4> : tensor<i64>
  %result = pphlo.dynamic_update_slice %operand, %update, %start_indices0, %start_indices1 :
      (tensor<4x4xi64>, tensor<2x3xi64>, tensor<i64>, tensor<i64>) -> tensor<4x4xi64>
  %expected = pphlo.constant dense<[[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]> : tensor<4x4xi64>
  pphlo.custom_call @expect_eq (%result, %expected) : (tensor<4x4xi64>,tensor<4x4xi64>)->()
  func.return
}
