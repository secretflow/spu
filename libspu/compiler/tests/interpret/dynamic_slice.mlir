// RUN: spu-translate --protocol_kind=1 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=2 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=3 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=4 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=5 --interpret -split-input-file %s

func.func @dynamic_slice() {
  %operand = arith.constant dense<[[1, 1, 1],
                                   [1, 1, 1],
                                   [1, 1, 1]]> : tensor<3x3xi64>
  %start_indices0 = arith.constant dense<3> : tensor<i64>
  %start_indices1 = arith.constant dense<3> : tensor<i64>
  %result = "pphlo.dynamic_slice"(%operand, %start_indices0, %start_indices1) {
    slice_sizes = array<i64: 3, 3>
  } : (tensor<3x3xi64>, tensor<i64>, tensor<i64>) -> tensor<3x3xi64>
  pphlo.custom_call @expect_eq (%result, %operand) : (tensor<3x3xi64>,tensor<3x3xi64>)->()
  func.return
}

// -----

func.func @dynamic_slice() {
  %operand = arith.constant dense<[[1, 2, 3],
                                   [4, 5, 6],
                                   [7, 8, 9]]> : tensor<3x3xi64>
  %i0 = arith.constant dense<1> : tensor<i64>
  %start_indices0 = pphlo.convert %i0 : (tensor<i64>) -> tensor<!pphlo.secret<i64>>
  %start_indices1 = arith.constant dense<1> : tensor<i64>
  %result = "pphlo.dynamic_slice"(%operand, %start_indices0, %start_indices1) {
    slice_sizes = array<i64: 2, 2>
  } : (tensor<3x3xi64>, tensor<!pphlo.secret<i64>>, tensor<i64>) -> tensor<2x2x!pphlo.secret<i64>>
  %expected = arith.constant dense<[[5, 6],
                                    [8, 9]]> : tensor<2x2xi64>
  pphlo.custom_call @expect_eq (%expected, %result) : (tensor<2x2xi64>, tensor<2x2x!pphlo.secret<i64>>)->()
  func.return
}
