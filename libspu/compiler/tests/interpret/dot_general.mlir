// RUN: spu-translate --protocol_kind=1 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=2 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=3 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=4 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=5 --interpret -split-input-file %s

func.func @dot_general_op_test_si64() {
  %lhs = arith.constant dense<[[[1, 2], [3, 4]],
                               [[5, 6], [7, 8]]]> : tensor<2x2x2xi64>
  %rhs = arith.constant dense<[[[1, 0], [0, 1]],
                               [[1, 0], [0, 1]]]> : tensor<2x2x2xi64>
  %result = pphlo.dot_general %lhs, %rhs,
    batching_dims = [0] x [0],
    contracting_dims = [2] x [1]
    : (tensor<2x2x2xi64>, tensor<2x2x2xi64>) -> tensor<2x2x2xi64>
  %expected = arith.constant dense<[[[1, 2], [3, 4]],
                                    [[5, 6], [7, 8]]]> : tensor<2x2x2xi64>
  pphlo.custom_call @expect_eq (%result, %expected) : (tensor<2x2x2xi64>, tensor<2x2x2xi64>)->()
  func.return
}

// -----

func.func @dot_general_op_test_empty_dims() {
  %lhs = arith.constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi64>
  %rhs = arith.constant dense<[[1, 0], [0, 1]]> : tensor<2x2xi64>
  %result = pphlo.dot_general %lhs, %rhs,
    batching_dims = [] x [],
    contracting_dims = [] x []
    : (tensor<2x2xi64>, tensor<2x2xi64>) -> tensor<2x2x2x2xi64>
  %expected = arith.constant dense<[[[[1, 0], [0, 1]],
                                     [[2, 0], [0, 2]]],
                                    [[[3, 0], [0, 3]],
                                     [[4, 0], [0, 4]]]]> : tensor<2x2x2x2xi64>
  pphlo.custom_call @expect_eq (%result, %expected) : (tensor<2x2x2x2xi64>,tensor<2x2x2x2xi64>)->()
  func.return
}

// -----

func.func @main() {
  %0 = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]> : tensor<12xi32>
  %1 = pphlo.reshape %0 : (tensor<12xi32>) -> tensor<3x1x4xi32>
  %2 = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                              16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                              31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
                              46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]> : tensor<60xi32>
  %3 = pphlo.reshape %2 : (tensor<60xi32>) -> tensor<3x4x5xi32>
  %4 = pphlo.dot_general %1, %3,
        batching_dims = [0] x [0],
        contracting_dims = [2] x [1]
        : (tensor<3x1x4xi32>, tensor<3x4x5xi32>) -> tensor<3x5xi32>
  %5 = arith.constant dense<[[ 70,  76,  82,  88,  94],
                             [630, 652, 674, 696, 718],
                             [1830, 1868, 1906, 1944, 1982]]> : tensor<3x5xi32>
  pphlo.custom_call @expect_eq (%4, %5) : (tensor<3x5xi32>, tensor<3x5xi32>)->()
  return
}
