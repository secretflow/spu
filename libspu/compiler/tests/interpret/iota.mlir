// RUN: spu-translate --protocol_kind=1 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=2 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=3 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=4 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=5 --interpret -split-input-file %s

func.func @iota_op_test_si8_dim_0() {
  %0 = pphlo.iota dim = 0 : tensor<3x4xi8>
  %expected = pphlo.constant dense<[[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]]> : tensor<3x4xi8>
  pphlo.custom_call @expect_eq (%0, %expected) : (tensor<3x4xi8>,tensor<3x4xi8>)->()
  func.return
}

// -----

func.func @iota_op_test_si8_dim_1() {
  %0 = pphlo.iota dim = 1 : tensor<3x4xi8>
  %expected = pphlo.constant dense<[[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]> : tensor<3x4xi8>
  pphlo.custom_call @expect_eq (%0, %expected) : (tensor<3x4xi8>,tensor<3x4xi8>)->()
  func.return
}

// -----

func.func @iota_op_test_si16_dim_0() {
  %0 = pphlo.iota dim = 0 : tensor<3x4xi16>
  %expected = pphlo.constant dense<[[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]]> : tensor<3x4xi16>
  pphlo.custom_call @expect_eq (%0, %expected) : (tensor<3x4xi16>,tensor<3x4xi16>)->()
  func.return
}

// -----

func.func @iota_op_test_si16_dim_1() {
  %0 = pphlo.iota dim = 1 : tensor<3x4xi16>
  %expected = pphlo.constant dense<[[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]> : tensor<3x4xi16>
  pphlo.custom_call @expect_eq (%0, %expected) : (tensor<3x4xi16>,tensor<3x4xi16>)->()
  func.return
}

// -----

func.func @iota_op_test_si32_dim_0() {
  %0 = pphlo.iota dim = 0 : tensor<3x4xi32>
  %expected = pphlo.constant dense<[[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]]> : tensor<3x4xi32>
  pphlo.custom_call @expect_eq (%0, %expected) : (tensor<3x4xi32>,tensor<3x4xi32>)->()
  func.return
}

// -----

func.func @iota_op_test_si32_dim_1() {
  %0 = pphlo.iota dim = 1 : tensor<3x4xi32>
  %expected = pphlo.constant dense<[[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]> : tensor<3x4xi32>
  pphlo.custom_call @expect_eq (%0, %expected) : (tensor<3x4xi32>,tensor<3x4xi32>)->()
  func.return
}

// -----

func.func @iota_op_test_si64_dim_0() {
  %0 = pphlo.iota dim = 0 : tensor<3x4xi64>
  %expected = pphlo.constant dense<[[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]]> : tensor<3x4xi64>
  pphlo.custom_call @expect_eq (%0, %expected) : (tensor<3x4xi64>,tensor<3x4xi64>)->()
  func.return
}
// -----


func.func @iota_op_test_si64_dim_1() {
  %0 = pphlo.iota dim = 1 : tensor<3x4xi64>
  %expected = pphlo.constant dense<[[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]> : tensor<3x4xi64>
  pphlo.custom_call @expect_eq (%0, %expected) : (tensor<3x4xi64>,tensor<3x4xi64>)->()
  func.return
}

// -----

func.func @iota_op_test_ui64_dim_0() {
  %0 = pphlo.iota dim = 0 : tensor<2x3x2xui64>
  %expected = pphlo.constant dense<[[[0, 0], [0, 0], [0, 0]], [[1, 1], [1, 1], [1, 1]]]> : tensor<2x3x2xui64>
  pphlo.custom_call @expect_eq (%0, %expected) : (tensor<2x3x2xui64>,tensor<2x3x2xui64>)->()
  func.return
}

// -----

func.func @iota_op_test_ui64_dim_1() {
  %0 = pphlo.iota dim = 1 : tensor<2x3x2xui64>
  %expected = pphlo.constant dense<[[[0, 0], [1, 1], [2, 2]], [[0, 0], [1, 1], [2, 2]]]> : tensor<2x3x2xui64>
  pphlo.custom_call @expect_eq (%0, %expected) : (tensor<2x3x2xui64>,tensor<2x3x2xui64>)->()
  func.return
}

// -----

func.func @iota_op_test_ui64_dim_2() {
  %0 = pphlo.iota dim = 2 : tensor<2x3x2xui64>
  %expected = pphlo.constant dense<[[[0, 1], [0, 1], [0, 1]], [[0, 1], [0, 1], [0, 1]]]> : tensor<2x3x2xui64>
  pphlo.custom_call @expect_eq (%0, %expected) : (tensor<2x3x2xui64>,tensor<2x3x2xui64>)->()
  func.return
}

// -----

func.func @iota_op_test_f16_dim_0() {
  %0 = pphlo.iota dim = 0 : tensor<3x4xf16>
  %expected = pphlo.constant dense<[[0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00], [2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00]]> : tensor<3x4xf16>
  pphlo.custom_call @expect_almost_eq (%0, %expected) : (tensor<3x4xf16>,tensor<3x4xf16>)->()
  func.return
}

// -----

func.func @iota_op_test_f16_dim_1() {
  %0 = pphlo.iota dim = 1 : tensor<3x4xf16>
  %expected = pphlo.constant dense<[[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00], [0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00], [0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00]]> : tensor<3x4xf16>
  pphlo.custom_call @expect_almost_eq (%0, %expected) : (tensor<3x4xf16>,tensor<3x4xf16>)->()
  func.return
}

// -----

func.func @iota_op_test_f32_dim_0() {
  %0 = pphlo.iota dim = 0 : tensor<3x4xf32>
  %expected = pphlo.constant dense<[[0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00], [2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00]]> : tensor<3x4xf32>
  pphlo.custom_call @expect_almost_eq (%0, %expected) : (tensor<3x4xf32>,tensor<3x4xf32>)->()
  func.return
}

// -----

func.func @iota_op_test_f32_dim_1() {
  %0 = pphlo.iota dim = 1 : tensor<3x4xf32>
  %expected = pphlo.constant dense<[[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00], [0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00], [0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00]]> : tensor<3x4xf32>
  pphlo.custom_call @expect_almost_eq (%0, %expected) : (tensor<3x4xf32>,tensor<3x4xf32>)->()
  func.return
}

// -----

func.func @iota_op_test_f64_dim_0() {
  %0 = pphlo.iota dim = 0 : tensor<3x4xf64>
  %expected = pphlo.constant dense<[[0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00], [2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00]]> : tensor<3x4xf64>
  pphlo.custom_call @expect_almost_eq (%0, %expected) : (tensor<3x4xf64>,tensor<3x4xf64>)->()
  func.return
}

// -----

func.func @iota_op_test_f64_dim_1() {
  %0 = pphlo.iota dim = 1 : tensor<3x4xf64>
  %expected = pphlo.constant dense<[[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00], [0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00], [0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00]]> : tensor<3x4xf64>
  pphlo.custom_call @expect_almost_eq (%0, %expected) : (tensor<3x4xf64>,tensor<3x4xf64>)->()
  func.return
}
