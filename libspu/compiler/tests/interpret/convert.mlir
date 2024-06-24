// RUN: spu-translate --interpret -split-input-file %s

func.func @convert_op_test_1() {
  %0 = pphlo.constant dense<[0, 1, 8, -9, 0]> : tensor<5xi32>
  %1 = pphlo.convert %0 : (tensor<5xi32>) -> tensor<5xf32>
  %2 = pphlo.convert %1 : (tensor<5xf32>) -> tensor<5xi32>
  %expected = pphlo.constant dense<[0, 1, 8, -9, 0]> : tensor<5xi32>
  pphlo.custom_call @expect_eq (%2, %expected) : (tensor<5xi32>, tensor<5xi32>) -> ()
  func.return
}

// -----

func.func @convert_op_test_3() {
  %0 = pphlo.constant() {value = dense<[0.0, 1.0, 8.0, -9.0, 10.0]> : tensor<5xf32>} : () -> tensor<5xf32>
  %1 = pphlo.convert %0 : (tensor<5xf32>) -> tensor<5x!pphlo.secret<f32>>
  %2 = pphlo.convert %1 : (tensor<5x!pphlo.secret<f32>>) -> tensor<5xf32>
  %3 = pphlo.convert %2 : (tensor<5xf32>) -> tensor<5xi32>
  %expected = pphlo.constant() {value = dense<[0, 1, 8, -9, 10]> : tensor<5xi32>} : () -> tensor<5xi32>
  pphlo.custom_call @expect_almost_eq (%2, %0) : (tensor<5xf32>, tensor<5xf32>) -> ()
  pphlo.custom_call @expect_eq (%3, %expected) : (tensor<5xi32>, tensor<5xi32>) -> ()
  func.return
}
