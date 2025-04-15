// RUN: spu-translate --protocol_kind=1 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=2 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=3 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=4 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=5 --interpret -split-input-file %s
// AUTO GENERATED, DO NOT EDIT

func.func @reshape_op_test_i32_i32_p() {
   %0 = pphlo.constant dense<[[1,2,3,4,5,6]]> : tensor<1x6xi32>
   %1 = pphlo.reshape %0 : (tensor<1x6xi32>)->tensor<6xi32>
   %2 = pphlo.constant dense<[1, 2, 3, 4, 5, 6]> : tensor<6xi32>
   pphlo.custom_call @expect_eq(%1, %2) : (tensor<6xi32>, tensor<6xi32>)->()
   func.return
}

// -----

func.func @reshape_op_test_i32_i32_s() {
   %0 = pphlo.constant dense<[[1,2,3,4,5,6]]> : tensor<1x6xi32>
   %1 = pphlo.convert %0 : (tensor<1x6xi32>)->tensor<1x6x!pphlo.secret<i32>>
   %2 = pphlo.reshape %1 : (tensor<1x6x!pphlo.secret<i32>>)->tensor<6x!pphlo.secret<i32>>
   %3 = pphlo.constant dense<[1, 2, 3, 4, 5, 6]> : tensor<6xi32>
   %4 = pphlo.convert %2 : (tensor<6x!pphlo.secret<i32>>)->tensor<6xi32>
   pphlo.custom_call @expect_eq(%3, %4) : (tensor<6xi32>, tensor<6xi32>)->()
   func.return
}

// -----

func.func @reshape_op_test_i32_i32_p() {
   %0 = pphlo.constant dense<[1,2,3,4,5,6]> : tensor<6xi32>
   %1 = pphlo.reshape %0 : (tensor<6xi32>)->tensor<2x3xi32>
   %2 = pphlo.constant dense<[[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi32>
   pphlo.custom_call @expect_eq(%1, %2) : (tensor<2x3xi32>, tensor<2x3xi32>)->()
   func.return
}

// -----

func.func @reshape_op_test_i32_i32_s() {
   %0 = pphlo.constant dense<[1,2,3,4,5,6]> : tensor<6xi32>
   %1 = pphlo.convert %0 : (tensor<6xi32>)->tensor<6x!pphlo.secret<i32>>
   %2 = pphlo.reshape %1 : (tensor<6x!pphlo.secret<i32>>)->tensor<2x3x!pphlo.secret<i32>>
   %3 = pphlo.constant dense<[[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi32>
   %4 = pphlo.convert %2 : (tensor<2x3x!pphlo.secret<i32>>)->tensor<2x3xi32>
   pphlo.custom_call @expect_eq(%3, %4) : (tensor<2x3xi32>, tensor<2x3xi32>)->()
   func.return
}

// -----

func.func @reshape_op_test_i32_i32_p() {
   %0 = pphlo.constant dense<[[1,2,3],[4,5,6]]> : tensor<2x3xi32>
   %1 = pphlo.reshape %0 : (tensor<2x3xi32>)->tensor<3x2xi32>
   %2 = pphlo.constant dense<[[1, 2], [3, 4], [5, 6]]> : tensor<3x2xi32>
   pphlo.custom_call @expect_eq(%1, %2) : (tensor<3x2xi32>, tensor<3x2xi32>)->()
   func.return
}

// -----

func.func @reshape_op_test_i32_i32_s() {
   %0 = pphlo.constant dense<[[1,2,3],[4,5,6]]> : tensor<2x3xi32>
   %1 = pphlo.convert %0 : (tensor<2x3xi32>)->tensor<2x3x!pphlo.secret<i32>>
   %2 = pphlo.reshape %1 : (tensor<2x3x!pphlo.secret<i32>>)->tensor<3x2x!pphlo.secret<i32>>
   %3 = pphlo.constant dense<[[1, 2], [3, 4], [5, 6]]> : tensor<3x2xi32>
   %4 = pphlo.convert %2 : (tensor<3x2x!pphlo.secret<i32>>)->tensor<3x2xi32>
   pphlo.custom_call @expect_eq(%3, %4) : (tensor<3x2xi32>, tensor<3x2xi32>)->()
   func.return
}

// -----

func.func @reshape_op_test_i32_i32_p() {
   %0 = pphlo.constant dense<[[1,2],[3,4],[5,6]]> : tensor<3x2xi32>
   %1 = pphlo.reshape %0 : (tensor<3x2xi32>)->tensor<6xi32>
   %2 = pphlo.constant dense<[1, 2, 3, 4, 5, 6]> : tensor<6xi32>
   pphlo.custom_call @expect_eq(%1, %2) : (tensor<6xi32>, tensor<6xi32>)->()
   func.return
}

// -----

func.func @reshape_op_test_i32_i32_s() {
   %0 = pphlo.constant dense<[[1,2],[3,4],[5,6]]> : tensor<3x2xi32>
   %1 = pphlo.convert %0 : (tensor<3x2xi32>)->tensor<3x2x!pphlo.secret<i32>>
   %2 = pphlo.reshape %1 : (tensor<3x2x!pphlo.secret<i32>>)->tensor<6x!pphlo.secret<i32>>
   %3 = pphlo.constant dense<[1, 2, 3, 4, 5, 6]> : tensor<6xi32>
   %4 = pphlo.convert %2 : (tensor<6x!pphlo.secret<i32>>)->tensor<6xi32>
   pphlo.custom_call @expect_eq(%3, %4) : (tensor<6xi32>, tensor<6xi32>)->()
   func.return
}
