// RUN: spu-translate --interpret -split-input-file %s
// AUTO GENERATED, DO NOT EDIT

func.func @not_op_test_i8_i8_p() {
   %0 = pphlo.constant dense<[127, -128, 0]> : tensor<3xi8>
   %1 = pphlo.not %0 : (tensor<3xi8>)->tensor<3xi8>
   %2 = pphlo.constant dense<[-128, 127, -1]> : tensor<3xi8>
   pphlo.custom_call @expect_eq(%1, %2) : (tensor<3xi8>, tensor<3xi8>)->()
   func.return
}

// -----

func.func @not_op_test_i8_i8_s() {
   %0 = pphlo.constant dense<[127, -128, 0]> : tensor<3xi8>
   %1 = pphlo.convert %0 : (tensor<3xi8>)->tensor<3x!pphlo.secret<i8>>
   %2 = pphlo.not %1 : (tensor<3x!pphlo.secret<i8>>)->tensor<3x!pphlo.secret<i8>>
   %3 = pphlo.constant dense<[-128, 127, -1]> : tensor<3xi8>
   %4 = pphlo.convert %2 : (tensor<3x!pphlo.secret<i8>>)->tensor<3xi8>
   pphlo.custom_call @expect_eq(%3, %4) : (tensor<3xi8>, tensor<3xi8>)->()
   func.return
}

// -----

func.func @not_op_test_ui8_ui8_p() {
   %0 = pphlo.constant dense<[0, 127, 255]> : tensor<3xui8>
   %1 = pphlo.not %0 : (tensor<3xui8>)->tensor<3xui8>
   %2 = pphlo.constant dense<[255, 128, 0]> : tensor<3xui8>
   pphlo.custom_call @expect_eq(%1, %2) : (tensor<3xui8>, tensor<3xui8>)->()
   func.return
}

// -----

func.func @not_op_test_ui8_ui8_s() {
   %0 = pphlo.constant dense<[0, 127, 255]> : tensor<3xui8>
   %1 = pphlo.convert %0 : (tensor<3xui8>)->tensor<3x!pphlo.secret<ui8>>
   %2 = pphlo.not %1 : (tensor<3x!pphlo.secret<ui8>>)->tensor<3x!pphlo.secret<ui8>>
   %3 = pphlo.constant dense<[255, 128, 0]> : tensor<3xui8>
   %4 = pphlo.convert %2 : (tensor<3x!pphlo.secret<ui8>>)->tensor<3xui8>
   pphlo.custom_call @expect_eq(%3, %4) : (tensor<3xui8>, tensor<3xui8>)->()
   func.return
}

// -----

func.func @not_op_test_i16_i16_p() {
   %0 = pphlo.constant dense<[32767, -32768, 0]> : tensor<3xi16>
   %1 = pphlo.not %0 : (tensor<3xi16>)->tensor<3xi16>
   %2 = pphlo.constant dense<[-32768, 32767, -1]> : tensor<3xi16>
   pphlo.custom_call @expect_eq(%1, %2) : (tensor<3xi16>, tensor<3xi16>)->()
   func.return
}

// -----

func.func @not_op_test_i16_i16_s() {
   %0 = pphlo.constant dense<[32767, -32768, 0]> : tensor<3xi16>
   %1 = pphlo.convert %0 : (tensor<3xi16>)->tensor<3x!pphlo.secret<i16>>
   %2 = pphlo.not %1 : (tensor<3x!pphlo.secret<i16>>)->tensor<3x!pphlo.secret<i16>>
   %3 = pphlo.constant dense<[-32768, 32767, -1]> : tensor<3xi16>
   %4 = pphlo.convert %2 : (tensor<3x!pphlo.secret<i16>>)->tensor<3xi16>
   pphlo.custom_call @expect_eq(%3, %4) : (tensor<3xi16>, tensor<3xi16>)->()
   func.return
}

// -----

func.func @not_op_test_ui16_ui16_p() {
   %0 = pphlo.constant dense<[0, 32767, 65535]> : tensor<3xui16>
   %1 = pphlo.not %0 : (tensor<3xui16>)->tensor<3xui16>
   %2 = pphlo.constant dense<[65535, 32768, 0]> : tensor<3xui16>
   pphlo.custom_call @expect_eq(%1, %2) : (tensor<3xui16>, tensor<3xui16>)->()
   func.return
}

// -----

func.func @not_op_test_ui16_ui16_s() {
   %0 = pphlo.constant dense<[0, 32767, 65535]> : tensor<3xui16>
   %1 = pphlo.convert %0 : (tensor<3xui16>)->tensor<3x!pphlo.secret<ui16>>
   %2 = pphlo.not %1 : (tensor<3x!pphlo.secret<ui16>>)->tensor<3x!pphlo.secret<ui16>>
   %3 = pphlo.constant dense<[65535, 32768, 0]> : tensor<3xui16>
   %4 = pphlo.convert %2 : (tensor<3x!pphlo.secret<ui16>>)->tensor<3xui16>
   pphlo.custom_call @expect_eq(%3, %4) : (tensor<3xui16>, tensor<3xui16>)->()
   func.return
}

// -----

func.func @not_op_test_i32_i32_p() {
   %0 = pphlo.constant dense<[2147483647, -2147483648, 0]> : tensor<3xi32>
   %1 = pphlo.not %0 : (tensor<3xi32>)->tensor<3xi32>
   %2 = pphlo.constant dense<[-2147483648, 2147483647, -1]> : tensor<3xi32>
   pphlo.custom_call @expect_eq(%1, %2) : (tensor<3xi32>, tensor<3xi32>)->()
   func.return
}

// -----

func.func @not_op_test_i32_i32_s() {
   %0 = pphlo.constant dense<[2147483647, -2147483648, 0]> : tensor<3xi32>
   %1 = pphlo.convert %0 : (tensor<3xi32>)->tensor<3x!pphlo.secret<i32>>
   %2 = pphlo.not %1 : (tensor<3x!pphlo.secret<i32>>)->tensor<3x!pphlo.secret<i32>>
   %3 = pphlo.constant dense<[-2147483648, 2147483647, -1]> : tensor<3xi32>
   %4 = pphlo.convert %2 : (tensor<3x!pphlo.secret<i32>>)->tensor<3xi32>
   pphlo.custom_call @expect_eq(%3, %4) : (tensor<3xi32>, tensor<3xi32>)->()
   func.return
}

// -----

func.func @not_op_test_ui32_ui32_p() {
   %0 = pphlo.constant dense<[0, 2147483647, 4294967295]> : tensor<3xui32>
   %1 = pphlo.not %0 : (tensor<3xui32>)->tensor<3xui32>
   %2 = pphlo.constant dense<[4294967295, 2147483648, 0]> : tensor<3xui32>
   pphlo.custom_call @expect_eq(%1, %2) : (tensor<3xui32>, tensor<3xui32>)->()
   func.return
}

// -----

func.func @not_op_test_ui32_ui32_s() {
   %0 = pphlo.constant dense<[0, 2147483647, 4294967295]> : tensor<3xui32>
   %1 = pphlo.convert %0 : (tensor<3xui32>)->tensor<3x!pphlo.secret<ui32>>
   %2 = pphlo.not %1 : (tensor<3x!pphlo.secret<ui32>>)->tensor<3x!pphlo.secret<ui32>>
   %3 = pphlo.constant dense<[4294967295, 2147483648, 0]> : tensor<3xui32>
   %4 = pphlo.convert %2 : (tensor<3x!pphlo.secret<ui32>>)->tensor<3xui32>
   pphlo.custom_call @expect_eq(%3, %4) : (tensor<3xui32>, tensor<3xui32>)->()
   func.return
}

// -----

func.func @not_op_test_i64_i64_p() {
   %0 = pphlo.constant dense<[9223372036854775807, -9223372036854775808, 0]> : tensor<3xi64>
   %1 = pphlo.not %0 : (tensor<3xi64>)->tensor<3xi64>
   %2 = pphlo.constant dense<[-9223372036854775808, 9223372036854775807, -1]> : tensor<3xi64>
   pphlo.custom_call @expect_eq(%1, %2) : (tensor<3xi64>, tensor<3xi64>)->()
   func.return
}

// -----

func.func @not_op_test_i64_i64_s() {
   %0 = pphlo.constant dense<[9223372036854775807, -9223372036854775808, 0]> : tensor<3xi64>
   %1 = pphlo.convert %0 : (tensor<3xi64>)->tensor<3x!pphlo.secret<i64>>
   %2 = pphlo.not %1 : (tensor<3x!pphlo.secret<i64>>)->tensor<3x!pphlo.secret<i64>>
   %3 = pphlo.constant dense<[-9223372036854775808, 9223372036854775807, -1]> : tensor<3xi64>
   %4 = pphlo.convert %2 : (tensor<3x!pphlo.secret<i64>>)->tensor<3xi64>
   pphlo.custom_call @expect_eq(%3, %4) : (tensor<3xi64>, tensor<3xi64>)->()
   func.return
}

// -----

func.func @not_op_test_ui64_ui64_p() {
   %0 = pphlo.constant dense<[0, 9223372036854775807, 18446744073709551615]> : tensor<3xui64>
   %1 = pphlo.not %0 : (tensor<3xui64>)->tensor<3xui64>
   %2 = pphlo.constant dense<[18446744073709551615, 9223372036854775808, 0]> : tensor<3xui64>
   pphlo.custom_call @expect_eq(%1, %2) : (tensor<3xui64>, tensor<3xui64>)->()
   func.return
}

// -----

func.func @not_op_test_ui64_ui64_s() {
   %0 = pphlo.constant dense<[0, 9223372036854775807, 18446744073709551615]> : tensor<3xui64>
   %1 = pphlo.convert %0 : (tensor<3xui64>)->tensor<3x!pphlo.secret<ui64>>
   %2 = pphlo.not %1 : (tensor<3x!pphlo.secret<ui64>>)->tensor<3x!pphlo.secret<ui64>>
   %3 = pphlo.constant dense<[18446744073709551615, 9223372036854775808, 0]> : tensor<3xui64>
   %4 = pphlo.convert %2 : (tensor<3x!pphlo.secret<ui64>>)->tensor<3xui64>
   pphlo.custom_call @expect_eq(%3, %4) : (tensor<3xui64>, tensor<3xui64>)->()
   func.return
}

// -----

func.func @not_op_test_i1_i1_p() {
   %0 = pphlo.constant dense<[false, true]> : tensor<2xi1>
   %1 = pphlo.not %0 : (tensor<2xi1>)->tensor<2xi1>
   %2 = pphlo.constant dense<[true, false]> : tensor<2xi1>
   pphlo.custom_call @expect_eq(%1, %2) : (tensor<2xi1>, tensor<2xi1>)->()
   func.return
}

// -----

func.func @not_op_test_i1_i1_s() {
   %0 = pphlo.constant dense<[false, true]> : tensor<2xi1>
   %1 = pphlo.convert %0 : (tensor<2xi1>)->tensor<2x!pphlo.secret<i1>>
   %2 = pphlo.not %1 : (tensor<2x!pphlo.secret<i1>>)->tensor<2x!pphlo.secret<i1>>
   %3 = pphlo.constant dense<[true, false]> : tensor<2xi1>
   %4 = pphlo.convert %2 : (tensor<2x!pphlo.secret<i1>>)->tensor<2xi1>
   pphlo.custom_call @expect_eq(%3, %4) : (tensor<2xi1>, tensor<2xi1>)->()
   func.return
}

// -----

func.func @not_op_test_i1_i1_p() {
   %0 = pphlo.constant dense<false> : tensor<i1>
   %1 = pphlo.not %0 : (tensor<i1>)->tensor<i1>
   %2 = pphlo.constant dense<true> : tensor<i1>
   pphlo.custom_call @expect_eq(%1, %2) : (tensor<i1>, tensor<i1>)->()
   func.return
}

// -----

func.func @not_op_test_i1_i1_s() {
   %0 = pphlo.constant dense<false> : tensor<i1>
   %1 = pphlo.convert %0 : (tensor<i1>)->tensor<!pphlo.secret<i1>>
   %2 = pphlo.not %1 : (tensor<!pphlo.secret<i1>>)->tensor<!pphlo.secret<i1>>
   %3 = pphlo.constant dense<true> : tensor<i1>
   %4 = pphlo.convert %2 : (tensor<!pphlo.secret<i1>>)->tensor<i1>
   pphlo.custom_call @expect_eq(%3, %4) : (tensor<i1>, tensor<i1>)->()
   func.return
}

// -----

func.func @not_op_test_i1_i1_p() {
   %0 = pphlo.constant dense<true> : tensor<i1>
   %1 = pphlo.not %0 : (tensor<i1>)->tensor<i1>
   %2 = pphlo.constant dense<false> : tensor<i1>
   pphlo.custom_call @expect_eq(%1, %2) : (tensor<i1>, tensor<i1>)->()
   func.return
}

// -----

func.func @not_op_test_i1_i1_s() {
   %0 = pphlo.constant dense<true> : tensor<i1>
   %1 = pphlo.convert %0 : (tensor<i1>)->tensor<!pphlo.secret<i1>>
   %2 = pphlo.not %1 : (tensor<!pphlo.secret<i1>>)->tensor<!pphlo.secret<i1>>
   %3 = pphlo.constant dense<false> : tensor<i1>
   %4 = pphlo.convert %2 : (tensor<!pphlo.secret<i1>>)->tensor<i1>
   pphlo.custom_call @expect_eq(%3, %4) : (tensor<i1>, tensor<i1>)->()
   func.return
}
