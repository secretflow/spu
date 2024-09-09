// RUN: spu-translate --protocol_kind=1 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=2 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=3 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=5 --interpret -split-input-file %s
// AUTO GENERATED, DO NOT EDIT

func.func @xor_op_test_i8_i8_pp() {
   %0 = pphlo.constant dense<[127, -128, -128]> : tensor<3xi8>
   %1 = pphlo.constant dense<[0, 127, -128]> : tensor<3xi8>
   %2 = pphlo.xor %0,%1 : (tensor<3xi8>,tensor<3xi8>)->tensor<3xi8>
   %3 = pphlo.constant dense<[127, -1, 0]> : tensor<3xi8>
   pphlo.custom_call @expect_eq(%2, %3) : (tensor<3xi8>, tensor<3xi8>)->()
   func.return
}

// -----

func.func @xor_op_test_i8_i8_ss() {
   %0 = pphlo.constant dense<[127, -128, -128]> : tensor<3xi8>
   %1 = pphlo.constant dense<[0, 127, -128]> : tensor<3xi8>
   %2 = pphlo.convert %0 : (tensor<3xi8>)->tensor<3x!pphlo.secret<i8>>
   %3 = pphlo.convert %1 : (tensor<3xi8>)->tensor<3x!pphlo.secret<i8>>
   %4 = pphlo.xor %2, %3 : (tensor<3x!pphlo.secret<i8>>,tensor<3x!pphlo.secret<i8>>)->tensor<3x!pphlo.secret<i8>>
   %5 = pphlo.constant dense<[127, -1, 0]> : tensor<3xi8>
   %6 = pphlo.convert %4 : (tensor<3x!pphlo.secret<i8>>)->tensor<3xi8>
   pphlo.custom_call @expect_eq(%5, %6) : (tensor<3xi8>, tensor<3xi8>)->()
   func.return
}

// -----

func.func @xor_op_test_ui8_ui8_pp() {
   %0 = pphlo.constant dense<[0, 127, 255]> : tensor<3xui8>
   %1 = pphlo.constant dense<255> : tensor<3xui8>
   %2 = pphlo.xor %0,%1 : (tensor<3xui8>,tensor<3xui8>)->tensor<3xui8>
   %3 = pphlo.constant dense<[255, 128, 0]> : tensor<3xui8>
   pphlo.custom_call @expect_eq(%2, %3) : (tensor<3xui8>, tensor<3xui8>)->()
   func.return
}

// -----

func.func @xor_op_test_ui8_ui8_ss() {
   %0 = pphlo.constant dense<[0, 127, 255]> : tensor<3xui8>
   %1 = pphlo.constant dense<255> : tensor<3xui8>
   %2 = pphlo.convert %0 : (tensor<3xui8>)->tensor<3x!pphlo.secret<ui8>>
   %3 = pphlo.convert %1 : (tensor<3xui8>)->tensor<3x!pphlo.secret<ui8>>
   %4 = pphlo.xor %2, %3 : (tensor<3x!pphlo.secret<ui8>>,tensor<3x!pphlo.secret<ui8>>)->tensor<3x!pphlo.secret<ui8>>
   %5 = pphlo.constant dense<[255, 128, 0]> : tensor<3xui8>
   %6 = pphlo.convert %4 : (tensor<3x!pphlo.secret<ui8>>)->tensor<3xui8>
   pphlo.custom_call @expect_eq(%5, %6) : (tensor<3xui8>, tensor<3xui8>)->()
   func.return
}

// -----

func.func @xor_op_test_i16_i16_pp() {
   %0 = pphlo.constant dense<[32767, -32768, -32768]> : tensor<3xi16>
   %1 = pphlo.constant dense<[0, 32767, -32768]> : tensor<3xi16>
   %2 = pphlo.xor %0,%1 : (tensor<3xi16>,tensor<3xi16>)->tensor<3xi16>
   %3 = pphlo.constant dense<[32767, -1, 0]> : tensor<3xi16>
   pphlo.custom_call @expect_eq(%2, %3) : (tensor<3xi16>, tensor<3xi16>)->()
   func.return
}

// -----

func.func @xor_op_test_i16_i16_ss() {
   %0 = pphlo.constant dense<[32767, -32768, -32768]> : tensor<3xi16>
   %1 = pphlo.constant dense<[0, 32767, -32768]> : tensor<3xi16>
   %2 = pphlo.convert %0 : (tensor<3xi16>)->tensor<3x!pphlo.secret<i16>>
   %3 = pphlo.convert %1 : (tensor<3xi16>)->tensor<3x!pphlo.secret<i16>>
   %4 = pphlo.xor %2, %3 : (tensor<3x!pphlo.secret<i16>>,tensor<3x!pphlo.secret<i16>>)->tensor<3x!pphlo.secret<i16>>
   %5 = pphlo.constant dense<[32767, -1, 0]> : tensor<3xi16>
   %6 = pphlo.convert %4 : (tensor<3x!pphlo.secret<i16>>)->tensor<3xi16>
   pphlo.custom_call @expect_eq(%5, %6) : (tensor<3xi16>, tensor<3xi16>)->()
   func.return
}

// -----

func.func @xor_op_test_ui16_ui16_pp() {
   %0 = pphlo.constant dense<[0, 32767, 65535]> : tensor<3xui16>
   %1 = pphlo.constant dense<65535> : tensor<3xui16>
   %2 = pphlo.xor %0,%1 : (tensor<3xui16>,tensor<3xui16>)->tensor<3xui16>
   %3 = pphlo.constant dense<[65535, 32768, 0]> : tensor<3xui16>
   pphlo.custom_call @expect_eq(%2, %3) : (tensor<3xui16>, tensor<3xui16>)->()
   func.return
}

// -----

func.func @xor_op_test_ui16_ui16_ss() {
   %0 = pphlo.constant dense<[0, 32767, 65535]> : tensor<3xui16>
   %1 = pphlo.constant dense<65535> : tensor<3xui16>
   %2 = pphlo.convert %0 : (tensor<3xui16>)->tensor<3x!pphlo.secret<ui16>>
   %3 = pphlo.convert %1 : (tensor<3xui16>)->tensor<3x!pphlo.secret<ui16>>
   %4 = pphlo.xor %2, %3 : (tensor<3x!pphlo.secret<ui16>>,tensor<3x!pphlo.secret<ui16>>)->tensor<3x!pphlo.secret<ui16>>
   %5 = pphlo.constant dense<[65535, 32768, 0]> : tensor<3xui16>
   %6 = pphlo.convert %4 : (tensor<3x!pphlo.secret<ui16>>)->tensor<3xui16>
   pphlo.custom_call @expect_eq(%5, %6) : (tensor<3xui16>, tensor<3xui16>)->()
   func.return
}

// -----

func.func @xor_op_test_i32_i32_pp() {
   %0 = pphlo.constant dense<[2147483647, -2147483648, -2147483648]> : tensor<3xi32>
   %1 = pphlo.constant dense<[0, 2147483647, -2147483648]> : tensor<3xi32>
   %2 = pphlo.xor %0,%1 : (tensor<3xi32>,tensor<3xi32>)->tensor<3xi32>
   %3 = pphlo.constant dense<[2147483647, -1, 0]> : tensor<3xi32>
   pphlo.custom_call @expect_eq(%2, %3) : (tensor<3xi32>, tensor<3xi32>)->()
   func.return
}

// -----

func.func @xor_op_test_i32_i32_ss() {
   %0 = pphlo.constant dense<[2147483647, -2147483648, -2147483648]> : tensor<3xi32>
   %1 = pphlo.constant dense<[0, 2147483647, -2147483648]> : tensor<3xi32>
   %2 = pphlo.convert %0 : (tensor<3xi32>)->tensor<3x!pphlo.secret<i32>>
   %3 = pphlo.convert %1 : (tensor<3xi32>)->tensor<3x!pphlo.secret<i32>>
   %4 = pphlo.xor %2, %3 : (tensor<3x!pphlo.secret<i32>>,tensor<3x!pphlo.secret<i32>>)->tensor<3x!pphlo.secret<i32>>
   %5 = pphlo.constant dense<[2147483647, -1, 0]> : tensor<3xi32>
   %6 = pphlo.convert %4 : (tensor<3x!pphlo.secret<i32>>)->tensor<3xi32>
   pphlo.custom_call @expect_eq(%5, %6) : (tensor<3xi32>, tensor<3xi32>)->()
   func.return
}

// -----

func.func @xor_op_test_ui32_ui32_pp() {
   %0 = pphlo.constant dense<[0, 2147483647, 4294967295]> : tensor<3xui32>
   %1 = pphlo.constant dense<4294967295> : tensor<3xui32>
   %2 = pphlo.xor %0,%1 : (tensor<3xui32>,tensor<3xui32>)->tensor<3xui32>
   %3 = pphlo.constant dense<[4294967295, 2147483648, 0]> : tensor<3xui32>
   pphlo.custom_call @expect_eq(%2, %3) : (tensor<3xui32>, tensor<3xui32>)->()
   func.return
}

// -----

func.func @xor_op_test_ui32_ui32_ss() {
   %0 = pphlo.constant dense<[0, 2147483647, 4294967295]> : tensor<3xui32>
   %1 = pphlo.constant dense<4294967295> : tensor<3xui32>
   %2 = pphlo.convert %0 : (tensor<3xui32>)->tensor<3x!pphlo.secret<ui32>>
   %3 = pphlo.convert %1 : (tensor<3xui32>)->tensor<3x!pphlo.secret<ui32>>
   %4 = pphlo.xor %2, %3 : (tensor<3x!pphlo.secret<ui32>>,tensor<3x!pphlo.secret<ui32>>)->tensor<3x!pphlo.secret<ui32>>
   %5 = pphlo.constant dense<[4294967295, 2147483648, 0]> : tensor<3xui32>
   %6 = pphlo.convert %4 : (tensor<3x!pphlo.secret<ui32>>)->tensor<3xui32>
   pphlo.custom_call @expect_eq(%5, %6) : (tensor<3xui32>, tensor<3xui32>)->()
   func.return
}

// -----

func.func @xor_op_test_i64_i64_pp() {
   %0 = pphlo.constant dense<[9223372036854775807, -9223372036854775808, -9223372036854775808]> : tensor<3xi64>
   %1 = pphlo.constant dense<[0, 9223372036854775807, -9223372036854775808]> : tensor<3xi64>
   %2 = pphlo.xor %0,%1 : (tensor<3xi64>,tensor<3xi64>)->tensor<3xi64>
   %3 = pphlo.constant dense<[9223372036854775807, -1, 0]> : tensor<3xi64>
   pphlo.custom_call @expect_eq(%2, %3) : (tensor<3xi64>, tensor<3xi64>)->()
   func.return
}

// -----

func.func @xor_op_test_i64_i64_ss() {
   %0 = pphlo.constant dense<[9223372036854775807, -9223372036854775808, -9223372036854775808]> : tensor<3xi64>
   %1 = pphlo.constant dense<[0, 9223372036854775807, -9223372036854775808]> : tensor<3xi64>
   %2 = pphlo.convert %0 : (tensor<3xi64>)->tensor<3x!pphlo.secret<i64>>
   %3 = pphlo.convert %1 : (tensor<3xi64>)->tensor<3x!pphlo.secret<i64>>
   %4 = pphlo.xor %2, %3 : (tensor<3x!pphlo.secret<i64>>,tensor<3x!pphlo.secret<i64>>)->tensor<3x!pphlo.secret<i64>>
   %5 = pphlo.constant dense<[9223372036854775807, -1, 0]> : tensor<3xi64>
   %6 = pphlo.convert %4 : (tensor<3x!pphlo.secret<i64>>)->tensor<3xi64>
   pphlo.custom_call @expect_eq(%5, %6) : (tensor<3xi64>, tensor<3xi64>)->()
   func.return
}

// -----

func.func @xor_op_test_ui64_ui64_pp() {
   %0 = pphlo.constant dense<[0, 9223372036854775807, 18446744073709551615]> : tensor<3xui64>
   %1 = pphlo.constant dense<18446744073709551615> : tensor<3xui64>
   %2 = pphlo.xor %0,%1 : (tensor<3xui64>,tensor<3xui64>)->tensor<3xui64>
   %3 = pphlo.constant dense<[18446744073709551615, 9223372036854775808, 0]> : tensor<3xui64>
   pphlo.custom_call @expect_eq(%2, %3) : (tensor<3xui64>, tensor<3xui64>)->()
   func.return
}

// -----

func.func @xor_op_test_ui64_ui64_ss() {
   %0 = pphlo.constant dense<[0, 9223372036854775807, 18446744073709551615]> : tensor<3xui64>
   %1 = pphlo.constant dense<18446744073709551615> : tensor<3xui64>
   %2 = pphlo.convert %0 : (tensor<3xui64>)->tensor<3x!pphlo.secret<ui64>>
   %3 = pphlo.convert %1 : (tensor<3xui64>)->tensor<3x!pphlo.secret<ui64>>
   %4 = pphlo.xor %2, %3 : (tensor<3x!pphlo.secret<ui64>>,tensor<3x!pphlo.secret<ui64>>)->tensor<3x!pphlo.secret<ui64>>
   %5 = pphlo.constant dense<[18446744073709551615, 9223372036854775808, 0]> : tensor<3xui64>
   %6 = pphlo.convert %4 : (tensor<3x!pphlo.secret<ui64>>)->tensor<3xui64>
   pphlo.custom_call @expect_eq(%5, %6) : (tensor<3xui64>, tensor<3xui64>)->()
   func.return
}

// -----

func.func @xor_op_test_i1_i1_pp() {
   %0 = pphlo.constant dense<[false, false, true, true]> : tensor<4xi1>
   %1 = pphlo.constant dense<[false, true, false, true]> : tensor<4xi1>
   %2 = pphlo.xor %0,%1 : (tensor<4xi1>,tensor<4xi1>)->tensor<4xi1>
   %3 = pphlo.constant dense<[false, true, true, false]> : tensor<4xi1>
   pphlo.custom_call @expect_eq(%2, %3) : (tensor<4xi1>, tensor<4xi1>)->()
   func.return
}

// -----

func.func @xor_op_test_i1_i1_ss() {
   %0 = pphlo.constant dense<[false, false, true, true]> : tensor<4xi1>
   %1 = pphlo.constant dense<[false, true, false, true]> : tensor<4xi1>
   %2 = pphlo.convert %0 : (tensor<4xi1>)->tensor<4x!pphlo.secret<i1>>
   %3 = pphlo.convert %1 : (tensor<4xi1>)->tensor<4x!pphlo.secret<i1>>
   %4 = pphlo.xor %2, %3 : (tensor<4x!pphlo.secret<i1>>,tensor<4x!pphlo.secret<i1>>)->tensor<4x!pphlo.secret<i1>>
   %5 = pphlo.constant dense<[false, true, true, false]> : tensor<4xi1>
   %6 = pphlo.convert %4 : (tensor<4x!pphlo.secret<i1>>)->tensor<4xi1>
   pphlo.custom_call @expect_eq(%5, %6) : (tensor<4xi1>, tensor<4xi1>)->()
   func.return
}

// -----

func.func @xor_op_test_i1_i1_pp() {
   %0 = pphlo.constant dense<false> : tensor<2xi1>
   %1 = pphlo.constant dense<[false, true]> : tensor<2xi1>
   %2 = pphlo.xor %0,%1 : (tensor<2xi1>,tensor<2xi1>)->tensor<2xi1>
   %3 = pphlo.constant dense<[false, true]> : tensor<2xi1>
   pphlo.custom_call @expect_eq(%2, %3) : (tensor<2xi1>, tensor<2xi1>)->()
   func.return
}

// -----

func.func @xor_op_test_i1_i1_ss() {
   %0 = pphlo.constant dense<false> : tensor<2xi1>
   %1 = pphlo.constant dense<[false, true]> : tensor<2xi1>
   %2 = pphlo.convert %0 : (tensor<2xi1>)->tensor<2x!pphlo.secret<i1>>
   %3 = pphlo.convert %1 : (tensor<2xi1>)->tensor<2x!pphlo.secret<i1>>
   %4 = pphlo.xor %2, %3 : (tensor<2x!pphlo.secret<i1>>,tensor<2x!pphlo.secret<i1>>)->tensor<2x!pphlo.secret<i1>>
   %5 = pphlo.constant dense<[false, true]> : tensor<2xi1>
   %6 = pphlo.convert %4 : (tensor<2x!pphlo.secret<i1>>)->tensor<2xi1>
   pphlo.custom_call @expect_eq(%5, %6) : (tensor<2xi1>, tensor<2xi1>)->()
   func.return
}

// -----

func.func @xor_op_test_i1_i1_pp() {
   %0 = pphlo.constant dense<false> : tensor<2xi1>
   %1 = pphlo.constant dense<[false, true]> : tensor<2xi1>
   %2 = pphlo.xor %0,%1 : (tensor<2xi1>,tensor<2xi1>)->tensor<2xi1>
   %3 = pphlo.constant dense<[false, true]> : tensor<2xi1>
   pphlo.custom_call @expect_eq(%2, %3) : (tensor<2xi1>, tensor<2xi1>)->()
   func.return
}

// -----

func.func @xor_op_test_i1_i1_ss() {
   %0 = pphlo.constant dense<false> : tensor<2xi1>
   %1 = pphlo.constant dense<[false, true]> : tensor<2xi1>
   %2 = pphlo.convert %0 : (tensor<2xi1>)->tensor<2x!pphlo.secret<i1>>
   %3 = pphlo.convert %1 : (tensor<2xi1>)->tensor<2x!pphlo.secret<i1>>
   %4 = pphlo.xor %2, %3 : (tensor<2x!pphlo.secret<i1>>,tensor<2x!pphlo.secret<i1>>)->tensor<2x!pphlo.secret<i1>>
   %5 = pphlo.constant dense<[false, true]> : tensor<2xi1>
   %6 = pphlo.convert %4 : (tensor<2x!pphlo.secret<i1>>)->tensor<2xi1>
   pphlo.custom_call @expect_eq(%5, %6) : (tensor<2xi1>, tensor<2xi1>)->()
   func.return
}
