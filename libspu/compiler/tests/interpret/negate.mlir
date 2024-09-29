// RUN: spu-translate --protocol_kind=1 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=2 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=3 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=4 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=5 --interpret -split-input-file %s
// AUTO GENERATED, DO NOT EDIT

func.func @negate_op_test_i8_i8_p() {
   %0 = arith.constant dense<[-128, -9, 0, 8, 127]> : tensor<5xi8>
   %1 = pphlo.negate %0 : (tensor<5xi8>)->tensor<5xi8>
   %2 = arith.constant dense<[-128, 9, 0, -8, -127]> : tensor<5xi8>
   pphlo.custom_call @expect_eq(%1, %2) : (tensor<5xi8>, tensor<5xi8>)->()
   func.return
}

// -----

func.func @negate_op_test_i8_i8_s() {
   %0 = arith.constant dense<[-128, -9, 0, 8, 127]> : tensor<5xi8>
   %1 = pphlo.convert %0 : (tensor<5xi8>)->tensor<5x!pphlo.secret<i8>>
   %2 = pphlo.negate %1 : (tensor<5x!pphlo.secret<i8>>)->tensor<5x!pphlo.secret<i8>>
   %3 = arith.constant dense<[-128, 9, 0, -8, -127]> : tensor<5xi8>
   pphlo.custom_call @expect_eq(%3, %2) : (tensor<5xi8>, tensor<5x!pphlo.secret<i8>>)->()
   func.return
}

// -----

func.func @negate_op_test_ui8_ui8_p() {
   %0 = arith.constant dense<[0, 16, 255]> : tensor<3xui8>
   %1 = pphlo.negate %0 : (tensor<3xui8>)->tensor<3xui8>
   %2 = arith.constant dense<[0, 240, 1]> : tensor<3xui8>
   pphlo.custom_call @expect_eq(%1, %2) : (tensor<3xui8>, tensor<3xui8>)->()
   func.return
}

// -----

func.func @negate_op_test_ui8_ui8_s() {
   %0 = arith.constant dense<[0, 16, 255]> : tensor<3xui8>
   %1 = pphlo.convert %0 : (tensor<3xui8>)->tensor<3x!pphlo.secret<ui8>>
   %2 = pphlo.negate %1 : (tensor<3x!pphlo.secret<ui8>>)->tensor<3x!pphlo.secret<ui8>>
   %3 = arith.constant dense<[0, 240, 1]> : tensor<3xui8>
   pphlo.custom_call @expect_eq(%3, %2) : (tensor<3xui8>, tensor<3x!pphlo.secret<ui8>>)->()
   func.return
}

// -----

func.func @negate_op_test_i16_i16_p() {
   %0 = arith.constant dense<[-32768, -129, 0, 128, 32767]> : tensor<5xi16>
   %1 = pphlo.negate %0 : (tensor<5xi16>)->tensor<5xi16>
   %2 = arith.constant dense<[-32768, 129, 0, -128, -32767]> : tensor<5xi16>
   pphlo.custom_call @expect_eq(%1, %2) : (tensor<5xi16>, tensor<5xi16>)->()
   func.return
}

// -----

func.func @negate_op_test_i16_i16_s() {
   %0 = arith.constant dense<[-32768, -129, 0, 128, 32767]> : tensor<5xi16>
   %1 = pphlo.convert %0 : (tensor<5xi16>)->tensor<5x!pphlo.secret<i16>>
   %2 = pphlo.negate %1 : (tensor<5x!pphlo.secret<i16>>)->tensor<5x!pphlo.secret<i16>>
   %3 = arith.constant dense<[-32768, 129, 0, -128, -32767]> : tensor<5xi16>
   pphlo.custom_call @expect_eq(%3, %2) : (tensor<5xi16>, tensor<5x!pphlo.secret<i16>>)->()
   func.return
}

// -----

func.func @negate_op_test_ui16_ui16_p() {
   %0 = arith.constant dense<[0, 256, 65535]> : tensor<3xui16>
   %1 = pphlo.negate %0 : (tensor<3xui16>)->tensor<3xui16>
   %2 = arith.constant dense<[0, 65280, 1]> : tensor<3xui16>
   pphlo.custom_call @expect_eq(%1, %2) : (tensor<3xui16>, tensor<3xui16>)->()
   func.return
}

// -----

func.func @negate_op_test_ui16_ui16_s() {
   %0 = arith.constant dense<[0, 256, 65535]> : tensor<3xui16>
   %1 = pphlo.convert %0 : (tensor<3xui16>)->tensor<3x!pphlo.secret<ui16>>
   %2 = pphlo.negate %1 : (tensor<3x!pphlo.secret<ui16>>)->tensor<3x!pphlo.secret<ui16>>
   %3 = arith.constant dense<[0, 65280, 1]> : tensor<3xui16>
   pphlo.custom_call @expect_eq(%3, %2) : (tensor<3xui16>, tensor<3x!pphlo.secret<ui16>>)->()
   func.return
}

// -----

func.func @negate_op_test_i32_i32_p() {
   %0 = arith.constant dense<[-2147483648, -65537, 0, 65536, 2147483647]> : tensor<5xi32>
   %1 = pphlo.negate %0 : (tensor<5xi32>)->tensor<5xi32>
   %2 = arith.constant dense<[-2147483648, 65537, 0, -65536, -2147483647]> : tensor<5xi32>
   pphlo.custom_call @expect_eq(%1, %2) : (tensor<5xi32>, tensor<5xi32>)->()
   func.return
}

// -----

func.func @negate_op_test_i32_i32_s() {
   %0 = arith.constant dense<[-2147483648, -65537, 0, 65536, 2147483647]> : tensor<5xi32>
   %1 = pphlo.convert %0 : (tensor<5xi32>)->tensor<5x!pphlo.secret<i32>>
   %2 = pphlo.negate %1 : (tensor<5x!pphlo.secret<i32>>)->tensor<5x!pphlo.secret<i32>>
   %3 = arith.constant dense<[-2147483648, 65537, 0, -65536, -2147483647]> : tensor<5xi32>
   pphlo.custom_call @expect_eq(%3, %2) : (tensor<5xi32>, tensor<5x!pphlo.secret<i32>>)->()
   func.return
}

// -----

func.func @negate_op_test_ui32_ui32_p() {
   %0 = arith.constant dense<[0, 65536, 4294967295]> : tensor<3xui32>
   %1 = pphlo.negate %0 : (tensor<3xui32>)->tensor<3xui32>
   %2 = arith.constant dense<[0, 4294901760, 1]> : tensor<3xui32>
   pphlo.custom_call @expect_eq(%1, %2) : (tensor<3xui32>, tensor<3xui32>)->()
   func.return
}

// -----

func.func @negate_op_test_ui32_ui32_s() {
   %0 = arith.constant dense<[0, 65536, 4294967295]> : tensor<3xui32>
   %1 = pphlo.convert %0 : (tensor<3xui32>)->tensor<3x!pphlo.secret<ui32>>
   %2 = pphlo.negate %1 : (tensor<3x!pphlo.secret<ui32>>)->tensor<3x!pphlo.secret<ui32>>
   %3 = arith.constant dense<[0, 4294901760, 1]> : tensor<3xui32>
   pphlo.custom_call @expect_eq(%3, %2) : (tensor<3xui32>, tensor<3x!pphlo.secret<ui32>>)->()
   func.return
}

// -----

func.func @negate_op_test_i64_i64_p() {
   %0 = arith.constant dense<[-9223372036854775808, -2147483649, 0, 2147483648, 9223372036854775807]> : tensor<5xi64>
   %1 = pphlo.negate %0 : (tensor<5xi64>)->tensor<5xi64>
   %2 = arith.constant dense<[-9223372036854775808, 2147483649, 0, -2147483648, -9223372036854775807]> : tensor<5xi64>
   pphlo.custom_call @expect_eq(%1, %2) : (tensor<5xi64>, tensor<5xi64>)->()
   func.return
}

// -----

func.func @negate_op_test_i64_i64_s() {
   %0 = arith.constant dense<[-9223372036854775808, -2147483649, 0, 2147483648, 9223372036854775807]> : tensor<5xi64>
   %1 = pphlo.convert %0 : (tensor<5xi64>)->tensor<5x!pphlo.secret<i64>>
   %2 = pphlo.negate %1 : (tensor<5x!pphlo.secret<i64>>)->tensor<5x!pphlo.secret<i64>>
   %3 = arith.constant dense<[-9223372036854775808, 2147483649, 0, -2147483648, -9223372036854775807]> : tensor<5xi64>
   pphlo.custom_call @expect_eq(%3, %2) : (tensor<5xi64>, tensor<5x!pphlo.secret<i64>>)->()
   func.return
}

// -----

func.func @negate_op_test_ui64_ui64_p() {
   %0 = arith.constant dense<[0, 4294967296, 18446744073709551615]> : tensor<3xui64>
   %1 = pphlo.negate %0 : (tensor<3xui64>)->tensor<3xui64>
   %2 = arith.constant dense<[0, 18446744069414584320, 1]> : tensor<3xui64>
   pphlo.custom_call @expect_eq(%1, %2) : (tensor<3xui64>, tensor<3xui64>)->()
   func.return
}

// -----

func.func @negate_op_test_ui64_ui64_s() {
   %0 = arith.constant dense<[0, 4294967296, 18446744073709551615]> : tensor<3xui64>
   %1 = pphlo.convert %0 : (tensor<3xui64>)->tensor<3x!pphlo.secret<ui64>>
   %2 = pphlo.negate %1 : (tensor<3x!pphlo.secret<ui64>>)->tensor<3x!pphlo.secret<ui64>>
   %3 = arith.constant dense<[0, 18446744069414584320, 1]> : tensor<3xui64>
   pphlo.custom_call @expect_eq(%3, %2) : (tensor<3xui64>, tensor<3x!pphlo.secret<ui64>>)->()
   func.return
}

// -----

func.func @negate_op_test_f16_f16_p() {
   %0 = arith.constant dense<[0.0, 1.0, 0.125, 0.1, 3.140630]> : tensor<5xf16>
   %1 = pphlo.negate %0 : (tensor<5xf16>)->tensor<5xf16>
   %2 = arith.constant dense<[0.000000e+00, -1.000000e+00, -1.250000e-01, -9.997550e-02, -3.140630e+00]> : tensor<5xf16>
   pphlo.custom_call @expect_almost_eq(%1, %2) : (tensor<5xf16>, tensor<5xf16>)->()
   func.return
}

// -----

func.func @negate_op_test_f16_f16_s() {
   %0 = arith.constant dense<[0.0, 1.0, 0.125, 0.1, 3.140630]> : tensor<5xf16>
   %1 = pphlo.convert %0 : (tensor<5xf16>)->tensor<5x!pphlo.secret<f16>>
   %2 = pphlo.negate %1 : (tensor<5x!pphlo.secret<f16>>)->tensor<5x!pphlo.secret<f16>>
   %3 = arith.constant dense<[0.000000e+00, -1.000000e+00, -1.250000e-01, -9.997550e-02, -3.140630e+00]> : tensor<5xf16>
   pphlo.custom_call @expect_almost_eq(%3, %2) : (tensor<5xf16>, tensor<5x!pphlo.secret<f16>>)->()
   func.return
}

// -----

func.func @negate_op_test_f32_f32_p() {
   %0 = arith.constant dense<[0.0, 1.0, 0.125, 0.1, 3.14159274]> : tensor<5xf32>
   %1 = pphlo.negate %0 : (tensor<5xf32>)->tensor<5xf32>
   %2 = arith.constant dense<[0.000000e+00, -1.000000e+00, -1.250000e-01, -1.000000e-01, -3.14159274]> : tensor<5xf32>
   pphlo.custom_call @expect_almost_eq(%1, %2) : (tensor<5xf32>, tensor<5xf32>)->()
   func.return
}

// -----

func.func @negate_op_test_f32_f32_s() {
   %0 = arith.constant dense<[0.0, 1.0, 0.125, 0.1, 3.14159274]> : tensor<5xf32>
   %1 = pphlo.convert %0 : (tensor<5xf32>)->tensor<5x!pphlo.secret<f32>>
   %2 = pphlo.negate %1 : (tensor<5x!pphlo.secret<f32>>)->tensor<5x!pphlo.secret<f32>>
   %3 = arith.constant dense<[0.000000e+00, -1.000000e+00, -1.250000e-01, -1.000000e-01, -3.14159274]> : tensor<5xf32>
   pphlo.custom_call @expect_almost_eq(%3, %2) : (tensor<5xf32>, tensor<5x!pphlo.secret<f32>>)->()
   func.return
}

// -----

func.func @negate_op_test_f64_f64_p() {
   %0 = arith.constant dense<[0.0, 1.0, 0.125, 0.1, 3.1415926535897931]> : tensor<5xf64>
   %1 = pphlo.negate %0 : (tensor<5xf64>)->tensor<5xf64>
   %2 = arith.constant dense<[0.000000e+00, -1.000000e+00, -1.250000e-01, -1.000000e-01, -3.1415926535897931]> : tensor<5xf64>
   pphlo.custom_call @expect_almost_eq(%1, %2) : (tensor<5xf64>, tensor<5xf64>)->()
   func.return
}

// -----

func.func @negate_op_test_f64_f64_s() {
   %0 = arith.constant dense<[0.0, 1.0, 0.125, 0.1, 3.1415926535897931]> : tensor<5xf64>
   %1 = pphlo.convert %0 : (tensor<5xf64>)->tensor<5x!pphlo.secret<f64>>
   %2 = pphlo.negate %1 : (tensor<5x!pphlo.secret<f64>>)->tensor<5x!pphlo.secret<f64>>
   %3 = arith.constant dense<[0.000000e+00, -1.000000e+00, -1.250000e-01, -1.000000e-01, -3.1415926535897931]> : tensor<5xf64>
   pphlo.custom_call @expect_almost_eq(%3, %2) : (tensor<5xf64>, tensor<5x!pphlo.secret<f64>>)->()
   func.return
}
