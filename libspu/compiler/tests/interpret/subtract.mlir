// RUN: spu-translate --protocol_kind=1 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=2 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=3 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=4 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=5 --interpret -split-input-file %s
// AUTO GENERATED, DO NOT EDIT

func.func @subtract_op_test_i8_i8_pp() {
   %0 = arith.constant dense<[0, 1, 8, -9, 0]> : tensor<5xi8>
   %1 = arith.constant dense<[-128, -1, 8, -9, 127]> : tensor<5xi8>
   %2 = pphlo.subtract %0,%1 : (tensor<5xi8>,tensor<5xi8>)->tensor<5xi8>
   %3 = arith.constant dense<[-128, 2, 0, 0, -127]> : tensor<5xi8>
   pphlo.custom_call @expect_eq(%2, %3) : (tensor<5xi8>, tensor<5xi8>)->()
   func.return
}

// -----

func.func @subtract_op_test_i8_i8_ss() {
   %0 = arith.constant dense<[0, 1, 8, -9, 0]> : tensor<5xi8>
   %1 = arith.constant dense<[-128, -1, 8, -9, 127]> : tensor<5xi8>
   %2 = pphlo.convert %0 : (tensor<5xi8>)->tensor<5x!pphlo.secret<i8>>
   %3 = pphlo.convert %1 : (tensor<5xi8>)->tensor<5x!pphlo.secret<i8>>
   %4 = pphlo.subtract %2, %3 : (tensor<5x!pphlo.secret<i8>>,tensor<5x!pphlo.secret<i8>>)->tensor<5x!pphlo.secret<i8>>
   %5 = arith.constant dense<[-128, 2, 0, 0, -127]> : tensor<5xi8>
   pphlo.custom_call @expect_eq(%5, %4) : (tensor<5xi8>, tensor<5x!pphlo.secret<i8>>)->()
   func.return
}

// -----

func.func @subtract_op_test_ui8_ui8_pp() {
   %0 = arith.constant dense<[0, 16]> : tensor<2xui8>
   %1 = arith.constant dense<[255, 16]> : tensor<2xui8>
   %2 = pphlo.subtract %0,%1 : (tensor<2xui8>,tensor<2xui8>)->tensor<2xui8>
   %3 = arith.constant dense<[1, 0]> : tensor<2xui8>
   pphlo.custom_call @expect_eq(%2, %3) : (tensor<2xui8>, tensor<2xui8>)->()
   func.return
}

// -----

func.func @subtract_op_test_ui8_ui8_ss() {
   %0 = arith.constant dense<[0, 16]> : tensor<2xui8>
   %1 = arith.constant dense<[255, 16]> : tensor<2xui8>
   %2 = pphlo.convert %0 : (tensor<2xui8>)->tensor<2x!pphlo.secret<ui8>>
   %3 = pphlo.convert %1 : (tensor<2xui8>)->tensor<2x!pphlo.secret<ui8>>
   %4 = pphlo.subtract %2, %3 : (tensor<2x!pphlo.secret<ui8>>,tensor<2x!pphlo.secret<ui8>>)->tensor<2x!pphlo.secret<ui8>>
   %5 = arith.constant dense<[1, 0]> : tensor<2xui8>
   pphlo.custom_call @expect_eq(%5, %4) : (tensor<2xui8>, tensor<2x!pphlo.secret<ui8>>)->()
   func.return
}

// -----

func.func @subtract_op_test_i16_i16_pp() {
   %0 = arith.constant dense<[0, 1, 128, -129, 0]> : tensor<5xi16>
   %1 = arith.constant dense<[-32768, -1, 128, -129, 32767]> : tensor<5xi16>
   %2 = pphlo.subtract %0,%1 : (tensor<5xi16>,tensor<5xi16>)->tensor<5xi16>
   %3 = arith.constant dense<[-32768, 2, 0, 0, -32767]> : tensor<5xi16>
   pphlo.custom_call @expect_eq(%2, %3) : (tensor<5xi16>, tensor<5xi16>)->()
   func.return
}

// -----

func.func @subtract_op_test_i16_i16_ss() {
   %0 = arith.constant dense<[0, 1, 128, -129, 0]> : tensor<5xi16>
   %1 = arith.constant dense<[-32768, -1, 128, -129, 32767]> : tensor<5xi16>
   %2 = pphlo.convert %0 : (tensor<5xi16>)->tensor<5x!pphlo.secret<i16>>
   %3 = pphlo.convert %1 : (tensor<5xi16>)->tensor<5x!pphlo.secret<i16>>
   %4 = pphlo.subtract %2, %3 : (tensor<5x!pphlo.secret<i16>>,tensor<5x!pphlo.secret<i16>>)->tensor<5x!pphlo.secret<i16>>
   %5 = arith.constant dense<[-32768, 2, 0, 0, -32767]> : tensor<5xi16>
   pphlo.custom_call @expect_eq(%5, %4) : (tensor<5xi16>, tensor<5x!pphlo.secret<i16>>)->()
   func.return
}

// -----

func.func @subtract_op_test_ui16_ui16_pp() {
   %0 = arith.constant dense<[0, 256]> : tensor<2xui16>
   %1 = arith.constant dense<[65535, 256]> : tensor<2xui16>
   %2 = pphlo.subtract %0,%1 : (tensor<2xui16>,tensor<2xui16>)->tensor<2xui16>
   %3 = arith.constant dense<[1, 0]> : tensor<2xui16>
   pphlo.custom_call @expect_eq(%2, %3) : (tensor<2xui16>, tensor<2xui16>)->()
   func.return
}

// -----

func.func @subtract_op_test_ui16_ui16_ss() {
   %0 = arith.constant dense<[0, 256]> : tensor<2xui16>
   %1 = arith.constant dense<[65535, 256]> : tensor<2xui16>
   %2 = pphlo.convert %0 : (tensor<2xui16>)->tensor<2x!pphlo.secret<ui16>>
   %3 = pphlo.convert %1 : (tensor<2xui16>)->tensor<2x!pphlo.secret<ui16>>
   %4 = pphlo.subtract %2, %3 : (tensor<2x!pphlo.secret<ui16>>,tensor<2x!pphlo.secret<ui16>>)->tensor<2x!pphlo.secret<ui16>>
   %5 = arith.constant dense<[1, 0]> : tensor<2xui16>
   pphlo.custom_call @expect_eq(%5, %4) : (tensor<2xui16>, tensor<2x!pphlo.secret<ui16>>)->()
   func.return
}

// -----

func.func @subtract_op_test_i32_i32_pp() {
   %0 = arith.constant dense<[0, 1, 32768, -32769, 0]> : tensor<5xi32>
   %1 = arith.constant dense<[-2147483648, -1, 32768, -32769, 2147483647]> : tensor<5xi32>
   %2 = pphlo.subtract %0,%1 : (tensor<5xi32>,tensor<5xi32>)->tensor<5xi32>
   %3 = arith.constant dense<[-2147483648, 2, 0, 0, -2147483647]> : tensor<5xi32>
   pphlo.custom_call @expect_eq(%2, %3) : (tensor<5xi32>, tensor<5xi32>)->()
   func.return
}

// -----

func.func @subtract_op_test_i32_i32_ss() {
   %0 = arith.constant dense<[0, 1, 32768, -32769, 0]> : tensor<5xi32>
   %1 = arith.constant dense<[-2147483648, -1, 32768, -32769, 2147483647]> : tensor<5xi32>
   %2 = pphlo.convert %0 : (tensor<5xi32>)->tensor<5x!pphlo.secret<i32>>
   %3 = pphlo.convert %1 : (tensor<5xi32>)->tensor<5x!pphlo.secret<i32>>
   %4 = pphlo.subtract %2, %3 : (tensor<5x!pphlo.secret<i32>>,tensor<5x!pphlo.secret<i32>>)->tensor<5x!pphlo.secret<i32>>
   %5 = arith.constant dense<[-2147483648, 2, 0, 0, -2147483647]> : tensor<5xi32>
   pphlo.custom_call @expect_eq(%5, %4) : (tensor<5xi32>, tensor<5x!pphlo.secret<i32>>)->()
   func.return
}

// -----

func.func @subtract_op_test_ui32_ui32_pp() {
   %0 = arith.constant dense<[0, 65536]> : tensor<2xui32>
   %1 = arith.constant dense<[4294967295, 65536]> : tensor<2xui32>
   %2 = pphlo.subtract %0,%1 : (tensor<2xui32>,tensor<2xui32>)->tensor<2xui32>
   %3 = arith.constant dense<[1, 0]> : tensor<2xui32>
   pphlo.custom_call @expect_eq(%2, %3) : (tensor<2xui32>, tensor<2xui32>)->()
   func.return
}

// -----

func.func @subtract_op_test_ui32_ui32_ss() {
   %0 = arith.constant dense<[0, 65536]> : tensor<2xui32>
   %1 = arith.constant dense<[4294967295, 65536]> : tensor<2xui32>
   %2 = pphlo.convert %0 : (tensor<2xui32>)->tensor<2x!pphlo.secret<ui32>>
   %3 = pphlo.convert %1 : (tensor<2xui32>)->tensor<2x!pphlo.secret<ui32>>
   %4 = pphlo.subtract %2, %3 : (tensor<2x!pphlo.secret<ui32>>,tensor<2x!pphlo.secret<ui32>>)->tensor<2x!pphlo.secret<ui32>>
   %5 = arith.constant dense<[1, 0]> : tensor<2xui32>
   pphlo.custom_call @expect_eq(%5, %4) : (tensor<2xui32>, tensor<2x!pphlo.secret<ui32>>)->()
   func.return
}

// -----

func.func @subtract_op_test_i64_i64_pp() {
   %0 = arith.constant dense<[0, 1, 2147483648, -2147483649, 0]> : tensor<5xi64>
   %1 = arith.constant dense<[-9223372036854775808, -1, 2147483648, -2147483649, 9223372036854775807]> : tensor<5xi64>
   %2 = pphlo.subtract %0,%1 : (tensor<5xi64>,tensor<5xi64>)->tensor<5xi64>
   %3 = arith.constant dense<[-9223372036854775808, 2, 0, 0, -9223372036854775807]> : tensor<5xi64>
   pphlo.custom_call @expect_eq(%2, %3) : (tensor<5xi64>, tensor<5xi64>)->()
   func.return
}

// -----

func.func @subtract_op_test_i64_i64_ss() {
   %0 = arith.constant dense<[0, 1, 2147483648, -2147483649, 0]> : tensor<5xi64>
   %1 = arith.constant dense<[-9223372036854775808, -1, 2147483648, -2147483649, 9223372036854775807]> : tensor<5xi64>
   %2 = pphlo.convert %0 : (tensor<5xi64>)->tensor<5x!pphlo.secret<i64>>
   %3 = pphlo.convert %1 : (tensor<5xi64>)->tensor<5x!pphlo.secret<i64>>
   %4 = pphlo.subtract %2, %3 : (tensor<5x!pphlo.secret<i64>>,tensor<5x!pphlo.secret<i64>>)->tensor<5x!pphlo.secret<i64>>
   %5 = arith.constant dense<[-9223372036854775808, 2, 0, 0, -9223372036854775807]> : tensor<5xi64>
   pphlo.custom_call @expect_eq(%5, %4) : (tensor<5xi64>, tensor<5x!pphlo.secret<i64>>)->()
   func.return
}

// -----

func.func @subtract_op_test_ui64_ui64_pp() {
   %0 = arith.constant dense<[0, 4294967296]> : tensor<2xui64>
   %1 = arith.constant dense<[18446744073709551615, 4294967296]> : tensor<2xui64>
   %2 = pphlo.subtract %0,%1 : (tensor<2xui64>,tensor<2xui64>)->tensor<2xui64>
   %3 = arith.constant dense<[1, 0]> : tensor<2xui64>
   pphlo.custom_call @expect_eq(%2, %3) : (tensor<2xui64>, tensor<2xui64>)->()
   func.return
}

// -----

func.func @subtract_op_test_ui64_ui64_ss() {
   %0 = arith.constant dense<[0, 4294967296]> : tensor<2xui64>
   %1 = arith.constant dense<[18446744073709551615, 4294967296]> : tensor<2xui64>
   %2 = pphlo.convert %0 : (tensor<2xui64>)->tensor<2x!pphlo.secret<ui64>>
   %3 = pphlo.convert %1 : (tensor<2xui64>)->tensor<2x!pphlo.secret<ui64>>
   %4 = pphlo.subtract %2, %3 : (tensor<2x!pphlo.secret<ui64>>,tensor<2x!pphlo.secret<ui64>>)->tensor<2x!pphlo.secret<ui64>>
   %5 = arith.constant dense<[1, 0]> : tensor<2xui64>
   pphlo.custom_call @expect_eq(%5, %4) : (tensor<2xui64>, tensor<2x!pphlo.secret<ui64>>)->()
   func.return
}

// -----

func.func @subtract_op_test_f16_f16_pp() {
   %0 = arith.constant dense<[0.0, 1.0, 0.125, 0.1, 3.141]> : tensor<5xf16>
   %1 = arith.constant dense<[0.0, 7.0, 0.75 , 0.3, 3.141]> : tensor<5xf16>
   %2 = pphlo.subtract %0,%1 : (tensor<5xf16>,tensor<5xf16>)->tensor<5xf16>
   %3 = arith.constant dense<[0.000000e+00, -6.000000e+00, -6.250000e-01, -2.000730e-01, 0.000000e+00]> : tensor<5xf16>
   pphlo.custom_call @expect_almost_eq(%2, %3) : (tensor<5xf16>, tensor<5xf16>)->()
   func.return
}

// -----

func.func @subtract_op_test_f16_f16_ss() {
   %0 = arith.constant dense<[0.0, 1.0, 0.125, 0.1, 3.141]> : tensor<5xf16>
   %1 = arith.constant dense<[0.0, 7.0, 0.75 , 0.3, 3.141]> : tensor<5xf16>
   %2 = pphlo.convert %0 : (tensor<5xf16>)->tensor<5x!pphlo.secret<f16>>
   %3 = pphlo.convert %1 : (tensor<5xf16>)->tensor<5x!pphlo.secret<f16>>
   %4 = pphlo.subtract %2, %3 : (tensor<5x!pphlo.secret<f16>>,tensor<5x!pphlo.secret<f16>>)->tensor<5x!pphlo.secret<f16>>
   %5 = arith.constant dense<[0.000000e+00, -6.000000e+00, -6.250000e-01, -2.000730e-01, 0.000000e+00]> : tensor<5xf16>
   pphlo.custom_call @expect_almost_eq(%5, %4) : (tensor<5xf16>, tensor<5x!pphlo.secret<f16>>)->()
   func.return
}

// -----

func.func @subtract_op_test_f32_f32_pp() {
   %0 = arith.constant dense<[0.0, 1.0, 0.125, 0.1, 3.14159265]> : tensor<5xf32>
   %1 = arith.constant dense<[0.0, 7.0, 0.75 , 0.3, 3.14159265]> : tensor<5xf32>
   %2 = pphlo.subtract %0,%1 : (tensor<5xf32>,tensor<5xf32>)->tensor<5xf32>
   %3 = arith.constant dense<[0.000000e+00, -6.000000e+00, -6.250000e-01, -0.200000018, 0.000000e+0]> : tensor<5xf32>
   pphlo.custom_call @expect_almost_eq(%2, %3) : (tensor<5xf32>, tensor<5xf32>)->()
   func.return
}

// -----

func.func @subtract_op_test_f32_f32_ss() {
   %0 = arith.constant dense<[0.0, 1.0, 0.125, 0.1, 3.14159265]> : tensor<5xf32>
   %1 = arith.constant dense<[0.0, 7.0, 0.75 , 0.3, 3.14159265]> : tensor<5xf32>
   %2 = pphlo.convert %0 : (tensor<5xf32>)->tensor<5x!pphlo.secret<f32>>
   %3 = pphlo.convert %1 : (tensor<5xf32>)->tensor<5x!pphlo.secret<f32>>
   %4 = pphlo.subtract %2, %3 : (tensor<5x!pphlo.secret<f32>>,tensor<5x!pphlo.secret<f32>>)->tensor<5x!pphlo.secret<f32>>
   %5 = arith.constant dense<[0.000000e+00, -6.000000e+00, -6.250000e-01, -0.200000018, 0.000000e+0]> : tensor<5xf32>
   pphlo.custom_call @expect_almost_eq(%5, %4) : (tensor<5xf32>, tensor<5x!pphlo.secret<f32>>)->()
   func.return
}

// -----

func.func @subtract_op_test_f64_f64_pp() {
   %0 = arith.constant dense<[0.0, 1.0, 0.125, 0.1, 3.14159265358979323846]> : tensor<5xf64>
   %1 = arith.constant dense<[0.0, 7.0, 0.75 , 0.3, 3.14159265358979323846]> : tensor<5xf64>
   %2 = pphlo.subtract %0,%1 : (tensor<5xf64>,tensor<5xf64>)->tensor<5xf64>
   %3 = arith.constant dense<[0.000000e+00, -6.000000e+00, -6.250000e-01, -0.19999999999999998, 0.000000e+00]> : tensor<5xf64>
   pphlo.custom_call @expect_almost_eq(%2, %3) : (tensor<5xf64>, tensor<5xf64>)->()
   func.return
}

// -----

func.func @subtract_op_test_f64_f64_ss() {
   %0 = arith.constant dense<[0.0, 1.0, 0.125, 0.1, 3.14159265358979323846]> : tensor<5xf64>
   %1 = arith.constant dense<[0.0, 7.0, 0.75 , 0.3, 3.14159265358979323846]> : tensor<5xf64>
   %2 = pphlo.convert %0 : (tensor<5xf64>)->tensor<5x!pphlo.secret<f64>>
   %3 = pphlo.convert %1 : (tensor<5xf64>)->tensor<5x!pphlo.secret<f64>>
   %4 = pphlo.subtract %2, %3 : (tensor<5x!pphlo.secret<f64>>,tensor<5x!pphlo.secret<f64>>)->tensor<5x!pphlo.secret<f64>>
   %5 = arith.constant dense<[0.000000e+00, -6.000000e+00, -6.250000e-01, -0.19999999999999998, 0.000000e+00]> : tensor<5xf64>
   pphlo.custom_call @expect_almost_eq(%5, %4) : (tensor<5xf64>, tensor<5x!pphlo.secret<f64>>)->()
   func.return
}
