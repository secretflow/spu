// RUN: spu-translate --interpret -split-input-file %s
// AUTO GENERATED, DO NOT EDIT

func.func @add_op_test_i8_i8_pp() {
   %0 = pphlo.constant dense<[0, 1, 8, -9, 0]> : tensor<5xi8>
   %1 = pphlo.constant dense<[-128, -1, 8, -9, 127]> : tensor<5xi8>
   %2 = pphlo.add %0,%1 : (tensor<5xi8>,tensor<5xi8>)->tensor<5xi8>
   %3 = pphlo.constant dense<[-128, 0, 16, -18, 127]> : tensor<5xi8>
   pphlo.custom_call @expect_eq(%2, %3) : (tensor<5xi8>, tensor<5xi8>)->()
   func.return
}

// -----

func.func @add_op_test_i8_i8_ss() {
   %0 = pphlo.constant dense<[0, 1, 8, -9, 0]> : tensor<5xi8>
   %1 = pphlo.constant dense<[-128, -1, 8, -9, 127]> : tensor<5xi8>
   %2 = pphlo.convert %0 : (tensor<5xi8>)->tensor<5x!pphlo.secret<i8>>
   %3 = pphlo.convert %1 : (tensor<5xi8>)->tensor<5x!pphlo.secret<i8>>
   %4 = pphlo.add %2, %3 : (tensor<5x!pphlo.secret<i8>>,tensor<5x!pphlo.secret<i8>>)->tensor<5x!pphlo.secret<i8>>
   %5 = pphlo.constant dense<[-128, 0, 16, -18, 127]> : tensor<5xi8>
   %6 = pphlo.convert %4 : (tensor<5x!pphlo.secret<i8>>)->tensor<5xi8>
   pphlo.custom_call @expect_eq(%5, %6) : (tensor<5xi8>, tensor<5xi8>)->()
   func.return
}

// -----

func.func @add_op_test_ui8_ui8_pp() {
   %0 = pphlo.constant dense<[0, 16]> : tensor<2xui8>
   %1 = pphlo.constant dense<[255, 16]> : tensor<2xui8>
   %2 = pphlo.add %0,%1 : (tensor<2xui8>,tensor<2xui8>)->tensor<2xui8>
   %3 = pphlo.constant dense<[255, 32]> : tensor<2xui8>
   pphlo.custom_call @expect_eq(%2, %3) : (tensor<2xui8>, tensor<2xui8>)->()
   func.return
}

// -----

func.func @add_op_test_ui8_ui8_ss() {
   %0 = pphlo.constant dense<[0, 16]> : tensor<2xui8>
   %1 = pphlo.constant dense<[255, 16]> : tensor<2xui8>
   %2 = pphlo.convert %0 : (tensor<2xui8>)->tensor<2x!pphlo.secret<ui8>>
   %3 = pphlo.convert %1 : (tensor<2xui8>)->tensor<2x!pphlo.secret<ui8>>
   %4 = pphlo.add %2, %3 : (tensor<2x!pphlo.secret<ui8>>,tensor<2x!pphlo.secret<ui8>>)->tensor<2x!pphlo.secret<ui8>>
   %5 = pphlo.constant dense<[255, 32]> : tensor<2xui8>
   %6 = pphlo.convert %4 : (tensor<2x!pphlo.secret<ui8>>)->tensor<2xui8>
   pphlo.custom_call @expect_eq(%5, %6) : (tensor<2xui8>, tensor<2xui8>)->()
   func.return
}

// -----

func.func @add_op_test_i16_i16_pp() {
   %0 = pphlo.constant dense<[0, 1, 128, -129, 0]> : tensor<5xi16>
   %1 = pphlo.constant dense<[-32768, -1, 128, -129, 32767]> : tensor<5xi16>
   %2 = pphlo.add %0,%1 : (tensor<5xi16>,tensor<5xi16>)->tensor<5xi16>
   %3 = pphlo.constant dense<[-32768, 0, 256, -258, 32767]> : tensor<5xi16>
   pphlo.custom_call @expect_eq(%2, %3) : (tensor<5xi16>, tensor<5xi16>)->()
   func.return
}

// -----

func.func @add_op_test_i16_i16_ss() {
   %0 = pphlo.constant dense<[0, 1, 128, -129, 0]> : tensor<5xi16>
   %1 = pphlo.constant dense<[-32768, -1, 128, -129, 32767]> : tensor<5xi16>
   %2 = pphlo.convert %0 : (tensor<5xi16>)->tensor<5x!pphlo.secret<i16>>
   %3 = pphlo.convert %1 : (tensor<5xi16>)->tensor<5x!pphlo.secret<i16>>
   %4 = pphlo.add %2, %3 : (tensor<5x!pphlo.secret<i16>>,tensor<5x!pphlo.secret<i16>>)->tensor<5x!pphlo.secret<i16>>
   %5 = pphlo.constant dense<[-32768, 0, 256, -258, 32767]> : tensor<5xi16>
   %6 = pphlo.convert %4 : (tensor<5x!pphlo.secret<i16>>)->tensor<5xi16>
   pphlo.custom_call @expect_eq(%5, %6) : (tensor<5xi16>, tensor<5xi16>)->()
   func.return
}

// -----

func.func @add_op_test_ui16_ui16_pp() {
   %0 = pphlo.constant dense<[0, 256]> : tensor<2xui16>
   %1 = pphlo.constant dense<[65535, 256]> : tensor<2xui16>
   %2 = pphlo.add %0,%1 : (tensor<2xui16>,tensor<2xui16>)->tensor<2xui16>
   %3 = pphlo.constant dense<[65535, 512]> : tensor<2xui16>
   pphlo.custom_call @expect_eq(%2, %3) : (tensor<2xui16>, tensor<2xui16>)->()
   func.return
}

// -----

func.func @add_op_test_ui16_ui16_ss() {
   %0 = pphlo.constant dense<[0, 256]> : tensor<2xui16>
   %1 = pphlo.constant dense<[65535, 256]> : tensor<2xui16>
   %2 = pphlo.convert %0 : (tensor<2xui16>)->tensor<2x!pphlo.secret<ui16>>
   %3 = pphlo.convert %1 : (tensor<2xui16>)->tensor<2x!pphlo.secret<ui16>>
   %4 = pphlo.add %2, %3 : (tensor<2x!pphlo.secret<ui16>>,tensor<2x!pphlo.secret<ui16>>)->tensor<2x!pphlo.secret<ui16>>
   %5 = pphlo.constant dense<[65535, 512]> : tensor<2xui16>
   %6 = pphlo.convert %4 : (tensor<2x!pphlo.secret<ui16>>)->tensor<2xui16>
   pphlo.custom_call @expect_eq(%5, %6) : (tensor<2xui16>, tensor<2xui16>)->()
   func.return
}

// -----

func.func @add_op_test_i32_i32_pp() {
   %0 = pphlo.constant dense<[0, 1, 32768, -32769, 0]> : tensor<5xi32>
   %1 = pphlo.constant dense<[-2147483648, -1, 32768, -32769, 2147483647]> : tensor<5xi32>
   %2 = pphlo.add %0,%1 : (tensor<5xi32>,tensor<5xi32>)->tensor<5xi32>
   %3 = pphlo.constant dense<[-2147483648, 0, 65536, -65538, 2147483647]> : tensor<5xi32>
   pphlo.custom_call @expect_eq(%2, %3) : (tensor<5xi32>, tensor<5xi32>)->()
   func.return
}

// -----

func.func @add_op_test_i32_i32_ss() {
   %0 = pphlo.constant dense<[0, 1, 32768, -32769, 0]> : tensor<5xi32>
   %1 = pphlo.constant dense<[-2147483648, -1, 32768, -32769, 2147483647]> : tensor<5xi32>
   %2 = pphlo.convert %0 : (tensor<5xi32>)->tensor<5x!pphlo.secret<i32>>
   %3 = pphlo.convert %1 : (tensor<5xi32>)->tensor<5x!pphlo.secret<i32>>
   %4 = pphlo.add %2, %3 : (tensor<5x!pphlo.secret<i32>>,tensor<5x!pphlo.secret<i32>>)->tensor<5x!pphlo.secret<i32>>
   %5 = pphlo.constant dense<[-2147483648, 0, 65536, -65538, 2147483647]> : tensor<5xi32>
   %6 = pphlo.convert %4 : (tensor<5x!pphlo.secret<i32>>)->tensor<5xi32>
   pphlo.custom_call @expect_eq(%5, %6) : (tensor<5xi32>, tensor<5xi32>)->()
   func.return
}

// -----

func.func @add_op_test_ui32_ui32_pp() {
   %0 = pphlo.constant dense<[0, 65536]> : tensor<2xui32>
   %1 = pphlo.constant dense<[4294967295, 65536]> : tensor<2xui32>
   %2 = pphlo.add %0,%1 : (tensor<2xui32>,tensor<2xui32>)->tensor<2xui32>
   %3 = pphlo.constant dense<[4294967295, 131072]> : tensor<2xui32>
   pphlo.custom_call @expect_eq(%2, %3) : (tensor<2xui32>, tensor<2xui32>)->()
   func.return
}

// -----

func.func @add_op_test_ui32_ui32_ss() {
   %0 = pphlo.constant dense<[0, 65536]> : tensor<2xui32>
   %1 = pphlo.constant dense<[4294967295, 65536]> : tensor<2xui32>
   %2 = pphlo.convert %0 : (tensor<2xui32>)->tensor<2x!pphlo.secret<ui32>>
   %3 = pphlo.convert %1 : (tensor<2xui32>)->tensor<2x!pphlo.secret<ui32>>
   %4 = pphlo.add %2, %3 : (tensor<2x!pphlo.secret<ui32>>,tensor<2x!pphlo.secret<ui32>>)->tensor<2x!pphlo.secret<ui32>>
   %5 = pphlo.constant dense<[4294967295, 131072]> : tensor<2xui32>
   %6 = pphlo.convert %4 : (tensor<2x!pphlo.secret<ui32>>)->tensor<2xui32>
   pphlo.custom_call @expect_eq(%5, %6) : (tensor<2xui32>, tensor<2xui32>)->()
   func.return
}

// -----

func.func @add_op_test_i64_i64_pp() {
   %0 = pphlo.constant dense<[0, 1, 2147483648, -2147483649, 0]> : tensor<5xi64>
   %1 = pphlo.constant dense<[-9223372036854775808, -1, 2147483648, -2147483649, 9223372036854775807]> : tensor<5xi64>
   %2 = pphlo.add %0,%1 : (tensor<5xi64>,tensor<5xi64>)->tensor<5xi64>
   %3 = pphlo.constant dense<[-9223372036854775808, 0, 4294967296, -4294967298, 9223372036854775807]> : tensor<5xi64>
   pphlo.custom_call @expect_eq(%2, %3) : (tensor<5xi64>, tensor<5xi64>)->()
   func.return
}

// -----

func.func @add_op_test_i64_i64_ss() {
   %0 = pphlo.constant dense<[0, 1, 2147483648, -2147483649, 0]> : tensor<5xi64>
   %1 = pphlo.constant dense<[-9223372036854775808, -1, 2147483648, -2147483649, 9223372036854775807]> : tensor<5xi64>
   %2 = pphlo.convert %0 : (tensor<5xi64>)->tensor<5x!pphlo.secret<i64>>
   %3 = pphlo.convert %1 : (tensor<5xi64>)->tensor<5x!pphlo.secret<i64>>
   %4 = pphlo.add %2, %3 : (tensor<5x!pphlo.secret<i64>>,tensor<5x!pphlo.secret<i64>>)->tensor<5x!pphlo.secret<i64>>
   %5 = pphlo.constant dense<[-9223372036854775808, 0, 4294967296, -4294967298, 9223372036854775807]> : tensor<5xi64>
   %6 = pphlo.convert %4 : (tensor<5x!pphlo.secret<i64>>)->tensor<5xi64>
   pphlo.custom_call @expect_eq(%5, %6) : (tensor<5xi64>, tensor<5xi64>)->()
   func.return
}

// -----

func.func @add_op_test_ui64_ui64_pp() {
   %0 = pphlo.constant dense<[0, 4294967296]> : tensor<2xui64>
   %1 = pphlo.constant dense<[18446744073709551615, 4294967296]> : tensor<2xui64>
   %2 = pphlo.add %0,%1 : (tensor<2xui64>,tensor<2xui64>)->tensor<2xui64>
   %3 = pphlo.constant dense<[18446744073709551615, 8589934592]> : tensor<2xui64>
   pphlo.custom_call @expect_eq(%2, %3) : (tensor<2xui64>, tensor<2xui64>)->()
   func.return
}

// -----

func.func @add_op_test_ui64_ui64_ss() {
   %0 = pphlo.constant dense<[0, 4294967296]> : tensor<2xui64>
   %1 = pphlo.constant dense<[18446744073709551615, 4294967296]> : tensor<2xui64>
   %2 = pphlo.convert %0 : (tensor<2xui64>)->tensor<2x!pphlo.secret<ui64>>
   %3 = pphlo.convert %1 : (tensor<2xui64>)->tensor<2x!pphlo.secret<ui64>>
   %4 = pphlo.add %2, %3 : (tensor<2x!pphlo.secret<ui64>>,tensor<2x!pphlo.secret<ui64>>)->tensor<2x!pphlo.secret<ui64>>
   %5 = pphlo.constant dense<[18446744073709551615, 8589934592]> : tensor<2xui64>
   %6 = pphlo.convert %4 : (tensor<2x!pphlo.secret<ui64>>)->tensor<2xui64>
   pphlo.custom_call @expect_eq(%5, %6) : (tensor<2xui64>, tensor<2xui64>)->()
   func.return
}

// -----

func.func @add_op_test_i1_i1_pp() {
   %0 = pphlo.constant dense<[false, false, true, true]> : tensor<4xi1>
   %1 = pphlo.constant dense<[false, true, false, true]> : tensor<4xi1>
   %2 = pphlo.add %0,%1 : (tensor<4xi1>,tensor<4xi1>)->tensor<4xi1>
   %3 = pphlo.constant dense<[false, true, true, true]> : tensor<4xi1>
   pphlo.custom_call @expect_eq(%2, %3) : (tensor<4xi1>, tensor<4xi1>)->()
   func.return
}

// -----

func.func @add_op_test_i1_i1_ss() {
   %0 = pphlo.constant dense<[false, false, true, true]> : tensor<4xi1>
   %1 = pphlo.constant dense<[false, true, false, true]> : tensor<4xi1>
   %2 = pphlo.convert %0 : (tensor<4xi1>)->tensor<4x!pphlo.secret<i1>>
   %3 = pphlo.convert %1 : (tensor<4xi1>)->tensor<4x!pphlo.secret<i1>>
   %4 = pphlo.add %2, %3 : (tensor<4x!pphlo.secret<i1>>,tensor<4x!pphlo.secret<i1>>)->tensor<4x!pphlo.secret<i1>>
   %5 = pphlo.constant dense<[false, true, true, true]> : tensor<4xi1>
   %6 = pphlo.convert %4 : (tensor<4x!pphlo.secret<i1>>)->tensor<4xi1>
   pphlo.custom_call @expect_eq(%5, %6) : (tensor<4xi1>, tensor<4xi1>)->()
   func.return
}

// -----

func.func @add_op_test_f16_f16_pp() {
   %0 = pphlo.constant dense<[0.0, 1.0, 0.125, 0.1, 3.141]> : tensor<5xf16>
   %1 = pphlo.constant dense<[0.0, 7.0, 0.75, 0.3, 3.141]> : tensor<5xf16>
   %2 = pphlo.add %0,%1 : (tensor<5xf16>,tensor<5xf16>)->tensor<5xf16>
   %3 = pphlo.constant dense<[0.000000e+00, 8.000000e+00, 8.750000e-01, 3.999020e-01, 6.281250e+00]> : tensor<5xf16>
   pphlo.custom_call @expect_almost_eq(%2, %3) : (tensor<5xf16>, tensor<5xf16>)->()
   func.return
}

// -----

func.func @add_op_test_f16_f16_ss() {
   %0 = pphlo.constant dense<[0.0, 1.0, 0.125, 0.1, 3.141]> : tensor<5xf16>
   %1 = pphlo.constant dense<[0.0, 7.0, 0.75, 0.3, 3.141]> : tensor<5xf16>
   %2 = pphlo.convert %0 : (tensor<5xf16>)->tensor<5x!pphlo.secret<f16>>
   %3 = pphlo.convert %1 : (tensor<5xf16>)->tensor<5x!pphlo.secret<f16>>
   %4 = pphlo.add %2, %3 : (tensor<5x!pphlo.secret<f16>>,tensor<5x!pphlo.secret<f16>>)->tensor<5x!pphlo.secret<f16>>
   %5 = pphlo.constant dense<[0.000000e+00, 8.000000e+00, 8.750000e-01, 3.999020e-01, 6.281250e+00]> : tensor<5xf16>
   %6 = pphlo.convert %4 : (tensor<5x!pphlo.secret<f16>>)->tensor<5xf16>
   pphlo.custom_call @expect_almost_eq(%5, %6) : (tensor<5xf16>, tensor<5xf16>)->()
   func.return
}

// -----

func.func @add_op_test_f32_f32_pp() {
   %0 = pphlo.constant dense<[0.0, 1.0, 0.125, 0.1, 3.14159265]> : tensor<5xf32>
   %1 = pphlo.constant dense<[0.0, 7.0, 0.75,  0.3, 3.14159265]> : tensor<5xf32>
   %2 = pphlo.add %0,%1 : (tensor<5xf32>,tensor<5xf32>)->tensor<5xf32>
   %3 = pphlo.constant dense<[0.000000e+00, 8.000000e+00, 8.750000e-01, 4.000000e-01, 6.28318548]> : tensor<5xf32>
   pphlo.custom_call @expect_almost_eq(%2, %3) : (tensor<5xf32>, tensor<5xf32>)->()
   func.return
}

// -----

func.func @add_op_test_f32_f32_ss() {
   %0 = pphlo.constant dense<[0.0, 1.0, 0.125, 0.1, 3.14159265]> : tensor<5xf32>
   %1 = pphlo.constant dense<[0.0, 7.0, 0.75,  0.3, 3.14159265]> : tensor<5xf32>
   %2 = pphlo.convert %0 : (tensor<5xf32>)->tensor<5x!pphlo.secret<f32>>
   %3 = pphlo.convert %1 : (tensor<5xf32>)->tensor<5x!pphlo.secret<f32>>
   %4 = pphlo.add %2, %3 : (tensor<5x!pphlo.secret<f32>>,tensor<5x!pphlo.secret<f32>>)->tensor<5x!pphlo.secret<f32>>
   %5 = pphlo.constant dense<[0.000000e+00, 8.000000e+00, 8.750000e-01, 4.000000e-01, 6.28318548]> : tensor<5xf32>
   %6 = pphlo.convert %4 : (tensor<5x!pphlo.secret<f32>>)->tensor<5xf32>
   pphlo.custom_call @expect_almost_eq(%5, %6) : (tensor<5xf32>, tensor<5xf32>)->()
   func.return
}

// -----

func.func @add_op_test_f64_f64_pp() {
   %0 = pphlo.constant dense<[0.0, 1.0, 0.125, 0.1, 3.14159265358979323846]> : tensor<5xf64>
   %1 = pphlo.constant dense<[0.0, 7.0, 0.75, 0.3, 3.14159265358979323846]> : tensor<5xf64>
   %2 = pphlo.add %0,%1 : (tensor<5xf64>,tensor<5xf64>)->tensor<5xf64>
   %3 = pphlo.constant dense<[0.000000e+00, 8.000000e+00, 8.750000e-01, 4.000000e-01, 6.2831853071795862]> : tensor<5xf64>
   pphlo.custom_call @expect_almost_eq(%2, %3) : (tensor<5xf64>, tensor<5xf64>)->()
   func.return
}

// -----

func.func @add_op_test_f64_f64_ss() {
   %0 = pphlo.constant dense<[0.0, 1.0, 0.125, 0.1, 3.14159265358979323846]> : tensor<5xf64>
   %1 = pphlo.constant dense<[0.0, 7.0, 0.75, 0.3, 3.14159265358979323846]> : tensor<5xf64>
   %2 = pphlo.convert %0 : (tensor<5xf64>)->tensor<5x!pphlo.secret<f64>>
   %3 = pphlo.convert %1 : (tensor<5xf64>)->tensor<5x!pphlo.secret<f64>>
   %4 = pphlo.add %2, %3 : (tensor<5x!pphlo.secret<f64>>,tensor<5x!pphlo.secret<f64>>)->tensor<5x!pphlo.secret<f64>>
   %5 = pphlo.constant dense<[0.000000e+00, 8.000000e+00, 8.750000e-01, 4.000000e-01, 6.2831853071795862]> : tensor<5xf64>
   %6 = pphlo.convert %4 : (tensor<5x!pphlo.secret<f64>>)->tensor<5xf64>
   pphlo.custom_call @expect_almost_eq(%5, %6) : (tensor<5xf64>, tensor<5xf64>)->()
   func.return
}

// -----

func.func @add_op_test_i8_i8_pp() {
   %0 = pphlo.constant dense<2> : tensor<i8>
   %1 = pphlo.constant dense<3> : tensor<i8>
   %2 = pphlo.add %0,%1 : (tensor<i8>,tensor<i8>)->tensor<i8>
   %3 = pphlo.constant dense<5> : tensor<i8>
   pphlo.custom_call @expect_eq(%2, %3) : (tensor<i8>, tensor<i8>)->()
   func.return
}

// -----

func.func @add_op_test_i8_i8_ss() {
   %0 = pphlo.constant dense<2> : tensor<i8>
   %1 = pphlo.constant dense<3> : tensor<i8>
   %2 = pphlo.convert %0 : (tensor<i8>)->tensor<!pphlo.secret<i8>>
   %3 = pphlo.convert %1 : (tensor<i8>)->tensor<!pphlo.secret<i8>>
   %4 = pphlo.add %2, %3 : (tensor<!pphlo.secret<i8>>,tensor<!pphlo.secret<i8>>)->tensor<!pphlo.secret<i8>>
   %5 = pphlo.constant dense<5> : tensor<i8>
   %6 = pphlo.convert %4 : (tensor<!pphlo.secret<i8>>)->tensor<i8>
   pphlo.custom_call @expect_eq(%5, %6) : (tensor<i8>, tensor<i8>)->()
   func.return
}

// -----

func.func @add_op_test_i8_i8_pp() {
   %0 = pphlo.constant dense<2> : tensor<2x0x3xi8>
   %1 = pphlo.constant dense<3> : tensor<2x0x3xi8>
   %2 = pphlo.add %0,%1 : (tensor<2x0x3xi8>,tensor<2x0x3xi8>)->tensor<2x0x3xi8>
   %3 = pphlo.constant dense<> : tensor<2x0x3xi8>
   pphlo.custom_call @expect_eq(%2, %3) : (tensor<2x0x3xi8>, tensor<2x0x3xi8>)->()
   func.return
}

// -----

func.func @add_op_test_i8_i8_ss() {
   %0 = pphlo.constant dense<2> : tensor<2x0x3xi8>
   %1 = pphlo.constant dense<3> : tensor<2x0x3xi8>
   %2 = pphlo.convert %0 : (tensor<2x0x3xi8>)->tensor<2x0x3x!pphlo.secret<i8>>
   %3 = pphlo.convert %1 : (tensor<2x0x3xi8>)->tensor<2x0x3x!pphlo.secret<i8>>
   %4 = pphlo.add %2, %3 : (tensor<2x0x3x!pphlo.secret<i8>>,tensor<2x0x3x!pphlo.secret<i8>>)->tensor<2x0x3x!pphlo.secret<i8>>
   %5 = pphlo.constant dense<> : tensor<2x0x3xi8>
   %6 = pphlo.convert %4 : (tensor<2x0x3x!pphlo.secret<i8>>)->tensor<2x0x3xi8>
   pphlo.custom_call @expect_eq(%5, %6) : (tensor<2x0x3xi8>, tensor<2x0x3xi8>)->()
   func.return
}
