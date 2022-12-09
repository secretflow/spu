// RUN: mlir-pphlo-opt %s -verify-diagnostics -split-input-file

// -----

func.func @invalid_concate_dim() -> tensor<!pphlo.pub<i32>> {
  %0 = "pphlo.constant"() {value = dense<1.3347515E+38> : tensor<f32>} : () -> tensor<!pphlo.pub<f32>>
  // expected-error @+1 {{rank-0 values cannot be concatenated}}
  %1 = "pphlo.concatenate"(%0) {dimension = 27755 : i64} : (tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
  %2 = "pphlo.constant"() {value = dense<5> : tensor<i32>} : () -> tensor<!pphlo.pub<i32>>
  "pphlo.return"(%2) : (tensor<!pphlo.pub<i32>>) -> ()
}

// -----

// -----

func.func @invalid_broadcast_dim() -> tensor<!pphlo.pub<i32>> {
  %2 = "pphlo.constant"() {value = dense<[0x41DA6E5887800000, 0x41C94E3940000000, 0x41C4BD2007000000, 0x41DC95133AC00000, 0x41D1650CEC000000, 0x41C9DF42E7800000, 0x41D46C43B6800000, 0x41C467EE0E800000, 0x41DC705F14400000]> : tensor<9xf64>} : () -> tensor<9x!pphlo.pub<f64>>
  %3 = "pphlo.floor"(%2) : (tensor<9x!pphlo.pub<f64>>) -> tensor<9x!pphlo.pub<f64>>
  %9 = "pphlo.concatenate"(%3) {dimension = 0 : i64} : (tensor<9x!pphlo.pub<f64>>) -> tensor<9x!pphlo.pub<f64>>
  // expected-error @+1 {{broadcast_dimensions contains invalid value 13 for result with rank 1}}
  %10 = "pphlo.broadcast"(%9) {broadcast_dimensions = dense<13> : tensor<1xi64>} : (tensor<9x!pphlo.pub<f64>>) -> tensor<9x!pphlo.pub<f64>>
  %51 = "pphlo.constant"() {value = dense<5> : tensor<i32>} : () -> tensor<!pphlo.pub<i32>>
  "pphlo.return"(%51) : (tensor<!pphlo.pub<i32>>) -> ()
}

// -----

// -----

func.func @negative_broadcast_dims() -> tensor<!pphlo.pub<i32>> {
  %0 = "pphlo.constant"() {value = dense<[0.000000e+00, -3.40282347E+38]> : tensor<2xf32>} : () -> tensor<2x!pphlo.pub<f32>>
  // expected-error @+1 {{op broadcast_dimensions contains invalid value -6 for result with rank 1}}
  %1 = "pphlo.broadcast"(%0) {broadcast_dimensions = dense<-6> : tensor<1xi64>} : (tensor<2x!pphlo.pub<f32>>) -> tensor<2x!pphlo.pub<f32>>
  %2 = "pphlo.constant"() {value = dense<5> : tensor<i32>} : () -> tensor<!pphlo.pub<i32>>
  "pphlo.return"(%2) : (tensor<!pphlo.pub<i32>>) -> ()
}

// -----

// -----

func.func @main() -> tensor<!pphlo.pub<i32>> {
  // expected-error @+1 {{op iota dimension cannot go beyond the output rank or be negative}}
  %0 = "pphlo.iota"() {iota_dimension = 1000 : i64} : () -> tensor<1x!pphlo.pub<i32>>
  %1 = "pphlo.constant"() {value = dense<5> : tensor<i32>} : () -> tensor<!pphlo.pub<i32>>
  "pphlo.return"(%1) : (tensor<!pphlo.pub<i32>>) -> ()
}

// -----
