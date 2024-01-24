// RUN: mlir-pphlo-opt %s -verify-diagnostics -split-input-file

func.func @main() -> tensor<i32> {
  %0 = "pphlo.constant"() {value = dense<1.3347515E+38> : tensor<f32>} : () -> tensor<f32>
  // expected-error @+1 {{rank-0 values cannot be concatenated}}
  %1 = "pphlo.concatenate"(%0) {dimension = 27755 : i64} : (tensor<f32>) -> tensor<f32>
  %2 = "pphlo.constant"() {value = dense<5> : tensor<i32>} : () -> tensor<i32>
  "pphlo.return"(%2) : (tensor<i32>) -> ()
}

// -----

func.func @main() -> tensor<i32> {
  %2 = "pphlo.constant"() {value = dense<[0x41DA6E5887800000, 0x41C94E3940000000, 0x41C4BD2007000000, 0x41DC95133AC00000, 0x41D1650CEC000000, 0x41C9DF42E7800000, 0x41D46C43B6800000, 0x41C467EE0E800000, 0x41DC705F14400000]> : tensor<9xf64>} : () -> tensor<9xf64>
  %3 = "pphlo.floor"(%2) : (tensor<9xf64>) -> tensor<9xf64>
  %9 = "pphlo.concatenate"(%3) {dimension = 0 : i64} : (tensor<9xf64>) -> tensor<9xf64>
  // expected-error @+1 {{broadcast_dimensions contains invalid value 13 for result with rank 1}}
  %10 = "pphlo.broadcast"(%9) {broadcast_dimensions = array<i64: 13>} : (tensor<9xf64>) -> tensor<9xf64>
  %51 = "pphlo.constant"() {value = dense<5> : tensor<i32>} : () -> tensor<i32>
  "pphlo.return"(%51) : (tensor<i32>) -> ()
}

// -----

func.func @main() -> tensor<i32> {
  %0 = "pphlo.constant"() {value = dense<[0.000000e+00, -3.40282347E+38]> : tensor<2xf32>} : () -> tensor<2xf32>
  // expected-error @+1 {{op broadcast_dimensions contains invalid value -6 for result with rank 1}}
  %1 = "pphlo.broadcast"(%0) {broadcast_dimensions = array<i64: -6>} : (tensor<2xf32>) -> tensor<2xf32>
  %2 = "pphlo.constant"() {value = dense<5> : tensor<i32>} : () -> tensor<i32>
  "pphlo.return"(%2) : (tensor<i32>) -> ()
}

// -----

func.func @main() -> tensor<i32> {
  // expected-error @+1 {{op iota dimension cannot go beyond the output rank or be negative}}
  %0 = "pphlo.iota"() {iota_dimension = 1000 : i64} : () -> tensor<1xi32>
  %1 = "pphlo.constant"() {value = dense<5> : tensor<i32>} : () -> tensor<i32>
  "pphlo.return"(%1) : (tensor<i32>) -> ()
}

// -----

func.func @main(%arg0: tensor<9x9x1x!pphlo.secret<f64>>) -> tensor<9x9x1x!pphlo.secret<f64>> {
  // expected-error @+1 {{op permutation -837266656812241085 out of range [0, 2]}}
  %0 = "pphlo.transpose"(%arg0) {permutation = array<i64: -837266656812241085, -1986534498277253088, -6908486506403635863>} : (tensor<9x9x1x!pphlo.secret<f64>>) -> tensor<9x9x1x!pphlo.secret<f64>>
  "pphlo.return"(%0) : (tensor<9x9x1x!pphlo.secret<f64>>) -> ()
}

// -----

func.func @main(%arg0: tensor<9x1x!pphlo.secret<f64>>) -> tensor<9x1x!pphlo.secret<f32>> {
  // expected-error @+1 {{op requires the same element type for all operands and results}}
  %0 = "pphlo.transpose"(%arg0) {permutation = array<i64: 0, 1>} : (tensor<9x1x!pphlo.secret<f64>>) -> tensor<9x1x!pphlo.secret<f32>>
  "pphlo.return"(%0) : (tensor<9x1x!pphlo.secret<f32>>) -> ()
}

// -----

func.func @main(%arg0: tensor<9x1x!pphlo.secret<f64>>) -> tensor<9x1x!pphlo.secret<f64>> {
  // expected-error @+1 {{op shape mismatch input shape = 9x1, result shape = 9x1, permutation = 1x0}}
  %0 = "pphlo.transpose"(%arg0) {permutation = array<i64: 1, 0>} : (tensor<9x1x!pphlo.secret<f64>>) -> tensor<9x1x!pphlo.secret<f64>>
  "pphlo.return"(%0) : (tensor<9x1x!pphlo.secret<f64>>) -> ()
}

// -----

func.func @main(%arg0: tensor<9x9x1xf64>) -> tensor<9x9x1xf64> {
  // expected-error @+1 {{op all dimensions should be non-negative. Got dimension: -1191754011229144205.}}
  %0 = "pphlo.reverse"(%arg0) {dimensions = array<i64: -4367244339678518167, -1191754011229144205, -977434623931441042>} : (tensor<9x9x1xf64>) -> tensor<9x9x1xf64>
  "pphlo.return"(%0) : (tensor<9x9x1xf64>) -> ()
}

// -----

func.func @main(%arg0: tensor<9x9x1xf64>) -> tensor<9x9x1xf64> {
  // expected-error @+1 {{op all dimensions should be between [0, 3). Got dimension: 4367244339678518167.}}
  %0 = "pphlo.reverse"(%arg0) {dimensions = array<i64: 4367244339678518167, 1191754011229144205, 977434623931441042>} : (tensor<9x9x1xf64>) -> tensor<9x9x1xf64>
  "pphlo.return"(%0) : (tensor<9x9x1xf64>) -> ()
}

// -----

func.func @main(%arg0: tensor<9x9x1xf64>) -> tensor<9x9x1xf64> {
  // expected-error @+1 {{op dimensions are not unique}}
  %0 = "pphlo.reverse"(%arg0) {dimensions = array<i64: 1,1,1>} : (tensor<9x9x1xf64>) -> tensor<9x9x1xf64>
  "pphlo.return"(%0) : (tensor<9x9x1xf64>) -> ()
}

// -----

func.func @main(%arg0: tensor<10xi32>) -> (tensor<i32>) {
  %0 = "pphlo.constant"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  // expected-error @+1 {{Out-of-bounds dimension -12233434 for input-tensor rank: 1}}
  %1 = "pphlo.reduce"(%arg0, %0) ( {
        ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>): // no predecessors
         %2 = "pphlo.add"(%arg1, %arg2) : (tensor<i32>, tensor<i32>) -> tensor<i32>
         "pphlo.return"(%2) : (tensor<i32>) -> ()
  }) {dimensions = array<i64: -12233434>} : (tensor<10xi32>, tensor<i32>) -> tensor<i32>
  return %1 :  tensor<i32>
}

// -----

func.func @main() -> tensor<i32> {
    %0 = "pphlo.constant"() {value = dense<-1.7976931344453863E+308> : tensor<1x1xf64>} : () -> tensor<1x1xf64>
    // expected-error @+1 {{op negative start index -9220555925398487041 in dimension 0}}
    %1 = "pphlo.slice"(%0) {limit_indices = array<i64: -9220555925398487041, 0>, start_indices = array<i64: -9220555925398487041, 0>, strides = array<i64: -9220555925398487041, 0>} : (tensor<1x1xf64>) -> tensor<1x1xf64>
    %2 = "pphlo.slice"(%0) {limit_indices = array<i64: -8502447508339815911, -9223371558496411295>, start_indices = array<i64: -8502447508339815911, -9223371558496411295>, strides = array<i64: -8502447508339815911, -9223371558496411295>} : (tensor<1x1xf64>) -> tensor<1x1xf64>
    %3 = "pphlo.constant"() {value = dense<5> : tensor<i32>} : () -> tensor<i32>
    "pphlo.return"(%3) : (tensor<i32>) -> ()
  }

