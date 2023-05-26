// RUN: mlir-pphlo-opt %s -verify-diagnostics -split-input-file

func.func @main() -> tensor<!pphlo.pub<i32>> {
  %0 = "pphlo.constant"() {value = dense<1.3347515E+38> : tensor<f32>} : () -> tensor<!pphlo.pub<f32>>
  // expected-error @+1 {{rank-0 values cannot be concatenated}}
  %1 = "pphlo.concatenate"(%0) {dimension = 27755 : i64} : (tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
  %2 = "pphlo.constant"() {value = dense<5> : tensor<i32>} : () -> tensor<!pphlo.pub<i32>>
  "pphlo.return"(%2) : (tensor<!pphlo.pub<i32>>) -> ()
}

// -----

func.func @main() -> tensor<!pphlo.pub<i32>> {
  %2 = "pphlo.constant"() {value = dense<[0x41DA6E5887800000, 0x41C94E3940000000, 0x41C4BD2007000000, 0x41DC95133AC00000, 0x41D1650CEC000000, 0x41C9DF42E7800000, 0x41D46C43B6800000, 0x41C467EE0E800000, 0x41DC705F14400000]> : tensor<9xf64>} : () -> tensor<9x!pphlo.pub<f64>>
  %3 = "pphlo.floor"(%2) : (tensor<9x!pphlo.pub<f64>>) -> tensor<9x!pphlo.pub<f64>>
  %9 = "pphlo.concatenate"(%3) {dimension = 0 : i64} : (tensor<9x!pphlo.pub<f64>>) -> tensor<9x!pphlo.pub<f64>>
  // expected-error @+1 {{broadcast_dimensions contains invalid value 13 for result with rank 1}}
  %10 = "pphlo.broadcast"(%9) {broadcast_dimensions = dense<13> : tensor<1xi64>} : (tensor<9x!pphlo.pub<f64>>) -> tensor<9x!pphlo.pub<f64>>
  %51 = "pphlo.constant"() {value = dense<5> : tensor<i32>} : () -> tensor<!pphlo.pub<i32>>
  "pphlo.return"(%51) : (tensor<!pphlo.pub<i32>>) -> ()
}

// -----

func.func @main() -> tensor<!pphlo.pub<i32>> {
  %0 = "pphlo.constant"() {value = dense<[0.000000e+00, -3.40282347E+38]> : tensor<2xf32>} : () -> tensor<2x!pphlo.pub<f32>>
  // expected-error @+1 {{op broadcast_dimensions contains invalid value -6 for result with rank 1}}
  %1 = "pphlo.broadcast"(%0) {broadcast_dimensions = dense<-6> : tensor<1xi64>} : (tensor<2x!pphlo.pub<f32>>) -> tensor<2x!pphlo.pub<f32>>
  %2 = "pphlo.constant"() {value = dense<5> : tensor<i32>} : () -> tensor<!pphlo.pub<i32>>
  "pphlo.return"(%2) : (tensor<!pphlo.pub<i32>>) -> ()
}

// -----

func.func @main() -> tensor<!pphlo.pub<i32>> {
  // expected-error @+1 {{op iota dimension cannot go beyond the output rank or be negative}}
  %0 = "pphlo.iota"() {iota_dimension = 1000 : i64} : () -> tensor<1x!pphlo.pub<i32>>
  %1 = "pphlo.constant"() {value = dense<5> : tensor<i32>} : () -> tensor<!pphlo.pub<i32>>
  "pphlo.return"(%1) : (tensor<!pphlo.pub<i32>>) -> ()
}

// -----

func.func @main(%arg0: tensor<9x9x1x!pphlo.sec<f64>>) -> tensor<9x9x1x!pphlo.sec<f64>> {
  // expected-error @+1 {{op permutation -837266656812241085 out of range [0, 2]}}
  %0 = "pphlo.transpose"(%arg0) {permutation = dense<[-837266656812241085, -1986534498277253088, -6908486506403635863]> : tensor<3xi64>} : (tensor<9x9x1x!pphlo.sec<f64>>) -> tensor<9x9x1x!pphlo.sec<f64>>
  "pphlo.return"(%0) : (tensor<9x9x1x!pphlo.sec<f64>>) -> ()
}

// -----

func.func @main(%arg0: tensor<9x1x!pphlo.sec<f64>>) -> tensor<9x1x!pphlo.sec<f32>> {
  // expected-error @+1 {{op requires the same element type for all operands and results}}
  %0 = "pphlo.transpose"(%arg0) {permutation = dense<[0, 1]> : tensor<2xi64>} : (tensor<9x1x!pphlo.sec<f64>>) -> tensor<9x1x!pphlo.sec<f32>>
  "pphlo.return"(%0) : (tensor<9x1x!pphlo.sec<f32>>) -> ()
}

// -----

func.func @main(%arg0: tensor<9x1x!pphlo.sec<f64>>) -> tensor<9x1x!pphlo.sec<f64>> {
  // expected-error @+1 {{op shape mismatch input shape = 9x1, result shape = 9x1, permutation = 1x0}}
  %0 = "pphlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<9x1x!pphlo.sec<f64>>) -> tensor<9x1x!pphlo.sec<f64>>
  "pphlo.return"(%0) : (tensor<9x1x!pphlo.sec<f64>>) -> ()
}

// -----

func.func @main(%arg0: tensor<9x9x1x!pphlo.pub<f64>>) -> tensor<9x9x1x!pphlo.pub<f64>> {
  // expected-error @+1 {{op all dimensions should be non-negative. Got dimension: -1191754011229144205.}}
  %0 = "pphlo.reverse"(%arg0) {dimensions = dense<[-4367244339678518167, -1191754011229144205, -977434623931441042]> : tensor<3xi64>} : (tensor<9x9x1x!pphlo.pub<f64>>) -> tensor<9x9x1x!pphlo.pub<f64>>
  "pphlo.return"(%0) : (tensor<9x9x1x!pphlo.pub<f64>>) -> ()
}

// -----

func.func @main(%arg0: tensor<9x9x1x!pphlo.pub<f64>>) -> tensor<9x9x1x!pphlo.pub<f64>> {
  // expected-error @+1 {{op all dimensions should be between [0, 3). Got dimension: 4367244339678518167.}}
  %0 = "pphlo.reverse"(%arg0) {dimensions = dense<[4367244339678518167, 1191754011229144205, 977434623931441042]> : tensor<3xi64>} : (tensor<9x9x1x!pphlo.pub<f64>>) -> tensor<9x9x1x!pphlo.pub<f64>>
  "pphlo.return"(%0) : (tensor<9x9x1x!pphlo.pub<f64>>) -> ()
}

// -----

func.func @main(%arg0: tensor<9x9x1x!pphlo.pub<f64>>) -> tensor<9x9x1x!pphlo.pub<f64>> {
  // expected-error @+1 {{op dimensions are not unique}}
  %0 = "pphlo.reverse"(%arg0) {dimensions = dense<[1,1,1]> : tensor<3xi64>} : (tensor<9x9x1x!pphlo.pub<f64>>) -> tensor<9x9x1x!pphlo.pub<f64>>
  "pphlo.return"(%0) : (tensor<9x9x1x!pphlo.pub<f64>>) -> ()
}

// -----

func.func @main(%arg0: tensor<9x9x1x!pphlo.pub<f64>>) -> tensor<9x9x1x!pphlo.pub<f64>> {
  // expected-error @+1 {{op dimensions must be a 1-dimensional tensor}}
  %0 = "pphlo.reverse"(%arg0) {dimensions = dense<[[1,2],[3,4]]> : tensor<2x2xi64>} : (tensor<9x9x1x!pphlo.pub<f64>>) -> tensor<9x9x1x!pphlo.pub<f64>>
  "pphlo.return"(%0) : (tensor<9x9x1x!pphlo.pub<f64>>) -> ()
}

// -----

func.func @main(%arg0: tensor<10x!pphlo.pub<i32>>) -> (tensor<!pphlo.pub<i32>>) {
  %0 = "pphlo.constant"() {value = dense<0> : tensor<i32>} : () -> tensor<!pphlo.pub<i32>>
  // expected-error @+1 {{Out-of-bounds dimension -12233434 for input-tensor rank: 1}}
  %1 = "pphlo.reduce"(%arg0, %0) ( {
        ^bb0(%arg1: tensor<!pphlo.pub<i32>>, %arg2: tensor<!pphlo.pub<i32>>): // no predecessors
         %2 = "pphlo.add"(%arg1, %arg2) : (tensor<!pphlo.pub<i32>>, tensor<!pphlo.pub<i32>>) -> tensor<!pphlo.pub<i32>>
         "pphlo.return"(%2) : (tensor<!pphlo.pub<i32>>) -> ()
  }) {dimensions = dense<-12233434> : tensor<1xi64>} : (tensor<10x!pphlo.pub<i32>>, tensor<!pphlo.pub<i32>>) -> tensor<!pphlo.pub<i32>>
  return %1 :  tensor<!pphlo.pub<i32>>
}

// -----

func.func @main() -> tensor<!pphlo.pub<i32>> {
    %0 = "pphlo.constant"() {value = dense<127> : tensor<i8>} : () -> tensor<!pphlo.pub<i8>>
    %1 = "pphlo.slice"(%0) {limit_indices = dense<> : tensor<0xi64>, start_indices = dense<> : tensor<0xi64>, strides = dense<> : tensor<0xi64>} : (tensor<!pphlo.pub<i8>>) -> tensor<!pphlo.pub<i8>>
    %2 = "pphlo.slice"(%1) {limit_indices = dense<> : tensor<0xi64>, start_indices = dense<> : tensor<0xi64>, strides = dense<> : tensor<0xi64>} : (tensor<!pphlo.pub<i8>>) -> tensor<!pphlo.pub<i8>>
    %3 = "pphlo.slice"(%0) {limit_indices = dense<> : tensor<0xi64>, start_indices = dense<> : tensor<0xi64>, strides = dense<> : tensor<0xi64>} : (tensor<!pphlo.pub<i8>>) -> tensor<!pphlo.pub<i8>>
    %4 = "pphlo.constant"() {value = dense<-1.7976931344453863E+308> : tensor<1x1xf64>} : () -> tensor<1x1x!pphlo.pub<f64>>
    // expected-error @+1 {{op negative start index -9220555925398487041 in dimension 0}}
    %5 = "pphlo.slice"(%4) {limit_indices = dense<[-9220555925398487041, 0]> : tensor<2xi64>, start_indices = dense<[-9220555925398487041, 0]> : tensor<2xi64>, strides = dense<[-9220555925398487041, 0]> : tensor<2xi64>} : (tensor<1x1x!pphlo.pub<f64>>) -> tensor<1x1x!pphlo.pub<f64>>
    %6 = "pphlo.slice"(%4) {limit_indices = dense<[-8502447508339815911, -9223371558496411295]> : tensor<2xi64>, start_indices = dense<[-8502447508339815911, -9223371558496411295]> : tensor<2xi64>, strides = dense<[-8502447508339815911, -9223371558496411295]> : tensor<2xi64>} : (tensor<1x1x!pphlo.pub<f64>>) -> tensor<1x1x!pphlo.pub<f64>>
    %7 = "pphlo.constant"() {value = dense<5> : tensor<i32>} : () -> tensor<!pphlo.pub<i32>>
    "pphlo.return"(%7) : (tensor<!pphlo.pub<i32>>) -> ()
  }

