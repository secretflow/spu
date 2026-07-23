// RUN: spu-opt %s -verify-diagnostics -split-input-file

func.func @main() -> tensor<i32> {
  %0 = pphlo.constant dense<1.3347515E+38> : tensor<f32>
  // expected-error @+1 {{rank-0 values cannot be concatenated}}
  %1 = pphlo.concatenate %0 dim = 27755 : (tensor<f32>) -> tensor<f32>
  %2 = pphlo.constant dense<5> : tensor<i32>
  pphlo.return %2 : tensor<i32>
}

// -----

func.func @main() -> tensor<i32> {
  %2 = pphlo.constant dense<[0x41DA6E5887800000, 0x41C94E3940000000, 0x41C4BD2007000000, 0x41DC95133AC00000, 0x41D1650CEC000000, 0x41C9DF42E7800000, 0x41D46C43B6800000, 0x41C467EE0E800000, 0x41DC705F14400000]> : tensor<9xf64>
  %3 = pphlo.floor %2 : tensor<9xf64>
  %9 = pphlo.concatenate %3 dim = 0 : (tensor<9xf64>) -> tensor<9xf64>
  // expected-error @+1 {{broadcast_dimensions contains invalid value 13 for result with rank 1}}
  %10 = pphlo.broadcast %9, dims = [13] : (tensor<9xf64>) -> tensor<9xf64>
  %51 = pphlo.constant dense<5> : tensor<i32>
  pphlo.return %51 : tensor<i32>
}

// -----

func.func @main() -> tensor<i32> {
  %0 = pphlo.constant dense<[0.000000e+00, -3.40282347E+38]> : tensor<2xf32>
  // expected-error @+1 {{broadcast_dimensions contains invalid value -6 for result with rank 1}}
  %1 = pphlo.broadcast %0, dims = [-6] : (tensor<2xf32>) -> tensor<2xf32>
  %2 = pphlo.constant dense<5> : tensor<i32>
  pphlo.return %2 : tensor<i32>
}

// -----

func.func @main() -> tensor<i32> {
  // expected-error @+1 {{iota dimension cannot go beyond the output rank}}
  %0 = pphlo.iota dim = 1000 : tensor<1xi32>
  %1 = pphlo.constant dense<5> : tensor<i32>
  pphlo.return %1 : tensor<i32>
}

// -----

func.func @main(%arg0: tensor<9x9x1x!pphlo.secret<f64>>) -> tensor<9x9x1x!pphlo.secret<f64>> {
  // expected-error @+1 {{op permutation -837266685 out of range [0, 2]}}
  %0 = pphlo.transpose %arg0, dims = [-837266685, -198653443088, -690803635863] : (tensor<9x9x1x!pphlo.secret<f64>>) -> tensor<9x9x1x!pphlo.secret<f64>>
  pphlo.return %0 : tensor<9x9x1x!pphlo.secret<f64>>
}

// -----

func.func @main(%arg0: tensor<9x1x!pphlo.secret<f64>>) -> tensor<9x1x!pphlo.secret<f32>> {
  // expected-error @+1 {{op requires the same element type for all operands and results}}
  %0 = pphlo.transpose %arg0, dims = [0, 1] : (tensor<9x1x!pphlo.secret<f64>>) -> tensor<9x1x!pphlo.secret<f32>>
  pphlo.return %0 : tensor<9x1x!pphlo.secret<f32>>
}

// -----

func.func @main(%arg0: tensor<9x1x!pphlo.secret<f64>>) -> tensor<9x1x!pphlo.secret<f64>> {
  // expected-error @+1 {{op shape mismatch input shape = 9x1, result shape = 9x1, permutation = 1x0}}
  %0 = pphlo.transpose %arg0, dims = [1, 0] : (tensor<9x1x!pphlo.secret<f64>>) -> tensor<9x1x!pphlo.secret<f64>>
  pphlo.return %0 : tensor<9x1x!pphlo.secret<f64>>
}

// -----

func.func @main(%arg0: tensor<9x9x1xf64>) -> tensor<9x9x1xf64> {
  // expected-error @+1 {{op all dimensions should be non-negative. Got dimension: -11917540144205.}}
  %0 = pphlo.reverse %arg0, dims = [-4367244318167, -11917540144205, -9774346241042] : tensor<9x9x1xf64>
  pphlo.return %0 : tensor<9x9x1xf64>
}

// -----

func.func @main(%arg0: tensor<9x9x1xf64>) -> tensor<9x9x1xf64> {
  // expected-error @+1 {{op all dimensions should be between [0, 3). Got dimension: 4367244339678518167.}}
  %0 = pphlo.reverse %arg0, dims = [4367244339678518167, 1191754011229144205, 977434623931441042] : tensor<9x9x1xf64>
  pphlo.return %0 : tensor<9x9x1xf64>
}

// -----

func.func @main(%arg0: tensor<9x9x1xf64>) -> tensor<9x9x1xf64> {
  // expected-error @+1 {{op dimensions are not unique}}
  %0 = pphlo.reverse %arg0, dims = [1,1,1] : tensor<9x9x1xf64>
  pphlo.return %0 : tensor<9x9x1xf64>
}

// -----

func.func @main(%arg0: tensor<10xi32>) -> (tensor<i32>) {
  %0 = pphlo.constant dense<0> : tensor<i32>
  // expected-error @+1 {{Out-of-bounds dimension -12233434 for input-tensor rank: 1}}
  %1 = pphlo.reduce(%arg0 init: %0) applies pphlo.add across dimensions = [-12233434] : (tensor<10xi32>, tensor<i32>) -> tensor<i32>
  return %1 :  tensor<i32>
}

// -----

func.func @main() -> tensor<i32> {
    %0 = pphlo.constant dense<-1.7976931344453863E+308> : tensor<1x1xf64>
    // expected-error @+1 {{op negative start index -92225398487041 in dimension 0}}
    %1 = pphlo.slice %0 [-92225398487041:-92205558487041:-92205598487041, 0:0:0]: (tensor<1x1xf64>) -> tensor<1x1xf64>
    %2 = pphlo.slice %0 [-850248339815911:-850208339815911:-850248339815911, -92233715411295:-92233715511295:-92233715411295] : (tensor<1x1xf64>) -> tensor<1x1xf64>
    %3 = pphlo.constant dense<5> : tensor<i32>
    pphlo.return %3 : tensor<i32>
  }
