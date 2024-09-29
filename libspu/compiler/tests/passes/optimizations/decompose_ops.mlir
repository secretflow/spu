// RUN: spu-opt --general-decompose-ops --secret-decompose-ops --general-decompose-ops --canonicalize --cse --split-input-file %s | FileCheck %s

func.func @main(%arg0: tensor<2x2x!pphlo.secret<i32>>, %arg1: tensor<2x2x!pphlo.secret<i32>>) -> (tensor<2x2x!pphlo.secret<i1>>) {
    //CHECK: %0 = pphlo.equal %arg0, %arg1 : (tensor<2x2x!pphlo.secret<i32>>, tensor<2x2x!pphlo.secret<i32>>) -> tensor<2x2x!pphlo.secret<i1>>
    //CHECK: %1 = pphlo.not %0 : tensor<2x2x!pphlo.secret<i1>>
    %0 = pphlo.not_equal %arg0, %arg1 : (tensor<2x2x!pphlo.secret<i32>>, tensor<2x2x!pphlo.secret<i32>>) -> tensor<2x2x!pphlo.secret<i1>>
    return %0 : tensor<2x2x!pphlo.secret<i1>>
}

// -----

func.func @main(%arg0: tensor<2x2x!pphlo.secret<i32>>, %arg1: tensor<2x2x!pphlo.secret<i32>>) -> (tensor<2x2x!pphlo.secret<i1>>) {
    //CHECK: %0 = pphlo.less %arg1, %arg0 : (tensor<2x2x!pphlo.secret<i32>>, tensor<2x2x!pphlo.secret<i32>>) -> tensor<2x2x!pphlo.secret<i1>>
    //CHECK: %1 = pphlo.not %0 : tensor<2x2x!pphlo.secret<i1>>
    %0 = pphlo.less_equal %arg0, %arg1 : (tensor<2x2x!pphlo.secret<i32>>, tensor<2x2x!pphlo.secret<i32>>) -> tensor<2x2x!pphlo.secret<i1>>
    return %0 : tensor<2x2x!pphlo.secret<i1>>
}

// -----

func.func @main(%arg0: tensor<2x2x!pphlo.secret<i32>>, %arg1: tensor<2x2x!pphlo.secret<i32>>) -> (tensor<2x2x!pphlo.secret<i1>>) {
    // CHECK: %0 = pphlo.less %arg0, %arg1 : (tensor<2x2x!pphlo.secret<i32>>, tensor<2x2x!pphlo.secret<i32>>) -> tensor<2x2x!pphlo.secret<i1>>
    // CHECK: %1 = pphlo.not %0 : tensor<2x2x!pphlo.secret<i1>>
    %0 = pphlo.greater_equal %arg0, %arg1 : (tensor<2x2x!pphlo.secret<i32>>, tensor<2x2x!pphlo.secret<i32>>) -> tensor<2x2x!pphlo.secret<i1>>
    return %0 : tensor<2x2x!pphlo.secret<i1>>
}

// -----

func.func @main(%arg0: tensor<2x2x!pphlo.secret<i32>>, %arg1: tensor<2x2x!pphlo.secret<i32>>) -> (tensor<2x2x!pphlo.secret<i1>>, tensor<2x2x!pphlo.secret<i1>>, tensor<2x2x!pphlo.secret<i1>>, tensor<2x2x!pphlo.secret<i1>>, tensor<2x2x!pphlo.secret<i1>>, tensor<2x2x!pphlo.secret<i1>>) {
    //CHECK: %0 = pphlo.equal %arg0, %arg1 : (tensor<2x2x!pphlo.secret<i32>>, tensor<2x2x!pphlo.secret<i32>>) -> tensor<2x2x!pphlo.secret<i1>>
    //CHECK: %1 = pphlo.not %0 : tensor<2x2x!pphlo.secret<i1>>
    //CHECK: %2 = pphlo.less %arg0, %arg1 : (tensor<2x2x!pphlo.secret<i32>>, tensor<2x2x!pphlo.secret<i32>>) -> tensor<2x2x!pphlo.secret<i1>>
    //CHECK: %3 = pphlo.less %arg1, %arg0 : (tensor<2x2x!pphlo.secret<i32>>, tensor<2x2x!pphlo.secret<i32>>) -> tensor<2x2x!pphlo.secret<i1>>
    //CHECK: %4 = pphlo.not %3 : tensor<2x2x!pphlo.secret<i1>>
    //CHECK: %5 = pphlo.not %2 : tensor<2x2x!pphlo.secret<i1>>
    %0 = pphlo.equal %arg0, %arg1 : (tensor<2x2x!pphlo.secret<i32>>, tensor<2x2x!pphlo.secret<i32>>) -> tensor<2x2x!pphlo.secret<i1>>
    %1 = pphlo.not_equal %arg0, %arg1 : (tensor<2x2x!pphlo.secret<i32>>, tensor<2x2x!pphlo.secret<i32>>) -> tensor<2x2x!pphlo.secret<i1>>
    %2 = pphlo.less %arg0, %arg1 : (tensor<2x2x!pphlo.secret<i32>>, tensor<2x2x!pphlo.secret<i32>>) -> tensor<2x2x!pphlo.secret<i1>>
    %3 = pphlo.greater %arg0, %arg1 : (tensor<2x2x!pphlo.secret<i32>>, tensor<2x2x!pphlo.secret<i32>>) -> tensor<2x2x!pphlo.secret<i1>>
    %4 = pphlo.less_equal %arg0, %arg1 : (tensor<2x2x!pphlo.secret<i32>>, tensor<2x2x!pphlo.secret<i32>>) -> tensor<2x2x!pphlo.secret<i1>>
    %5 = pphlo.greater_equal %arg0, %arg1 : (tensor<2x2x!pphlo.secret<i32>>, tensor<2x2x!pphlo.secret<i32>>) -> tensor<2x2x!pphlo.secret<i1>>
    return %0, %1, %2, %3, %4, %5 : tensor<2x2x!pphlo.secret<i1>>, tensor<2x2x!pphlo.secret<i1>>, tensor<2x2x!pphlo.secret<i1>>, tensor<2x2x!pphlo.secret<i1>>, tensor<2x2x!pphlo.secret<i1>>, tensor<2x2x!pphlo.secret<i1>>
  }

// -----

func.func @main(%arg0: tensor<2x2x!pphlo.secret<f32>>, %arg1: tensor<2x2x!pphlo.secret<f32>>) -> (tensor<2x2x!pphlo.secret<f32>>) {
    //CHECK: %0 = pphlo.less %arg1, %arg0 : (tensor<2x2x!pphlo.secret<f32>>, tensor<2x2x!pphlo.secret<f32>>) -> tensor<2x2x!pphlo.secret<i1>>
    //CHECK: %1 = pphlo.select %0, %arg0, %arg1 : (tensor<2x2x!pphlo.secret<i1>>, tensor<2x2x!pphlo.secret<f32>>, tensor<2x2x!pphlo.secret<f32>>) -> tensor<2x2x!pphlo.secret<f32>>
    %0 = pphlo.maximum %arg0, %arg1 : tensor<2x2x!pphlo.secret<f32>>
    return %0 : tensor<2x2x!pphlo.secret<f32>>
}

// -----

func.func @main(%arg0: tensor<2x2x!pphlo.secret<f32>>, %arg1: tensor<2x2x!pphlo.secret<f32>>) -> (tensor<2x2x!pphlo.secret<f32>>) {
    //CHECK: %0 = pphlo.less %arg0, %arg1 : (tensor<2x2x!pphlo.secret<f32>>, tensor<2x2x!pphlo.secret<f32>>) -> tensor<2x2x!pphlo.secret<i1>>
    //CHECK: %1 = pphlo.select %0, %arg0, %arg1 : (tensor<2x2x!pphlo.secret<i1>>, tensor<2x2x!pphlo.secret<f32>>, tensor<2x2x!pphlo.secret<f32>>) -> tensor<2x2x!pphlo.secret<f32>>
    %0 = pphlo.minimum %arg0, %arg1 : tensor<2x2x!pphlo.secret<f32>>
    return %0 : tensor<2x2x!pphlo.secret<f32>>
}

// -----

func.func @main(%arg0: tensor<2x2x!pphlo.secret<f32>>, %arg1: tensor<2x2x!pphlo.secret<f32>>, %arg2: tensor<2x2x!pphlo.secret<f32>>) -> (tensor<2x2x!pphlo.secret<f32>>) {
    //CHECK: %0 = pphlo.less %arg1, %arg0 : (tensor<2x2x!pphlo.secret<f32>>, tensor<2x2x!pphlo.secret<f32>>) -> tensor<2x2x!pphlo.secret<i1>>
    //CHECK: %1 = pphlo.select %0, %arg0, %arg1 : (tensor<2x2x!pphlo.secret<i1>>, tensor<2x2x!pphlo.secret<f32>>, tensor<2x2x!pphlo.secret<f32>>) -> tensor<2x2x!pphlo.secret<f32>>
    //CHECK: %2 = pphlo.less %1, %arg2 : (tensor<2x2x!pphlo.secret<f32>>, tensor<2x2x!pphlo.secret<f32>>) -> tensor<2x2x!pphlo.secret<i1>>
    //CHECK: %3 = pphlo.select %2, %1, %arg2 : (tensor<2x2x!pphlo.secret<i1>>, tensor<2x2x!pphlo.secret<f32>>, tensor<2x2x!pphlo.secret<f32>>) -> tensor<2x2x!pphlo.secret<f32>>
    //CHECK: return %3 : tensor<2x2x!pphlo.secret<f32>>
    %0 = pphlo.clamp %arg0, %arg1, %arg2 : tensor<2x2x!pphlo.secret<f32>>
    return %0 : tensor<2x2x!pphlo.secret<f32>>
}

// -----

func.func @main(%arg0: tensor<2x2x!pphlo.secret<i32>>, %arg1: tensor<2x2x!pphlo.secret<i32>>) -> (tensor<2x2x!pphlo.secret<i32>>) {
    //CHECK: %0 = pphlo.sign %arg0 {ignore_zero = true} : tensor<2x2x!pphlo.secret<i32>>
    //CHECK: %1 = pphlo.sign %arg1 {ignore_zero = true} : tensor<2x2x!pphlo.secret<i32>>
    //CHECK: %2 = pphlo.multiply %arg0, %0 : tensor<2x2x!pphlo.secret<i32>>
    //CHECK: %3 = pphlo.multiply %arg1, %1 : tensor<2x2x!pphlo.secret<i32>>
    //CHECK: %4 = pphlo.convert %2 : (tensor<2x2x!pphlo.secret<i32>>) -> tensor<2x2x!pphlo.secret<f32>>
    //CHECK: %5 = pphlo.convert %3 : (tensor<2x2x!pphlo.secret<i32>>) -> tensor<2x2x!pphlo.secret<f32>>
    //CHECK: %6 = pphlo.divide %4, %5 : tensor<2x2x!pphlo.secret<f32>>
    //CHECK: %7 = pphlo.convert %6 : (tensor<2x2x!pphlo.secret<f32>>) -> tensor<2x2x!pphlo.secret<i32>>
    //CHECK: %8 = pphlo.multiply %3, %7 : tensor<2x2x!pphlo.secret<i32>>
    //CHECK: %9 = pphlo.add %8, %3 : tensor<2x2x!pphlo.secret<i32>>
    //CHECK: %10 = pphlo.less %2, %9 : (tensor<2x2x!pphlo.secret<i32>>, tensor<2x2x!pphlo.secret<i32>>) -> tensor<2x2x!pphlo.secret<i1>>
    //CHECK: %11 = pphlo.not %10 : tensor<2x2x!pphlo.secret<i1>>
    //CHECK: %12 = pphlo.less %2, %8 : (tensor<2x2x!pphlo.secret<i32>>, tensor<2x2x!pphlo.secret<i32>>) -> tensor<2x2x!pphlo.secret<i1>>
    //CHECK: %13 = pphlo.convert %11 : (tensor<2x2x!pphlo.secret<i1>>) -> tensor<2x2x!pphlo.secret<i32>>
    //CHECK: %14 = pphlo.convert %12 : (tensor<2x2x!pphlo.secret<i1>>) -> tensor<2x2x!pphlo.secret<i32>>
    //CHECK: %15 = pphlo.add %7, %13 : tensor<2x2x!pphlo.secret<i32>>
    //CHECK: %16 = pphlo.subtract %15, %14 : tensor<2x2x!pphlo.secret<i32>>
    //CHECK: %17 = pphlo.multiply %0, %1 : tensor<2x2x!pphlo.secret<i32>>
    //CHECK: %18 = pphlo.multiply %16, %17 : tensor<2x2x!pphlo.secret<i32>>
    //CHECK: return %18 : tensor<2x2x!pphlo.secret<i32>>
    %0 = pphlo.divide %arg0, %arg1 : tensor<2x2x!pphlo.secret<i32>>
    return %0 : tensor<2x2x!pphlo.secret<i32>>
}

// -----

func.func @main(%arg0: tensor<2x2x!pphlo.secret<f32>>) -> (tensor<2x2x!pphlo.secret<f32>>) {
    //CHECK: %cst = arith.constant dense<1.000000e+00> : tensor<2x2xf32>
    //CHECK: %0 = pphlo.epsilon : tensor<2x2xf32>
    //CHECK: %1 = arith.subf %cst, %0 : tensor<2x2xf32>
    //CHECK: %2 = pphlo.add %arg0, %1 : (tensor<2x2x!pphlo.secret<f32>>, tensor<2x2xf32>) -> tensor<2x2x!pphlo.secret<f32>>
    //CHECK: %3 = pphlo.floor %2 : tensor<2x2x!pphlo.secret<f32>>
    //CHECK: return %3 : tensor<2x2x!pphlo.secret<f32>>
    %0 = pphlo.ceil %arg0 : tensor<2x2x!pphlo.secret<f32>>
    return %0 : tensor<2x2x!pphlo.secret<f32>>
}

// -----

func.func @main(%arg0: tensor<2x2x!pphlo.secret<f32>>) -> (tensor<2x2x!pphlo.secret<f32>>) {
    //CHECK: %cst = arith.constant dense<5.000000e-01> : tensor<2x2xf32>
    //CHECK: %0 = pphlo.sign %arg0 {ignore_zero = true} : tensor<2x2x!pphlo.secret<f32>>
    //CHECK: %1 = pphlo.multiply %0, %cst : (tensor<2x2x!pphlo.secret<f32>>, tensor<2x2xf32>) -> tensor<2x2x!pphlo.secret<f32>>
    //CHECK: %2 = pphlo.add %arg0, %1 : tensor<2x2x!pphlo.secret<f32>>
    //CHECK: %3 = pphlo.convert %2 : (tensor<2x2x!pphlo.secret<f32>>) -> tensor<2x2x!pphlo.secret<i32>>
    //CHECK: %4 = pphlo.convert %3 : (tensor<2x2x!pphlo.secret<i32>>) -> tensor<2x2x!pphlo.secret<f32>>
    //CHECK: return %4 : tensor<2x2x!pphlo.secret<f32>>
    %0 = pphlo.round_nearest_afz %arg0 : tensor<2x2x!pphlo.secret<f32>>
    return %0 : tensor<2x2x!pphlo.secret<f32>>
}

// -----

func.func @main(%arg0: tensor<2x2x!pphlo.secret<f32>>) -> (tensor<2x2x!pphlo.secret<f32>>) {
    //CHECK: %0 = pphlo.sign %arg0 {ignore_zero = true} : tensor<2x2x!pphlo.secret<f32>>
    //CHECK: %1 = pphlo.multiply %0, %arg0 : tensor<2x2x!pphlo.secret<f32>>
    //CHECK: return %1 : tensor<2x2x!pphlo.secret<f32>>
    %0 = pphlo.abs %arg0 : tensor<2x2x!pphlo.secret<f32>>
    return %0 : tensor<2x2x!pphlo.secret<f32>>
}


// -----

func.func @main(%arg0: tensor<2x2x!pphlo.secret<i32>>, %arg1: tensor<2x2x!pphlo.secret<i32>>) -> (tensor<2x2x!pphlo.secret<i32>>) {
    //CHECK: %0 = pphlo.sign %arg0 {ignore_zero = true} : tensor<2x2x!pphlo.secret<i32>>
    //CHECK: %1 = pphlo.sign %arg1 {ignore_zero = true} : tensor<2x2x!pphlo.secret<i32>>
    //CHECK: %2 = pphlo.multiply %arg0, %0 : tensor<2x2x!pphlo.secret<i32>>
    //CHECK: %3 = pphlo.multiply %arg1, %1 : tensor<2x2x!pphlo.secret<i32>>
    //CHECK: %4 = pphlo.convert %2 : (tensor<2x2x!pphlo.secret<i32>>) -> tensor<2x2x!pphlo.secret<f32>>
    //CHECK: %5 = pphlo.convert %3 : (tensor<2x2x!pphlo.secret<i32>>) -> tensor<2x2x!pphlo.secret<f32>>
    //CHECK: %6 = pphlo.divide %4, %5 : tensor<2x2x!pphlo.secret<f32>>
    //CHECK: %7 = pphlo.convert %6 : (tensor<2x2x!pphlo.secret<f32>>) -> tensor<2x2x!pphlo.secret<i32>>
    //CHECK: %8 = pphlo.multiply %3, %7 : tensor<2x2x!pphlo.secret<i32>>
    //CHECK: %9 = pphlo.add %8, %3 : tensor<2x2x!pphlo.secret<i32>>
    //CHECK: %10 = pphlo.less %2, %9 : (tensor<2x2x!pphlo.secret<i32>>, tensor<2x2x!pphlo.secret<i32>>) -> tensor<2x2x!pphlo.secret<i1>>
    //CHECK: %11 = pphlo.not %10 : tensor<2x2x!pphlo.secret<i1>>
    //CHECK: %12 = pphlo.less %2, %8 : (tensor<2x2x!pphlo.secret<i32>>, tensor<2x2x!pphlo.secret<i32>>) -> tensor<2x2x!pphlo.secret<i1>>
    //CHECK: %13 = pphlo.convert %11 : (tensor<2x2x!pphlo.secret<i1>>) -> tensor<2x2x!pphlo.secret<i32>>
    //CHECK: %14 = pphlo.convert %12 : (tensor<2x2x!pphlo.secret<i1>>) -> tensor<2x2x!pphlo.secret<i32>>
    //CHECK: %15 = pphlo.add %7, %13 : tensor<2x2x!pphlo.secret<i32>>
    //CHECK: %16 = pphlo.subtract %15, %14 : tensor<2x2x!pphlo.secret<i32>>
    //CHECK: %17 = pphlo.multiply %0, %1 : tensor<2x2x!pphlo.secret<i32>>
    //CHECK: %18 = pphlo.multiply %16, %17 : tensor<2x2x!pphlo.secret<i32>>
    //CHECK: %19 = pphlo.multiply %18, %arg1 : tensor<2x2x!pphlo.secret<i32>>
    //CHECK: %20 = pphlo.subtract %arg0, %19 : tensor<2x2x!pphlo.secret<i32>>
    //CHECK: return %20 : tensor<2x2x!pphlo.secret<i32>>
    %0 = pphlo.remainder %arg0, %arg1 : tensor<2x2x!pphlo.secret<i32>>
    return %0 : tensor<2x2x!pphlo.secret<i32>>
}

// -----

func.func @main(%arg0: tensor<2x2xui32>, %arg1: tensor<2x2xui32>) -> (tensor<2x2xui32>) {
    //CHECK: %0 = pphlo.convert %arg0 : (tensor<2x2xui32>) -> tensor<2x2xf32>
    //CHECK: %1 = pphlo.convert %arg1 : (tensor<2x2xui32>) -> tensor<2x2xf32>
    //CHECK: %2 = pphlo.divide %0, %1 : tensor<2x2xf32>
    //CHECK: %3 = pphlo.convert %2 : (tensor<2x2xf32>) -> tensor<2x2xui32>
    //CHECK: %4 = pphlo.multiply %arg1, %3 : tensor<2x2xui32>
    //CHECK: %5 = pphlo.add %4, %arg1 : tensor<2x2xui32>
    //CHECK: %6 = pphlo.less %arg0, %5 : (tensor<2x2xui32>, tensor<2x2xui32>) -> tensor<2x2xi1>
    //CHECK: %7 = pphlo.not %6 : tensor<2x2xi1>
    //CHECK: %8 = pphlo.less %arg0, %4 : (tensor<2x2xui32>, tensor<2x2xui32>) -> tensor<2x2xi1>
    //CHECK: %9 = pphlo.convert %7 : (tensor<2x2xi1>) -> tensor<2x2xui32>
    //CHECK: %10 = pphlo.convert %8 : (tensor<2x2xi1>) -> tensor<2x2xui32>
    //CHECK: %11 = pphlo.add %3, %9 : tensor<2x2xui32>
    //CHECK: %12 = pphlo.subtract %11, %10 : tensor<2x2xui32>
    //CHECK: %13 = pphlo.multiply %12, %arg1 : tensor<2x2xui32>
    //CHECK: %14 = pphlo.subtract %arg0, %13 : tensor<2x2xui32>
    //CHECK: return %14 : tensor<2x2xui32>
    %0 = pphlo.remainder %arg0, %arg1 : tensor<2x2xui32>
    return %0 : tensor<2x2xui32>
}

// -----

func.func @main(%arg0: tensor<2x2x!pphlo.secret<f32>>, %arg1: tensor<2x2x!pphlo.secret<f32>>) -> (tensor<2x2x!pphlo.secret<f32>>) {
    //CHECK: %cst = arith.constant dense<1.000000e+00> : tensor<2x2xf32>
    //CHECK: %cst_0 = arith.constant dense<0.000000e+00> : tensor<2x2xf32>
    //CHECK: %0 = pphlo.divide %arg0, %arg1 : tensor<2x2x!pphlo.secret<f32>>
    //CHECK: %1 = pphlo.less %0, %cst_0 : (tensor<2x2x!pphlo.secret<f32>>, tensor<2x2xf32>) -> tensor<2x2x!pphlo.secret<i1>>
    //CHECK: %2 = pphlo.floor %0 : tensor<2x2x!pphlo.secret<f32>>
    //CHECK: %3 = pphlo.epsilon : tensor<2x2xf32>
    //CHECK: %4 = arith.subf %cst, %3 : tensor<2x2xf32>
    //CHECK: %5 = pphlo.add %0, %4 : (tensor<2x2x!pphlo.secret<f32>>, tensor<2x2xf32>) -> tensor<2x2x!pphlo.secret<f32>>
    //CHECK: %6 = pphlo.floor %5 : tensor<2x2x!pphlo.secret<f32>>
    //CHECK: %7 = pphlo.select %1, %6, %2 : (tensor<2x2x!pphlo.secret<i1>>, tensor<2x2x!pphlo.secret<f32>>, tensor<2x2x!pphlo.secret<f32>>) -> tensor<2x2x!pphlo.secret<f32>>
    //CHECK: %8 = pphlo.multiply %7, %arg1 : tensor<2x2x!pphlo.secret<f32>>
    //CHECK: %9 = pphlo.subtract %arg0, %8 : tensor<2x2x!pphlo.secret<f32>>
    //CHECK: return %9 : tensor<2x2x!pphlo.secret<f32>>
    %0 = pphlo.remainder %arg0, %arg1 : tensor<2x2x!pphlo.secret<f32>>
    return %0 : tensor<2x2x!pphlo.secret<f32>>
}

// -----

func.func @main(%arg0: tensor<5x4xi64>, %arg1: tensor<i64>) -> (tensor<7x5xi64>) {
    //CHECK: %0 = pphlo.pad %arg0, %arg1, low = [1, 0], high = [1, 1], interior = [0, 1] : (tensor<5x4xi64>, tensor<i64>) -> tensor<7x8xi64>
    //CHECK: %1 = pphlo.slice %0 [0:1:7, 1:1:6] : (tensor<7x8xi64>) -> tensor<7x5xi64>
    %result = pphlo.pad %arg0, %arg1, low = [1, -1], high = [1, -1], interior = [0, 1]
        : (tensor<5x4xi64>, tensor<i64>) -> tensor<7x5xi64>
    return %result : tensor<7x5xi64>
}
