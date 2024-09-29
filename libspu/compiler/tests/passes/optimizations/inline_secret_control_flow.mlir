// RUN: spu-opt --inline-secret-control-flow --split-input-file %s | FileCheck %s

func.func @main(%arg0: tensor<!pphlo.secret<f32>>, %arg1: tensor<f32>) -> tensor<!pphlo.secret<f32>> {
    //CHECK-NOT: pphlo.if
    //CHECK: %cst = arith.constant dense<1.000000e+01> : tensor<f32>
    //CHECK: %0 = pphlo.less %arg0, %cst : (tensor<!pphlo.secret<f32>>, tensor<f32>) -> tensor<!pphlo.secret<i1>>
    //CHECK: %1 = pphlo.add %arg1, %arg1 : tensor<f32>
    //CHECK: %2 = pphlo.convert %1 : (tensor<f32>) -> tensor<!pphlo.secret<f32>>
    //CHECK: %3 = pphlo.exponential %arg1 : tensor<f32>
    //CHECK: %4 = pphlo.convert %3 : (tensor<f32>) -> tensor<!pphlo.secret<f32>>
    //CHECK: %5 = pphlo.select %0, %2, %4 : (tensor<!pphlo.secret<i1>>, tensor<!pphlo.secret<f32>>, tensor<!pphlo.secret<f32>>) -> tensor<!pphlo.secret<f32>>
    //CHECK: return %5 : tensor<!pphlo.secret<f32>>
    %0 = arith.constant dense<1.000000e+01> : tensor<f32>
    %1 = pphlo.less %arg0, %0 : (tensor<!pphlo.secret<f32>>, tensor<f32>) -> tensor<!pphlo.secret<i1>>
    %2 = "pphlo.if"(%1) ({
      %3 = pphlo.add %arg1, %arg1 : tensor<f32>
      %4 = pphlo.convert %3 : (tensor<f32>) -> tensor<!pphlo.secret<f32>>
      pphlo.return %4 : tensor<!pphlo.secret<f32>>
    }, {
      %3 = pphlo.exponential %arg1 : tensor<f32>
      %4 = pphlo.convert %3 : (tensor<f32>) -> tensor<!pphlo.secret<f32>>
      pphlo.return %4 : tensor<!pphlo.secret<f32>>
    }) : (tensor<!pphlo.secret<i1>>) -> tensor<!pphlo.secret<f32>>
    return %2 : tensor<!pphlo.secret<f32>>
}

// -----

func.func @main(%arg0: tensor<!pphlo.secret<f32>>, %arg1: tensor<f32>) -> (tensor<!pphlo.secret<f32>>,tensor<!pphlo.secret<f32>>) {
    //CHECK-NOT: pphlo.if
    //CHECK: %cst = arith.constant dense<1.000000e+01> : tensor<f32>
    //CHECK: %0 = pphlo.less %arg0, %cst : (tensor<!pphlo.secret<f32>>, tensor<f32>) -> tensor<!pphlo.secret<i1>>
    //CHECK: %1 = pphlo.add %arg1, %arg1 : tensor<f32>
    //CHECK: %2 = pphlo.subtract %arg1, %arg1 : tensor<f32>
    //CHECK: %3 = pphlo.convert %1 : (tensor<f32>) -> tensor<!pphlo.secret<f32>>
    //CHECK: %4 = pphlo.convert %2 : (tensor<f32>) -> tensor<!pphlo.secret<f32>>
    //CHECK: %5 = pphlo.exponential %arg1 : tensor<f32>
    //CHECK: %6 = pphlo.log %arg1 : tensor<f32>
    //CHECK: %7 = pphlo.convert %5 : (tensor<f32>) -> tensor<!pphlo.secret<f32>>
    //CHECK: %8 = pphlo.convert %6 : (tensor<f32>) -> tensor<!pphlo.secret<f32>>
    //CHECK: %9 = pphlo.select %0, %3, %7 : (tensor<!pphlo.secret<i1>>, tensor<!pphlo.secret<f32>>, tensor<!pphlo.secret<f32>>) -> tensor<!pphlo.secret<f32>>
    //CHECK: %10 = pphlo.select %0, %4, %8 : (tensor<!pphlo.secret<i1>>, tensor<!pphlo.secret<f32>>, tensor<!pphlo.secret<f32>>) -> tensor<!pphlo.secret<f32>>
    //CHECK: return %9, %10 : tensor<!pphlo.secret<f32>>, tensor<!pphlo.secret<f32>>
    %0 = arith.constant dense<1.000000e+01> : tensor<f32>
    %1 = pphlo.less %arg0, %0 : (tensor<!pphlo.secret<f32>>, tensor<f32>) -> tensor<!pphlo.secret<i1>>
    %result0, %result1 = "pphlo.if"(%1) ({
      %4 = pphlo.add %arg1, %arg1 : tensor<f32>
      %5 = pphlo.subtract %arg1, %arg1 : tensor<f32>
      %6 = pphlo.convert %4 : (tensor<f32>) -> tensor<!pphlo.secret<f32>>
      %7 = pphlo.convert %5 : (tensor<f32>) -> tensor<!pphlo.secret<f32>>
      pphlo.return %6, %7 : tensor<!pphlo.secret<f32>>, tensor<!pphlo.secret<f32>>
    }, {
      %4 = pphlo.exponential %arg1 : tensor<f32>
      %5 = pphlo.log %arg1: tensor<f32>
      %6 = pphlo.convert %4 : (tensor<f32>) -> tensor<!pphlo.secret<f32>>
      %7 = pphlo.convert %5 : (tensor<f32>) -> tensor<!pphlo.secret<f32>>
      pphlo.return %6, %7 : tensor<!pphlo.secret<f32>>, tensor<!pphlo.secret<f32>>
    }) : (tensor<!pphlo.secret<i1>>) -> (tensor<!pphlo.secret<f32>>, tensor<!pphlo.secret<f32>>)
    return %result0, %result1 : tensor<!pphlo.secret<f32>>, tensor<!pphlo.secret<f32>>
}

// -----

func.func @main(%arg0: tensor<f32>, %arg1: tensor<f32>) -> (tensor<f32>, tensor<f32>) {
    //CHECK: pphlo.if
    %0 = arith.constant dense<1.000000e+01> : tensor<f32>
    %1 = pphlo.less %arg0, %0 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %result0, %result1 = "pphlo.if"(%1) ({
      %4 = pphlo.add %arg1, %arg1 : tensor<f32>
      %5 = pphlo.subtract %arg1, %arg1 : tensor<f32>
      pphlo.return %4, %5 : tensor<f32>, tensor<f32>
    }, {
      %4 = pphlo.exponential %arg1 : tensor<f32>
      %5 = pphlo.log %arg1: tensor<f32>
      pphlo.return %4, %5 : tensor<f32>, tensor<f32>
    }) : (tensor<i1>) -> (tensor<f32>, tensor<f32>)
    return %result0, %result1 : tensor<f32>, tensor<f32>
}

// -----

func.func @main(%arg0 : tensor<!pphlo.secret<i32>>) -> (tensor<2x!pphlo.secret<i64>>) {
    //CHECK: %cst = arith.constant dense<[0, 1]> : tensor<2xi32>
    //CHECK: %cst_0 = arith.constant dense<1> : tensor<i32>
    //CHECK: %cst_1 = arith.constant dense<0> : tensor<i32>
    //CHECK: %cst_2 = arith.constant dense<0> : tensor<2xi64>
    //CHECK: %cst_3 = arith.constant dense<1> : tensor<2xi64>
    //CHECK: %0 = pphlo.convert %cst_2 : (tensor<2xi64>) -> tensor<2x!pphlo.secret<i64>>
    //CHECK: %1 = pphlo.convert %cst_3 : (tensor<2xi64>) -> tensor<2x!pphlo.secret<i64>>
    //CHECK: %2 = pphlo.clamp %cst_1, %arg0, %cst_0 : (tensor<i32>, tensor<!pphlo.secret<i32>>, tensor<i32>) -> tensor<!pphlo.secret<i32>>
    //CHECK: %3 = pphlo.reshape %2 : (tensor<!pphlo.secret<i32>>) -> tensor<1x!pphlo.secret<i32>>
    //CHECK: %4 = pphlo.broadcast %3, dims = [0] : (tensor<1x!pphlo.secret<i32>>) -> tensor<2x!pphlo.secret<i32>>
    //CHECK: %5 = pphlo.equal %4, %cst : (tensor<2x!pphlo.secret<i32>>, tensor<2xi32>) -> tensor<2x!pphlo.secret<i1>>
    //CHECK: %6 = pphlo.slice %5 [0:1:1] : (tensor<2x!pphlo.secret<i1>>) -> tensor<1x!pphlo.secret<i1>>
    //CHECK: %7 = pphlo.slice %5 [1:1:2] : (tensor<2x!pphlo.secret<i1>>) -> tensor<1x!pphlo.secret<i1>>
    //CHECK: %8 = pphlo.broadcast %6, dims = [0] : (tensor<1x!pphlo.secret<i1>>) -> tensor<2x!pphlo.secret<i1>>
    //CHECK: %9 = pphlo.multiply %0, %8 : (tensor<2x!pphlo.secret<i64>>, tensor<2x!pphlo.secret<i1>>) -> tensor<2x!pphlo.secret<i64>>
    //CHECK: %10 = pphlo.broadcast %7, dims = [0] : (tensor<1x!pphlo.secret<i1>>) -> tensor<2x!pphlo.secret<i1>>
    //CHECK: %11 = pphlo.multiply %1, %10 : (tensor<2x!pphlo.secret<i64>>, tensor<2x!pphlo.secret<i1>>) -> tensor<2x!pphlo.secret<i64>>
    //CHECK: %12 = pphlo.add %9, %11 : tensor<2x!pphlo.secret<i64>>
    //CHECK: return %12 : tensor<2x!pphlo.secret<i64>>
    %0 = arith.constant dense<0> : tensor<2xi64>
    %1 = arith.constant dense<1> : tensor<2xi64>
    %result0 = "pphlo.case"(%arg0) ({
      %2 = pphlo.convert %0 : (tensor<2xi64>) -> (tensor<2x!pphlo.secret<i64>>)
      pphlo.return %2 : tensor<2x!pphlo.secret<i64>>
    }, {
      %2 = pphlo.convert %1 : (tensor<2xi64>) -> (tensor<2x!pphlo.secret<i64>>)
      pphlo.return %2 : tensor<2x!pphlo.secret<i64>>
    }) : (tensor<!pphlo.secret<i32>>) -> (tensor<2x!pphlo.secret<i64>>)
    return %result0 : tensor<2x!pphlo.secret<i64>>
}

// -----

func.func @main(%arg0 : tensor<!pphlo.secret<i32>>) -> (tensor<2x!pphlo.secret<i64>>, tensor<2x!pphlo.secret<i64>>) {
    //CHECK: %cst = arith.constant dense<[0, 1]> : tensor<2xi32>
    //CHECK: %cst_0 = arith.constant dense<1> : tensor<i32>
    //CHECK: %cst_1 = arith.constant dense<0> : tensor<i32>
    //CHECK: %cst_2 = arith.constant dense<0> : tensor<2xi64>
    //CHECK: %cst_3 = arith.constant dense<1> : tensor<2xi64>
    //CHECK: %0 = pphlo.convert %cst_2 : (tensor<2xi64>) -> tensor<2x!pphlo.secret<i64>>
    //CHECK: %1 = pphlo.convert %cst_3 : (tensor<2xi64>) -> tensor<2x!pphlo.secret<i64>>
    //CHECK: %2 = pphlo.convert %cst_3 : (tensor<2xi64>) -> tensor<2x!pphlo.secret<i64>>
    //CHECK: %3 = pphlo.convert %cst_2 : (tensor<2xi64>) -> tensor<2x!pphlo.secret<i64>>
    //CHECK: %4 = pphlo.clamp %cst_1, %arg0, %cst_0 : (tensor<i32>, tensor<!pphlo.secret<i32>>, tensor<i32>) -> tensor<!pphlo.secret<i32>>
    //CHECK: %5 = pphlo.reshape %4 : (tensor<!pphlo.secret<i32>>) -> tensor<1x!pphlo.secret<i32>>
    //CHECK: %6 = pphlo.broadcast %5, dims = [0] : (tensor<1x!pphlo.secret<i32>>) -> tensor<2x!pphlo.secret<i32>>
    //CHECK: %7 = pphlo.equal %6, %cst : (tensor<2x!pphlo.secret<i32>>, tensor<2xi32>) -> tensor<2x!pphlo.secret<i1>>
    //CHECK: %8 = pphlo.slice %7 [0:1:1] : (tensor<2x!pphlo.secret<i1>>) -> tensor<1x!pphlo.secret<i1>>
    //CHECK: %9 = pphlo.slice %7 [1:1:2] : (tensor<2x!pphlo.secret<i1>>) -> tensor<1x!pphlo.secret<i1>>
    //CHECK: %10 = pphlo.broadcast %8, dims = [0] : (tensor<1x!pphlo.secret<i1>>) -> tensor<2x!pphlo.secret<i1>>
    //CHECK: %11 = pphlo.multiply %0, %10 : (tensor<2x!pphlo.secret<i64>>, tensor<2x!pphlo.secret<i1>>) -> tensor<2x!pphlo.secret<i64>>
    //CHECK: %12 = pphlo.broadcast %8, dims = [0] : (tensor<1x!pphlo.secret<i1>>) -> tensor<2x!pphlo.secret<i1>>
    //CHECK: %13 = pphlo.multiply %1, %12 : (tensor<2x!pphlo.secret<i64>>, tensor<2x!pphlo.secret<i1>>) -> tensor<2x!pphlo.secret<i64>>
    //CHECK: %14 = pphlo.broadcast %9, dims = [0] : (tensor<1x!pphlo.secret<i1>>) -> tensor<2x!pphlo.secret<i1>>
    //CHECK: %15 = pphlo.multiply %2, %14 : (tensor<2x!pphlo.secret<i64>>, tensor<2x!pphlo.secret<i1>>) -> tensor<2x!pphlo.secret<i64>>
    //CHECK: %16 = pphlo.add %11, %15 : tensor<2x!pphlo.secret<i64>>
    //CHECK: %17 = pphlo.broadcast %9, dims = [0] : (tensor<1x!pphlo.secret<i1>>) -> tensor<2x!pphlo.secret<i1>>
    //CHECK: %18 = pphlo.multiply %3, %17 : (tensor<2x!pphlo.secret<i64>>, tensor<2x!pphlo.secret<i1>>) -> tensor<2x!pphlo.secret<i64>>
    //CHECK: %19 = pphlo.add %13, %18 : tensor<2x!pphlo.secret<i64>>
    //CHECK: return %16, %19 : tensor<2x!pphlo.secret<i64>>, tensor<2x!pphlo.secret<i64>>
    %0 = arith.constant dense<0> : tensor<2xi64>
    %1 = arith.constant dense<1> : tensor<2xi64>
    %result0, %result1 = "pphlo.case"(%arg0) ({
      %2 = pphlo.convert %0 : (tensor<2xi64>) -> (tensor<2x!pphlo.secret<i64>>)
      %3 = pphlo.convert %1 : (tensor<2xi64>) -> (tensor<2x!pphlo.secret<i64>>)
      pphlo.return %2, %3 : tensor<2x!pphlo.secret<i64>>, tensor<2x!pphlo.secret<i64>>
    }, {
      %2 = pphlo.convert %1 : (tensor<2xi64>) -> (tensor<2x!pphlo.secret<i64>>)
      %3 = pphlo.convert %0 : (tensor<2xi64>) -> (tensor<2x!pphlo.secret<i64>>)
      pphlo.return %2, %3 : tensor<2x!pphlo.secret<i64>>, tensor<2x!pphlo.secret<i64>>
    }) : (tensor<!pphlo.secret<i32>>) -> (tensor<2x!pphlo.secret<i64>>, tensor<2x!pphlo.secret<i64>>)
    return %result0, %result1 : tensor<2x!pphlo.secret<i64>>, tensor<2x!pphlo.secret<i64>>
}
