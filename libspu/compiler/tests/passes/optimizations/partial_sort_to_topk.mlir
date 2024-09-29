// RUN: spu-opt --partial-sort-to-topk --split-input-file %s | FileCheck %s

func.func @main(%arg0: tensor<12x2x!pphlo.secret<i32>>) -> tensor<2x!pphlo.secret<f32>> {
    // CHECK: pphlo.custom_call @"@spu.topk"(%7) {mhlo.attributes = {k = 5 : i64, k_hi = 7 : i64, largest = true, value_only = true}} : (tensor<2x12x!pphlo.secret<f32>>) -> tensor<2x7x!pphlo.secret<f32>>
    %0 = arith.constant dense<5.000000e-01> : tensor<1x2xf32>
    %1 = arith.constant dense<0x7FC00000> : tensor<12x2xf32>
    %2 = arith.constant dense<false> : tensor<i1>
    %3 = pphlo.convert %arg0 : (tensor<12x2x!pphlo.secret<i32>>) -> tensor<12x2x!pphlo.secret<f32>>
    %4 = pphlo.equal %3, %3 : (tensor<12x2x!pphlo.secret<f32>>, tensor<12x2x!pphlo.secret<f32>>) -> tensor<12x2x!pphlo.secret<i1>>
    %5 = pphlo.not %4 : tensor<12x2x!pphlo.secret<i1>>
    %6 = pphlo.convert %2 : (tensor<i1>) -> tensor<!pphlo.secret<i1>>
    %7 = pphlo.reduce(%5 init: %6) applies pphlo.or across dimensions = [0] : (tensor<12x2x!pphlo.secret<i1>>, tensor<!pphlo.secret<i1>>) -> tensor<2x!pphlo.secret<i1>>
    %8 = pphlo.broadcast %7, dims = [1] : (tensor<2x!pphlo.secret<i1>>) -> tensor<12x2x!pphlo.secret<i1>>
    %9 = pphlo.select %8, %1, %3 : (tensor<12x2x!pphlo.secret<i1>>, tensor<12x2xf32>, tensor<12x2x!pphlo.secret<f32>>) -> tensor<12x2x!pphlo.secret<f32>>
    %10 = pphlo.simple_sort %9  ASC, dim = 0, num_keys = 1 : (tensor<12x2x!pphlo.secret<f32>>) -> tensor<12x2x!pphlo.secret<f32>>
    %11 = pphlo.slice %10 [5:1:6, 0:1:2] : (tensor<12x2x!pphlo.secret<f32>>) -> tensor<1x2x!pphlo.secret<f32>>
    %12 = pphlo.slice %10 [6:1:7, 0:1:2] : (tensor<12x2x!pphlo.secret<f32>>) -> tensor<1x2x!pphlo.secret<f32>>
    %13 = pphlo.add %11, %12 : tensor<1x2x!pphlo.secret<f32>>
    %14 = pphlo.multiply %13, %0 : (tensor<1x2x!pphlo.secret<f32>>, tensor<1x2xf32>) -> tensor<1x2x!pphlo.secret<f32>>
    %15 = pphlo.reshape %14 : (tensor<1x2x!pphlo.secret<f32>>) -> tensor<2x!pphlo.secret<f32>>
    return %15 : tensor<2x!pphlo.secret<f32>>
}

// -----

func.func @main(%arg0: tensor<12x!pphlo.secret<i32>>) -> tensor<!pphlo.secret<f32>> {
    //CHECK: pphlo.custom_call @"@spu.topk"(%6) {mhlo.attributes = {k = 5 : i64, k_hi = 7 : i64, largest = true, value_only = true}} : (tensor<12x!pphlo.secret<f32>>) -> tensor<7x!pphlo.secret<f32>>
    %0 = arith.constant dense<0x7FC00000> : tensor<12xf32>
    %1 = arith.constant dense<5.000000e-01> : tensor<1xf32>
    %2 = arith.constant dense<false> : tensor<i1>
    %3 = pphlo.convert %arg0 : (tensor<12x!pphlo.secret<i32>>) -> tensor<12x!pphlo.secret<f32>>
    %4 = pphlo.equal %3, %3 : (tensor<12x!pphlo.secret<f32>>, tensor<12x!pphlo.secret<f32>>) -> tensor<12x!pphlo.secret<i1>>
    %5 = pphlo.not %4 : tensor<12x!pphlo.secret<i1>>
    %6 = pphlo.convert %2 : (tensor<i1>) -> tensor<!pphlo.secret<i1>>
    %7 = pphlo.reduce(%5 init: %6) applies pphlo.or across dimensions = [0] : (tensor<12x!pphlo.secret<i1>>, tensor<!pphlo.secret<i1>>) -> tensor<!pphlo.secret<i1>>
    %8 = pphlo.broadcast %7, dims = [] : (tensor<!pphlo.secret<i1>>) -> tensor<12x!pphlo.secret<i1>>
    %9 = pphlo.select %8, %0, %3 : (tensor<12x!pphlo.secret<i1>>, tensor<12xf32>, tensor<12x!pphlo.secret<f32>>) -> tensor<12x!pphlo.secret<f32>>
    %10 = pphlo.simple_sort %9  ASC, dim = 0, num_keys = 1 : (tensor<12x!pphlo.secret<f32>>) -> tensor<12x!pphlo.secret<f32>>
    %11 = pphlo.slice %10 [5:1:6] : (tensor<12x!pphlo.secret<f32>>) -> tensor<1x!pphlo.secret<f32>>
    %12 = pphlo.slice %10 [6:1:7] : (tensor<12x!pphlo.secret<f32>>) -> tensor<1x!pphlo.secret<f32>>
    %13 = pphlo.add %11, %12 : tensor<1x!pphlo.secret<f32>>
    %14 = pphlo.multiply %13, %1 : (tensor<1x!pphlo.secret<f32>>, tensor<1xf32>) -> tensor<1x!pphlo.secret<f32>>
    %15 = pphlo.reshape %14 : (tensor<1x!pphlo.secret<f32>>) -> tensor<!pphlo.secret<f32>>
    return %15 : tensor<!pphlo.secret<f32>>
}

// -----

func.func @main(%arg0: tensor<14x2x!pphlo.secret<i32>>) -> tensor<2x!pphlo.secret<f32>> {
    // CHECK: pphlo.custom_call @"@spu.topk"(%7) {mhlo.attributes = {k = 5 : i64, k_hi = 7 : i64, largest = true, value_only = true}} : (tensor<2x14x!pphlo.secret<f32>>) -> tensor<2x7x!pphlo.secret<f32>>
    %0 = arith.constant dense<5.000000e-01> : tensor<1x2xf32>
    %1 = arith.constant dense<0x7FC00000> : tensor<14x2xf32>
    %2 = arith.constant dense<false> : tensor<i1>
    %3 = pphlo.convert %arg0 : (tensor<14x2x!pphlo.secret<i32>>) -> tensor<14x2x!pphlo.secret<f32>>
    %4 = pphlo.equal %3, %3 : (tensor<14x2x!pphlo.secret<f32>>, tensor<14x2x!pphlo.secret<f32>>) -> tensor<14x2x!pphlo.secret<i1>>
    %5 = pphlo.not %4 : tensor<14x2x!pphlo.secret<i1>>
    %6 = pphlo.convert %2 : (tensor<i1>) -> tensor<!pphlo.secret<i1>>
    %7 = pphlo.reduce(%5 init: %6) applies pphlo.or across dimensions = [0] : (tensor<14x2x!pphlo.secret<i1>>, tensor<!pphlo.secret<i1>>) -> tensor<2x!pphlo.secret<i1>>
    %8 = pphlo.broadcast %7, dims = [1] : (tensor<2x!pphlo.secret<i1>>) -> tensor<14x2x!pphlo.secret<i1>>
    %9 = pphlo.select %8, %1, %3 : (tensor<14x2x!pphlo.secret<i1>>, tensor<14x2xf32>, tensor<14x2x!pphlo.secret<f32>>) -> tensor<14x2x!pphlo.secret<f32>>
    %10 = pphlo.simple_sort %9  DES, dim = 0, num_keys = 1 : (tensor<14x2x!pphlo.secret<f32>>) -> tensor<14x2x!pphlo.secret<f32>>
    %11 = pphlo.slice %10 [5:1:6, 0:1:2] : (tensor<14x2x!pphlo.secret<f32>>) -> tensor<1x2x!pphlo.secret<f32>>
    %12 = pphlo.slice %10 [6:1:7, 0:1:2] : (tensor<14x2x!pphlo.secret<f32>>) -> tensor<1x2x!pphlo.secret<f32>>
    %13 = pphlo.add %11, %12 : tensor<1x2x!pphlo.secret<f32>>
    %14 = pphlo.multiply %13, %0 : (tensor<1x2x!pphlo.secret<f32>>, tensor<1x2xf32>) -> tensor<1x2x!pphlo.secret<f32>>
    %15 = pphlo.reshape %14 : (tensor<1x2x!pphlo.secret<f32>>) -> tensor<2x!pphlo.secret<f32>>
    return %15 : tensor<2x!pphlo.secret<f32>>
}

// -----

func.func @main(%arg0: tensor<14x!pphlo.secret<i32>>) -> tensor<!pphlo.secret<f32>> {
    //CHECK: pphlo.custom_call @"@spu.topk"(%6) {mhlo.attributes = {k = 5 : i64, k_hi = 7 : i64, largest = true, value_only = true}} : (tensor<14x!pphlo.secret<f32>>) -> tensor<7x!pphlo.secret<f32>>
    %0 = arith.constant dense<0x7FC00000> : tensor<14xf32>
    %1 = arith.constant dense<5.000000e-01> : tensor<1xf32>
    %2 = arith.constant dense<false> : tensor<i1>
    %3 = pphlo.convert %arg0 : (tensor<14x!pphlo.secret<i32>>) -> tensor<14x!pphlo.secret<f32>>
    %4 = pphlo.equal %3, %3 : (tensor<14x!pphlo.secret<f32>>, tensor<14x!pphlo.secret<f32>>) -> tensor<14x!pphlo.secret<i1>>
    %5 = pphlo.not %4 : tensor<14x!pphlo.secret<i1>>
    %6 = pphlo.convert %2 : (tensor<i1>) -> tensor<!pphlo.secret<i1>>
    %7 = pphlo.reduce(%5 init: %6) applies pphlo.or across dimensions = [0] : (tensor<14x!pphlo.secret<i1>>, tensor<!pphlo.secret<i1>>) -> tensor<!pphlo.secret<i1>>
    %8 = pphlo.broadcast %7, dims = [] : (tensor<!pphlo.secret<i1>>) -> tensor<14x!pphlo.secret<i1>>
    %9 = pphlo.select %8, %0, %3 : (tensor<14x!pphlo.secret<i1>>, tensor<14xf32>, tensor<14x!pphlo.secret<f32>>) -> tensor<14x!pphlo.secret<f32>>
    %10 = pphlo.simple_sort %9  DES, dim = 0, num_keys = 1 : (tensor<14x!pphlo.secret<f32>>) -> tensor<14x!pphlo.secret<f32>>
    %11 = pphlo.slice %10 [5:1:6] : (tensor<14x!pphlo.secret<f32>>) -> tensor<1x!pphlo.secret<f32>>
    %12 = pphlo.slice %10 [6:1:7] : (tensor<14x!pphlo.secret<f32>>) -> tensor<1x!pphlo.secret<f32>>
    %13 = pphlo.add %11, %12 : tensor<1x!pphlo.secret<f32>>
    %14 = pphlo.multiply %13, %1 : (tensor<1x!pphlo.secret<f32>>, tensor<1xf32>) -> tensor<1x!pphlo.secret<f32>>
    %15 = pphlo.reshape %14 : (tensor<1x!pphlo.secret<f32>>) -> tensor<!pphlo.secret<f32>>
    return %15 : tensor<!pphlo.secret<f32>>
}
