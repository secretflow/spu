// RUN: spu-opt --sort-lowering --split-input-file %s | FileCheck %s

func.func @main(%arg0: tensor<10x!pphlo.secret<f32>>) -> tensor<10x!pphlo.secret<f32>> {
    // CHECK: pphlo.simple_sort %arg0 ASC, dim = 0, num_keys = 1 : (tensor<10x!pphlo.secret<f32>>) -> tensor<10x!pphlo.secret<f32>>
    %0 = "pphlo.sort"(%arg0) ({
    ^bb0(%arg1: tensor<!pphlo.secret<f32>>, %arg2: tensor<!pphlo.secret<f32>>):
      %1 = pphlo.less %arg1, %arg2 : (tensor<!pphlo.secret<f32>>, tensor<!pphlo.secret<f32>>) -> tensor<!pphlo.secret<i1>>
      pphlo.return %1 : tensor<!pphlo.secret<i1>>
    }) {dimension = 0 : i64, is_stable = false} : (tensor<10x!pphlo.secret<f32>>) -> tensor<10x!pphlo.secret<f32>>
    return %0 : tensor<10x!pphlo.secret<f32>>
  }

// -----

func.func @main(%arg0: tensor<10x!pphlo.secret<f32>>, %arg1: tensor<10x!pphlo.secret<f32>>) -> (tensor<10x!pphlo.secret<f32>>, tensor<10x!pphlo.secret<f32>>) {
    // CHECK: pphlo.simple_sort %arg0, %arg1 ASC, dim = 0, num_keys = 1 : (tensor<10x!pphlo.secret<f32>>, tensor<10x!pphlo.secret<f32>>) -> (tensor<10x!pphlo.secret<f32>>, tensor<10x!pphlo.secret<f32>>)
    %0:2 = "pphlo.sort"(%arg0, %arg1) ({
    ^bb0(%arg2: tensor<!pphlo.secret<f32>>, %arg3: tensor<!pphlo.secret<f32>>, %arg4: tensor<!pphlo.secret<f32>>, %arg5: tensor<!pphlo.secret<f32>>):
      %1 = pphlo.less %arg2, %arg3 : (tensor<!pphlo.secret<f32>>, tensor<!pphlo.secret<f32>>) -> tensor<!pphlo.secret<i1>>
      pphlo.return %1 : tensor<!pphlo.secret<i1>>
    }) {dimension = 0 : i64, is_stable = false} : (tensor<10x!pphlo.secret<f32>>, tensor<10x!pphlo.secret<f32>>) -> (tensor<10x!pphlo.secret<f32>>, tensor<10x!pphlo.secret<f32>>)
    return %0#0, %0#1 : tensor<10x!pphlo.secret<f32>>, tensor<10x!pphlo.secret<f32>>
  }

// -----

func.func @main(%arg0: tensor<3x4x!pphlo.secret<f32>>, %arg1: tensor<3x4x!pphlo.secret<f32>>, %arg2: tensor<3x4x!pphlo.secret<f32>>) -> (tensor<3x4x!pphlo.secret<f32>>, tensor<3x4x!pphlo.secret<f32>>, tensor<3x4x!pphlo.secret<f32>>) {
    // CHECK: %0:3 = pphlo.simple_sort %arg0, %arg1, %arg2  ASC, dim = 1, num_keys = 2 : (tensor<3x4x!pphlo.secret<f32>>, tensor<3x4x!pphlo.secret<f32>>, tensor<3x4x!pphlo.secret<f32>>) -> (tensor<3x4x!pphlo.secret<f32>>, tensor<3x4x!pphlo.secret<f32>>, tensor<3x4x!pphlo.secret<f32>>)
    %0:3 = "pphlo.sort"(%arg0, %arg1, %arg2) ({
    ^bb0(%arg3: tensor<!pphlo.secret<f32>>, %arg4: tensor<!pphlo.secret<f32>>, %arg5: tensor<!pphlo.secret<f32>>, %arg6: tensor<!pphlo.secret<f32>>, %arg7: tensor<!pphlo.secret<f32>>, %arg8: tensor<!pphlo.secret<f32>>):
      %1 = pphlo.less %arg3, %arg4 : (tensor<!pphlo.secret<f32>>, tensor<!pphlo.secret<f32>>) -> tensor<!pphlo.secret<i1>>
      %2 = pphlo.equal %arg3, %arg4 : (tensor<!pphlo.secret<f32>>, tensor<!pphlo.secret<f32>>) -> tensor<!pphlo.secret<i1>>
      %3 = pphlo.less %arg5, %arg6 : (tensor<!pphlo.secret<f32>>, tensor<!pphlo.secret<f32>>) -> tensor<!pphlo.secret<i1>>
      %4 = pphlo.and %2, %3 : tensor<!pphlo.secret<i1>>
      %5 = pphlo.or %1, %4 : tensor<!pphlo.secret<i1>>
      pphlo.return %5 : tensor<!pphlo.secret<i1>>
    }) {dimension = 1 : i64, is_stable = true} : (tensor<3x4x!pphlo.secret<f32>>, tensor<3x4x!pphlo.secret<f32>>, tensor<3x4x!pphlo.secret<f32>>) -> (tensor<3x4x!pphlo.secret<f32>>, tensor<3x4x!pphlo.secret<f32>>, tensor<3x4x!pphlo.secret<f32>>)
    return %0#0, %0#1, %0#2 : tensor<3x4x!pphlo.secret<f32>>, tensor<3x4x!pphlo.secret<f32>>, tensor<3x4x!pphlo.secret<f32>>
  }
