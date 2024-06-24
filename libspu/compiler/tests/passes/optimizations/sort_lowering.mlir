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
