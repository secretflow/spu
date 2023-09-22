// RUN: mlir-pphlo-opt --sort-lowering --split-input-file %s | FileCheck %s

func.func @main(%arg0: tensor<10x!pphlo.sec<f32>>) -> tensor<10x!pphlo.sec<f32>> {
    // CHECK: simple_sort
    %0 = "pphlo.sort"(%arg0) ({
    ^bb0(%arg1: tensor<!pphlo.sec<f32>>, %arg2: tensor<!pphlo.sec<f32>>):
      %1 = "pphlo.less"(%arg1, %arg2) : (tensor<!pphlo.sec<f32>>, tensor<!pphlo.sec<f32>>) -> tensor<!pphlo.sec<i1>>
      "pphlo.return"(%1) : (tensor<!pphlo.sec<i1>>) -> ()
    }) {dimension = 0 : i64, is_stable = false} : (tensor<10x!pphlo.sec<f32>>) -> tensor<10x!pphlo.sec<f32>>
    return %0 : tensor<10x!pphlo.sec<f32>>
  }
