// RUN: mlir-pphlo-opt --hlo-legalize-to-pphlo=input_vis_list=VIS_SECRET --lower-conversion-cast --split-input-file %s | FileCheck %s

func.func @main(%arg0: tensor<20xi32>) -> (tensor<20xi32>) {
    %0 = stablehlo.iota dim = 0 : tensor<20xi32>
    // CHECK: %1:2 = "pphlo.sort"(%arg0, %0) ({
    // CHECK: ^bb0(%arg1: tensor<!pphlo.secret<i32>>, %arg2: tensor<!pphlo.secret<i32>>, %arg3: tensor<i32>, %arg4: tensor<i32>):
    // CHECK:   %2 = "pphlo.less"(%arg1, %arg2) : (tensor<!pphlo.secret<i32>>, tensor<!pphlo.secret<i32>>) -> tensor<!pphlo.secret<i1>>
    // CHECK:   "pphlo.return"(%2) : (tensor<!pphlo.secret<i1>>) -> ()
    // CHECK:  }) {dimension = 0 : i64, is_stable = true} : (tensor<20x!pphlo.secret<i32>>, tensor<20xi32>) -> (tensor<20x!pphlo.secret<i32>>, tensor<20x!pphlo.secret<i32>>)
    %1:2 = "stablehlo.sort"(%arg0, %0) ({
    ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<i32>, %arg4: tensor<i32>):
      %2 = stablehlo.compare  LT, %arg1, %arg2 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %2 : tensor<i1>
    }) {dimension = 0 : i64, is_stable = true} : (tensor<20xi32>, tensor<20xi32>) -> (tensor<20xi32>, tensor<20xi32>)
    return %1#1: tensor<20xi32>
}
