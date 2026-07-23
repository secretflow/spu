// RUN: spu-opt -hlo-legalize-to-pphlo=input_vis_list=VIS_PUBLIC --lower-conversion-cast %s --split-input-file  | FileCheck %s

func.func @main(%arg0: tensor<i64>) -> tensor<i64> {
  //CHECK: %0 = pphlo.while(%arg1 = %arg0) : tensor<i64>
  //CHECK: cond {
  //CHECK:   %1 = pphlo.less %arg1, %arg1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
  //CHECK:   pphlo.return %1 : tensor<i1>
  //CHECK: } do {
  //CHECK:   %1 = pphlo.add %arg1, %arg1 {name = "compare.0"} : tensor<i64>
  //CHECK:   pphlo.return %1 : tensor<i64>
  //CHECK: }
  %0 = "stablehlo.while"(%arg0) ( {
  ^bb0(%arg1: tensor<i64>):
    %1 = "stablehlo.compare"(%arg1, %arg1) {comparison_direction = #stablehlo<comparison_direction LT>, name = "compare.2"} : (tensor<i64>, tensor<i64>) -> tensor<i1>
    "stablehlo.return"(%1) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg1: tensor<i64>):
    %1 = stablehlo.add %arg1, %arg1 {name = "compare.0"} : tensor<i64>
    "stablehlo.return"(%1) : (tensor<i64>) -> ()
  }) : (tensor<i64>) -> (tensor<i64>)

  return %0 : tensor<i64>
}