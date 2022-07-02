// RUN: mlir-pphlo-opt -hlo-legalize-to-pphlo %s --split-input-file  | FileCheck %s

func @main(%arg0: tensor<i64>) -> tensor<i64> {
  //CHECK: %0 = "pphlo.while"(%arg0) ({
  //CHECK: ^bb0(%arg1: tensor<!pphlo.pub<i64>>):
  //CHECK:   %1 = "pphlo.constant"() {value = dense<false> : tensor<i1>} : () -> tensor<!pphlo.pub<i1>>
  //CHECK:   "pphlo.return"(%1) : (tensor<!pphlo.pub<i1>>) -> ()
  //CHECK: },  {
  //CHECK: ^bb0(%arg1: tensor<!pphlo.pub<i64>>):
  //CHECK:   %1 = "pphlo.add"(%arg1, %arg1) {name = "compare.0"} : (tensor<!pphlo.pub<i64>>, tensor<!pphlo.pub<i64>>) -> tensor<!pphlo.pub<i64>>
  //CHECK:   "pphlo.return"(%1) : (tensor<!pphlo.pub<i64>>) -> ()
  //CHECK: }) : (tensor<!pphlo.pub<i64>>) -> tensor<!pphlo.pub<i64>>
  %0 = "mhlo.while"(%arg0) ( {
  ^bb0(%arg1: tensor<i64>):
    %1 = "mhlo.compare"(%arg1, %arg1) {comparison_direction = #mhlo<"comparison_direction LT">, name = "compare.2"} : (tensor<i64>, tensor<i64>) -> tensor<i1>
    "mhlo.return"(%1) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg1: tensor<i64>):
    %1 = mhlo.add %arg1, %arg1 {name = "compare.0"} : tensor<i64>
    "mhlo.return"(%1) : (tensor<i64>) -> ()
  }) : (tensor<i64>) -> (tensor<i64>)

  return %0 : tensor<i64>
}