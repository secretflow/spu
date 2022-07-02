// RUN: mlir-pphlo-opt -hlo-legalize-to-pphlo %s --split-input-file  | FileCheck %s

func @main(%arg0: tensor<f32>) -> tensor<f32> {
  //CHECK: %2 = "pphlo.if"(%1) ({
  //CHECK:   %3 = "pphlo.log"(%arg0) : (tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
  //CHECK:   "pphlo.return"(%3) : (tensor<!pphlo.pub<f32>>) -> ()
  //CHECK: },  {
  //CHECK:   %3 = "pphlo.exponential"(%arg0) : (tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
  //CHECK:   "pphlo.return"(%3) : (tensor<!pphlo.pub<f32>>) -> ()
  //CHECK: }) : (tensor<!pphlo.pub<i1>>) -> tensor<!pphlo.pub<f32>>
  %cst = mhlo.constant dense<1.000000e+01> : tensor<f32>
  %0 = "mhlo.compare"(%arg0, %cst) {comparison_direction = #mhlo<"comparison_direction LT">} : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %1 = "mhlo.if"(%0) ( {
    %2 = "mhlo.log"(%arg0) : (tensor<f32>) -> tensor<f32>
    "mhlo.return"(%2) : (tensor<f32>) -> ()
  },  {
    %2 = "mhlo.exponential"(%arg0) : (tensor<f32>) -> tensor<f32>
    "mhlo.return"(%2) : (tensor<f32>) -> ()
  }) : (tensor<i1>) -> tensor<f32>
  return %1 : tensor<f32>
}