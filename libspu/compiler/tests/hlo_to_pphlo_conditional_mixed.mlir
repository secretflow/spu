// RUN: mlir-pphlo-opt -hlo-legalize-to-pphlo='io-visibility-json={"inputs":["VIS_PUBLIC","VIS_SECRET"]}' %s --split-input-file  | FileCheck %s

func.func @main(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  //CHECK: %2 = "pphlo.if"(%1) ({
  //CHECK:   %3 = "pphlo.add"(%arg0, %arg1) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.sec<f32>>) -> tensor<!pphlo.sec<f32>>
  //CHECK:   "pphlo.return"(%3) : (tensor<!pphlo.sec<f32>>) -> ()
  //CHECK: },  {
  //CHECK:   %3 = "pphlo.exponential"(%arg0) : (tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
  //CHECK:   %4 = builtin.unrealized_conversion_cast %3 : tensor<!pphlo.pub<f32>> to tensor<!pphlo.sec<f32>>
  //CHECK:   "pphlo.return"(%4) : (tensor<!pphlo.sec<f32>>) -> ()
  //CHECK: }) : (tensor<!pphlo.pub<i1>>) -> tensor<!pphlo.sec<f32>>
  %cst = mhlo.constant dense<1.000000e+01> : tensor<f32>
  %0 = "mhlo.compare"(%arg0, %cst) {comparison_direction = #mhlo<comparison_direction LT>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %1 = "mhlo.if"(%0) ( {
    %2 = mhlo.add %arg0, %arg1 : tensor<f32>
    "mhlo.return"(%2) : (tensor<f32>) -> ()
  },  {
    %2 = "mhlo.exponential"(%arg0) : (tensor<f32>) -> tensor<f32>
    "mhlo.return"(%2) : (tensor<f32>) -> ()
  }) : (tensor<i1>) -> tensor<f32>
  return %1 : tensor<f32>
}

// -----

func.func @main(%arg0: tensor<i32>, %arg1: tensor<f32>) -> tensor<f32> {
  //CHECK: %0 = "pphlo.case"(%arg0) ({
  //CHECK:   "pphlo.return"(%arg1) : (tensor<!pphlo.sec<f32>>) -> ()
  //CHECK: },  {
  //CHECK:   %1 = "pphlo.constant"() {value = dense<1.000000e+01> : tensor<f32>} : () -> tensor<!pphlo.pub<f32>>
  //CHECK:   %2 = builtin.unrealized_conversion_cast %1 : tensor<!pphlo.pub<f32>> to tensor<!pphlo.sec<f32>>
  //CHECK:   "pphlo.return"(%2) : (tensor<!pphlo.sec<f32>>) -> ()
  //CHECK: }) : (tensor<!pphlo.pub<i32>>) -> tensor<!pphlo.sec<f32>>
  %0 = "mhlo.case"(%arg0) ( {
    "mhlo.return"(%arg1) : (tensor<f32>) -> ()
  },  {
    %1 = mhlo.constant dense<1.000000e+01> : tensor<f32>
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) : (tensor<i32>) -> tensor<f32>
  return %0 : tensor<f32>
}
