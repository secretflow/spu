
// RUN: pphlo-opt -hlo-legalize-to-pphlo=input_vis_list=VIS_SECRET,VIS_PUBLIC --lower-conversion-cast %s --split-input-file  | FileCheck %s

func.func @main(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  //CHECK: %2 = "pphlo.if"(%1) ({
  //CHECK:   %3 = pphlo.add %arg1, %arg1 : tensor<f32>
  //CHECK:   %4 = pphlo.convert %3 : (tensor<f32>) -> tensor<!pphlo.secret<f32>>
  //CHECK:   pphlo.return %4 : tensor<!pphlo.secret<f32>>
  //CHECK: },  {
  //CHECK:   %3 = pphlo.exponential %arg1 : tensor<f32>
  //CHECK:   %4 = pphlo.convert %3 : (tensor<f32>) -> tensor<!pphlo.secret<f32>>
  //CHECK:   pphlo.return %4 : tensor<!pphlo.secret<f32>>
  //CHECK: }) : (tensor<!pphlo.secret<i1>>) -> tensor<!pphlo.secret<f32>>
  %cst = stablehlo.constant dense<1.000000e+01> : tensor<f32>
  %0 = "stablehlo.compare"(%arg0, %cst) {comparison_direction = #stablehlo<comparison_direction LT>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %1 = "stablehlo.if"(%0) ( {
    %2 = stablehlo.add %arg1, %arg1 : tensor<f32>
    "stablehlo.return"(%2) : (tensor<f32>) -> ()
  },  {
    %2 = "stablehlo.exponential"(%arg1) : (tensor<f32>) -> tensor<f32>
    "stablehlo.return"(%2) : (tensor<f32>) -> ()
  }) : (tensor<i1>) -> tensor<f32>
  return %1 : tensor<f32>
}

// -----

func.func @main(%arg0: tensor<i32>, %arg1: tensor<f32>) -> tensor<f32> {
  //CHECK: %1 = "pphlo.case"(%arg0) ({
  //CHECK:   %2 = pphlo.convert %arg1 : (tensor<f32>) -> tensor<!pphlo.secret<f32>>
  //CHECK:   pphlo.return %2 : tensor<!pphlo.secret<f32>>
  //CHECK: },  {
  //CHECK:   %2 = pphlo.convert %0 : (tensor<f32>) -> tensor<!pphlo.secret<f32>>
  //CHECK:   pphlo.return %2 : tensor<!pphlo.secret<f32>>
  //CHECK: }) : (tensor<!pphlo.secret<i32>>) -> tensor<!pphlo.secret<f32>>
  %0 = "stablehlo.case"(%arg0) ( {
    "stablehlo.return"(%arg1) : (tensor<f32>) -> ()
  },  {
    %1 = stablehlo.constant dense<1.000000e+01> : tensor<f32>
    "stablehlo.return"(%1) : (tensor<f32>) -> ()
  }) : (tensor<i32>) -> tensor<f32>
  return %0 : tensor<f32>
}
