// RUN: mlir-pphlo-opt --hlo-legalize-to-pphlo=input_vis_list=VIS_PUBLIC,VIS_SECRET --lower-conversion-cast --split-input-file %s | FileCheck %s

func.func @main(%arg0: tensor<2x2xi32>, %arg1: tensor<2x2xi32>) -> (tensor<2x2xi1>) {
    // CHECK: %[[COMP:.+]] = "pphlo.equal"(%arg0, %arg1) : (tensor<2x2xi32>, tensor<2x2x!pphlo.secret<i32>>) -> tensor<2x2x!pphlo.secret<i1>>
    %0 = "stablehlo.compare"(%arg0, %arg1) {comparison_direction = #stablehlo<comparison_direction EQ>} : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi1>
    // CHECK: return %[[COMP]] : tensor<2x2x!pphlo.secret<i1>>
    return %0 : tensor<2x2xi1>
}

// -----

func.func @main(%arg0: tensor<2x2xi32>, %arg1: tensor<2x2xi32>) -> (tensor<2x2xi1>) {
    // CHECK: %[[COMP:.+]] = "pphlo.not_equal"(%arg0, %arg1) : (tensor<2x2xi32>, tensor<2x2x!pphlo.secret<i32>>) -> tensor<2x2x!pphlo.secret<i1>>
    %0 = "stablehlo.compare"(%arg0, %arg1) {comparison_direction = #stablehlo<comparison_direction NE>} : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi1>
    // CHECK: return %[[COMP]] : tensor<2x2x!pphlo.secret<i1>>
    return %0 : tensor<2x2xi1>
}

// -----

func.func @main(%arg0: tensor<2x2xi32>, %arg1: tensor<2x2xi32>) -> (tensor<2x2xi1>) {
    // CHECK: %[[COMP:.+]] = "pphlo.less"(%arg0, %arg1) : (tensor<2x2xi32>, tensor<2x2x!pphlo.secret<i32>>) -> tensor<2x2x!pphlo.secret<i1>>
    %0 = "stablehlo.compare"(%arg0, %arg1) {comparison_direction = #stablehlo<comparison_direction LT>} : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi1>
    // CHECK: return %[[COMP]] : tensor<2x2x!pphlo.secret<i1>>
    return %0 : tensor<2x2xi1>
}

// -----

func.func @main(%arg0: tensor<2x2xi32>, %arg1: tensor<2x2xi32>) -> (tensor<2x2xi1>) {
    // CHECK: %[[COMP:.+]] = "pphlo.greater_equal"(%arg0, %arg1) : (tensor<2x2xi32>, tensor<2x2x!pphlo.secret<i32>>) -> tensor<2x2x!pphlo.secret<i1>>
    %0 = "stablehlo.compare"(%arg0, %arg1) {comparison_direction = #stablehlo<comparison_direction GE>} : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi1>
    // CHECK: return %[[COMP]] : tensor<2x2x!pphlo.secret<i1>>
    return %0 : tensor<2x2xi1>
}

// -----

func.func @main(%arg0: tensor<2x2xi32>, %arg1: tensor<2x2xi32>) -> (tensor<2x2xi1>) {
    // CHECK: %[[COMP:.+]] = "pphlo.less_equal"(%arg0, %arg1) : (tensor<2x2xi32>, tensor<2x2x!pphlo.secret<i32>>) -> tensor<2x2x!pphlo.secret<i1>>
    %0 = "stablehlo.compare"(%arg0, %arg1) {comparison_direction = #stablehlo<comparison_direction LE>} : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi1>
    // CHECK: return %[[COMP]] : tensor<2x2x!pphlo.secret<i1>>
    return %0 : tensor<2x2xi1>
}

// -----

func.func @main(%arg0: tensor<2x2xi32>, %arg1: tensor<2x2xi32>) -> (tensor<2x2xi1>) {
    // CHECK: %[[COMP:.+]] = "pphlo.greater"(%arg0, %arg1) : (tensor<2x2xi32>, tensor<2x2x!pphlo.secret<i32>>) -> tensor<2x2x!pphlo.secret<i1>>
    %0 = "stablehlo.compare"(%arg0, %arg1) {comparison_direction = #stablehlo<comparison_direction GT>} : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi1>
    // CHECK: return %[[COMP]] : tensor<2x2x!pphlo.secret<i1>>
    return %0 : tensor<2x2xi1>
}
