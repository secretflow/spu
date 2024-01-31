// RUN: mlir-pphlo-opt --optimize-sqrt-plus-eps --rewrite-div-sqrt-pattern --split-input-file %s | FileCheck %s

func.func @main(%arg0: tensor<f32>, %arg1: tensor<f32>) -> (tensor<f32>) {
    %0 = "pphlo.constant"() {value = dense<9.99999993E-9> : tensor<f32>} : () -> tensor<f32>
    %1 = "pphlo.sqrt"(%arg0) : (tensor<f32>) -> tensor<f32>
    %2 = "pphlo.add"(%1, %0) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    //CHECK: pphlo.rsqrt
    //CHECK: pphlo.mul
    %3 = "pphlo.divide"(%arg1, %2) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    return %3: tensor<f32>
}

// -----

func.func @main(%arg0: tensor<f32>, %arg1: tensor<f32>) -> (tensor<f32>) {
    %0 = "pphlo.constant"() {value = dense<9.99999993> : tensor<f32>} : () -> tensor<f32>
    %1 = "pphlo.sqrt"(%arg0) : (tensor<f32>) -> tensor<f32>
    %2 = "pphlo.add"(%1, %0) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    //CHECK-NOT: pphlo.rsqrt
    %3 = "pphlo.divide"(%arg1, %2) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    return %3: tensor<f32>
}

// -----

func.func @main(%arg0: tensor<f32>, %arg1: tensor<f32>) -> (tensor<f32>) {
    %0 = "pphlo.constant"() {value = dense<9.99999993E-9> : tensor<f32>} : () -> tensor<f32>
    %1 = "pphlo.sqrt"(%arg0) : (tensor<f32>) -> tensor<f32>
    %2 = "pphlo.add"(%1, %0) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %3 = "pphlo.multiply"(%1, %1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    //CHECK: pphlo.rsqrt
    //CHECK: pphlo.mul
    %4 = "pphlo.divide"(%arg1, %3) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    return %4: tensor<f32>
}

// -----

func.func @main(%arg0: tensor<3x4x!pphlo.secret<f32>>) -> tensor<3x4x!pphlo.secret<f32>> {
    // CHECK: %[[RSQRT:.+]] = "pphlo.rsqrt"(%arg0) : (tensor<3x4x!pphlo.secret<f32>>) -> tensor<3x4x!pphlo.secret<f32>>
    // CHECK: "pphlo.multiply"(%arg0, %[[RSQRT]]) : (tensor<3x4x!pphlo.secret<f32>>, tensor<3x4x!pphlo.secret<f32>>) -> tensor<3x4x!pphlo.secret<f32>>
    %0 = "pphlo.sqrt"(%arg0) : (tensor<3x4x!pphlo.secret<f32>>) -> tensor<3x4x!pphlo.secret<f32>>
    %1 = "pphlo.divide"(%arg0, %0) : (tensor<3x4x!pphlo.secret<f32>>, tensor<3x4x!pphlo.secret<f32>>) -> tensor<3x4x!pphlo.secret<f32>>
    return %1 : tensor<3x4x!pphlo.secret<f32>>
}

// -----

func.func @main(%arg0: tensor<3x4x!pphlo.secret<i32>>) -> tensor<3x4x!pphlo.secret<f32>> {
    %0 = "pphlo.convert"(%arg0) : (tensor<3x4x!pphlo.secret<i32>>) -> tensor<3x4x!pphlo.secret<f32>>
    %1 = "pphlo.reshape"(%arg0) : (tensor<3x4x!pphlo.secret<i32>>) -> tensor<3x4x1x!pphlo.secret<i32>>
    %2 = "pphlo.transpose"(%1) {permutation = array<i64: 0, 2, 1>} : (tensor<3x4x1x!pphlo.secret<i32>>) -> tensor<3x1x4x!pphlo.secret<i32>>
    %3 = "pphlo.dot_general"(%2, %1) {dot_dimension_numbers = #pphlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<3x1x4x!pphlo.secret<i32>>, tensor<3x4x1x!pphlo.secret<i32>>) -> tensor<3x!pphlo.secret<i32>>
    %4 = "pphlo.convert"(%3) : (tensor<3x!pphlo.secret<i32>>) -> tensor<3x!pphlo.secret<f32>>
    // CHECK: %[[RSQRT:.+]] = "pphlo.rsqrt"
    // CHECK: %[[BCAST:.+]] = "pphlo.broadcast"(%[[RSQRT]]) {broadcast_dimensions = array<i64: 0>} : (tensor<3x!pphlo.secret<f32>>) -> tensor<3x4x!pphlo.secret<f32>>
    // CHECK: "pphlo.multiply"(%0, %[[BCAST]]) : (tensor<3x4x!pphlo.secret<f32>>, tensor<3x4x!pphlo.secret<f32>>) -> tensor<3x4x!pphlo.secret<f32>>
    %5 = "pphlo.sqrt"(%4) : (tensor<3x!pphlo.secret<f32>>) -> tensor<3x!pphlo.secret<f32>>
    %6 = "pphlo.broadcast"(%5) {broadcast_dimensions = array<i64: 0>} : (tensor<3x!pphlo.secret<f32>>) -> tensor<3x4x!pphlo.secret<f32>>
    %7 = "pphlo.divide"(%0, %6) : (tensor<3x4x!pphlo.secret<f32>>, tensor<3x4x!pphlo.secret<f32>>) -> tensor<3x4x!pphlo.secret<f32>>
    return %7 : tensor<3x4x!pphlo.secret<f32>>
}
