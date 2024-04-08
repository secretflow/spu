// RUN: pphlo-opt --insert-deallocation --split-input-file %s | FileCheck %s

func.func @main() -> (tensor<f32>) {
    %0 = pphlo.constant dense<0xFF800000> : tensor<f32>
    %1 = pphlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = pphlo.less %0, %1: (tensor<f32>, tensor<f32>) -> tensor<i1>
    %3 = pphlo.select %2, %0, %1: (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    //CHECK: pphlo.free %1
    //CHECK: pphlo.free %0
    //CHECK: pphlo.free %2
    //CHECK-NOT: pphlo.free %3
    return %3: tensor<f32>
}
