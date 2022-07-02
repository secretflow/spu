// RUN: mlir-pphlo-opt --hlo-legalize-to-pphlo --split-input-file %s | FileCheck %s

func @main() -> (tensor<2x2xi1>) {
    // CHECK: "pphlo.constant"() {value = dense<true> : tensor<2x2xi1>} : () -> tensor<2x2x!pphlo.pub<i1>>
    %0 = mhlo.constant dense<true> : tensor<2x2xi1>
    // CHECK: "pphlo.constant"() {value = dense<1> : tensor<2x2xi8>} : () -> tensor<2x2x!pphlo.pub<i8>> 
    %1 = mhlo.constant dense<1> : tensor<2x2xi8>
    // CHECK: "pphlo.constant"() {value = dense<1> : tensor<2x2xi16>} : () -> tensor<2x2x!pphlo.pub<i16>> 
    %2 = mhlo.constant dense<1> : tensor<2x2xi16>
    // CHECK: "pphlo.constant"() {value = dense<1> : tensor<2x2xi32>} : () -> tensor<2x2x!pphlo.pub<i32>> 
    %3 = mhlo.constant dense<1> : tensor<2x2xi32>
    // CHECK: "pphlo.constant"() {value = dense<1> : tensor<2x2xi64>} : () -> tensor<2x2x!pphlo.pub<i64>> 
    %4 = mhlo.constant dense<1> : tensor<2x2xi64>
    // CHECK: "pphlo.constant"() {value = dense<1> : tensor<2x2xui8>} : () -> tensor<2x2x!pphlo.pub<ui8>> 
    %5 = mhlo.constant dense<1> : tensor<2x2xui8>
    // CHECK: "pphlo.constant"() {value = dense<1> : tensor<2x2xui16>} : () -> tensor<2x2x!pphlo.pub<ui16>> 
    %6 = mhlo.constant dense<1> : tensor<2x2xui16>
    // CHECK: "pphlo.constant"() {value = dense<1> : tensor<2x2xui32>} : () -> tensor<2x2x!pphlo.pub<ui32>> 
    %7 = mhlo.constant dense<1> : tensor<2x2xui32>
    // CHECK: "pphlo.constant"() {value = dense<1> : tensor<2x2xui64>} : () -> tensor<2x2x!pphlo.pub<ui64>> 
    %8 = mhlo.constant dense<1> : tensor<2x2xui64>
    // CHECK: "pphlo.constant"() {value = dense<1.000000e+00> : tensor<2x2xf32>} : () -> tensor<2x2x!pphlo.pub<f32>> 
    %9 = mhlo.constant dense<1.0> : tensor<2x2xf32>
    // CHECK: "pphlo.constant"() {value = dense<1.000000e+00> : tensor<2x2xf64>} : () -> tensor<2x2x!pphlo.pub<f64>> 
    %10 = mhlo.constant dense<1.0> : tensor<2x2xf64>
    return %0 : tensor<2x2xi1>
}


