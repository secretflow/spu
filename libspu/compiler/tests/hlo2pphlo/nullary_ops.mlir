// RUN: mlir-pphlo-opt --hlo-legalize-to-pphlo --split-input-file %s | FileCheck %s

func.func @main() -> (tensor<2x2xi1>) {
    // CHECK: "pphlo.constant"() {value = dense<true> : tensor<2x2xi1>} : () -> tensor<2x2xi1>
    %0 = stablehlo.constant dense<true> : tensor<2x2xi1>
    // CHECK: "pphlo.constant"() {value = dense<1> : tensor<2x2xi8>} : () -> tensor<2x2xi8>
    %1 = stablehlo.constant dense<1> : tensor<2x2xi8>
    // CHECK: "pphlo.constant"() {value = dense<1> : tensor<2x2xi16>} : () -> tensor<2x2xi16>
    %2 = stablehlo.constant dense<1> : tensor<2x2xi16>
    // CHECK: "pphlo.constant"() {value = dense<1> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
    %3 = stablehlo.constant dense<1> : tensor<2x2xi32>
    // CHECK: "pphlo.constant"() {value = dense<1> : tensor<2x2xi64>} : () -> tensor<2x2xi64>
    %4 = stablehlo.constant dense<1> : tensor<2x2xi64>
    // CHECK: "pphlo.constant"() {value = dense<1> : tensor<2x2xui8>} : () -> tensor<2x2xui8>
    %5 = stablehlo.constant dense<1> : tensor<2x2xui8>
    // CHECK: "pphlo.constant"() {value = dense<1> : tensor<2x2xui16>} : () -> tensor<2x2xui16>
    %6 = stablehlo.constant dense<1> : tensor<2x2xui16>
    // CHECK: "pphlo.constant"() {value = dense<1> : tensor<2x2xui32>} : () -> tensor<2x2xui32>
    %7 = stablehlo.constant dense<1> : tensor<2x2xui32>
    // CHECK: "pphlo.constant"() {value = dense<1> : tensor<2x2xui64>} : () -> tensor<2x2xui64>
    %8 = stablehlo.constant dense<1> : tensor<2x2xui64>
    // CHECK: "pphlo.constant"() {value = dense<1.000000e+00> : tensor<2x2xf16>} : () -> tensor<2x2xf16>
    %9 = stablehlo.constant dense<1.0> : tensor<2x2xf16>
    // CHECK: "pphlo.constant"() {value = dense<1.000000e+00> : tensor<2x2xf32>} : () -> tensor<2x2xf32>
    %10 = stablehlo.constant dense<1.0> : tensor<2x2xf32>
    // CHECK: "pphlo.constant"() {value = dense<1.000000e+00> : tensor<2x2xf64>} : () -> tensor<2x2xf64>
    %11 = stablehlo.constant dense<1.0> : tensor<2x2xf64>
    return %0 : tensor<2x2xi1>
}


