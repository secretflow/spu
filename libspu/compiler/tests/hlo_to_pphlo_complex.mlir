// RUN: mlir-pphlo-opt --hlo-legalize-to-pphlo=input_vis_list=VIS_PUBLIC --split-input-file %s | FileCheck %s

func.func @main(%arg0: tensor<3xcomplex<f32>>) -> tensor<3xcomplex<f32>> {
    // CHECK: pphlo.real
    %0 = stablehlo.real %arg0 : (tensor<3xcomplex<f32>>) -> tensor<3xf32>
    // CHECK: pphlo.imag
    %1 = stablehlo.imag %arg0 : (tensor<3xcomplex<f32>>) -> tensor<3xf32>
    // CHECK: pphlo.complex
    %2 = stablehlo.complex %0, %1 : tensor<3xcomplex<f32>>
    return %2 : tensor<3xcomplex<f32>>
  }
