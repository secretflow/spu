// RUN: mlir-pphlo-opt --hlo-legalize-to-pphlo=input_vis_list=VIS_SECRET --lower-conversion-cast --split-input-file %s | FileCheck %s

func.func @main(%arg0: tensor<3xcomplex<f32>>) -> tensor<3xcomplex<f32>> {
    // CHECK: %[[REAL:.+]] = pphlo.real %[[ARG0:.+]] : (tensor<3x!pphlo.secret<complex<f32>>>) -> tensor<3x!pphlo.secret<f32>>
    %0 = stablehlo.real %arg0 : (tensor<3xcomplex<f32>>) -> tensor<3xf32>
    // CHECK: %[[IMAG:.+]] = pphlo.imag %[[ARG1:.+]] : (tensor<3x!pphlo.secret<complex<f32>>>) -> tensor<3x!pphlo.secret<f32>>
    %1 = stablehlo.imag %arg0 : (tensor<3xcomplex<f32>>) -> tensor<3xf32>
    // CHECK: pphlo.complex %[[REAL]], %[[IMAG]] : (tensor<3x!pphlo.secret<f32>>, tensor<3x!pphlo.secret<f32>>) -> tensor<3x!pphlo.secret<complex<f32>>>
    %2 = stablehlo.complex %0, %1 : tensor<3xcomplex<f32>>
    return %2 : tensor<3xcomplex<f32>>
  }
