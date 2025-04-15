// RUN: spu-opt --hlo-legalize-to-pphlo=input_vis_list=VIS_SECRET,VIS_SECRET --lower-conversion-cast %s --split-input-file  | FileCheck %s

func.func @main(%arg0: tensor<10xf64>, %arg1: tensor<10xf64>) -> (tensor<10xf64>, tensor<10xf64>) {
    // CHECK: return %arg0, %arg1 : tensor<10x!pphlo.secret<f64>>, tensor<10x!pphlo.secret<f64>>
    return %arg0, %arg1 : tensor<10xf64>, tensor<10xf64>
}