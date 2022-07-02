// RUN: mlir-pphlo-opt --hlo-legalize-to-pphlo='io-visibility-json={"inputs":["VIS_SECRET","VIS_SECRET"]}' %s --split-input-file  | FileCheck %s

func @main(%arg0: tensor<10xf64>, %arg1: tensor<10xf64>) -> (tensor<10xf64>, tensor<10xf64>) {
    // CHECK: return %arg0, %arg1 : tensor<10x!pphlo.sec<f64>>, tensor<10x!pphlo.sec<f64>>
    return %arg0, %arg1 : tensor<10xf64>, tensor<10xf64>
}