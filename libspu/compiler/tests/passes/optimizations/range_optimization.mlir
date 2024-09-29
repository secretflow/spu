// RUN: spu-opt --secret-decompose-ops --lower-sfloat-to-fxp --expand-fixedpoint-approx --canonicalize --cse --range-optimization --cse --split-input-file %s | FileCheck %s

func.func @main(%arg0: tensor<2x2x!pphlo.secret<ui32>>, %arg1: tensor<2x2x!pphlo.secret<ui32>>) -> (tensor<2x2x!pphlo.secret<ui32>>) {
    //CHECK-NOT: pphlo.sign
    %0 = pphlo.divide %arg0, %arg1 : tensor<2x2x!pphlo.secret<ui32>>
    return %0 : tensor<2x2x!pphlo.secret<ui32>>
}
