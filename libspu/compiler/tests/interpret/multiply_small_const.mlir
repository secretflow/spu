// RUN: spu-translate --protocol_kind=1 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=2 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=3 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=4 --interpret -split-input-file %s

func.func @multiply_small_const_positive() {
    %0 = arith.constant dense<9.99999997E-7> : tensor<1xf32>
    %1 = arith.constant dense<10000000.0> : tensor<1xf32>
    %2 = pphlo.convert %1 : (tensor<1xf32>)->tensor<1x!pphlo.secret<f32>>
    %3 = pphlo.multiply %2, %0 : (tensor<1x!pphlo.secret<f32>>, tensor<1xf32>) -> tensor<1x!pphlo.secret<f32>>
    %expected = arith.constant dense<10.0> : tensor<1xf32>
    pphlo.custom_call @expect_almost_eq(%expected, %3) : (tensor<1xf32>, tensor<1x!pphlo.secret<f32>>)->()
    func.return
}

// -----

func.func @multiply_small_const_negative() {
    %0 = arith.constant dense<-9.99999997E-7> : tensor<1xf32>
    %1 = arith.constant dense<10000000.0> : tensor<1xf32>
    %2 = pphlo.convert %1 : (tensor<1xf32>)->tensor<1x!pphlo.secret<f32>>
    %3 = pphlo.multiply %2, %0 : (tensor<1x!pphlo.secret<f32>>, tensor<1xf32>) -> tensor<1x!pphlo.secret<f32>>
    %expected = arith.constant dense<-10.0> : tensor<1xf32>
    pphlo.custom_call @expect_almost_eq(%expected, %3) : (tensor<1xf32>, tensor<1x!pphlo.secret<f32>>)->()
    func.return
}
