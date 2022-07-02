module @xla_computation_selu.26 {
  func @main(%arg0: tensor<100x!pphlo.pub<f32>>) -> tensor<100x!pphlo.pub<f32>> {
    %0 = "pphlo.constant"() {value = dense<1.050000e+00> : tensor<100xf32>} : () -> tensor<100x!pphlo.pub<f32>>
    %1 = "pphlo.constant"() {value = dense<-1.670000e+00> : tensor<100xf32>} : () -> tensor<100x!pphlo.pub<f32>>
    %2 = "pphlo.constant"() {value = dense<1.670000e+00> : tensor<100xf32>} : () -> tensor<100x!pphlo.pub<f32>>
    %3 = "pphlo.constant"() {value = dense<0.000000e+00> : tensor<100xf32>} : () -> tensor<100x!pphlo.pub<f32>>
    %4 = "pphlo.greater"(%arg0, %3) : (tensor<100x!pphlo.pub<f32>>, tensor<100x!pphlo.pub<f32>>) -> tensor<100x!pphlo.pub<i1>>
    %5 = "pphlo.exponential"(%arg0) : (tensor<100x!pphlo.pub<f32>>) -> tensor<100x!pphlo.pub<f32>>
    %6 = "pphlo.multiply"(%5, %2) : (tensor<100x!pphlo.pub<f32>>, tensor<100x!pphlo.pub<f32>>) -> tensor<100x!pphlo.pub<f32>>
    %7 = "pphlo.add"(%6, %1) : (tensor<100x!pphlo.pub<f32>>, tensor<100x!pphlo.pub<f32>>) -> tensor<100x!pphlo.pub<f32>>
    %8 = "pphlo.select"(%4, %arg0, %7) : (tensor<100x!pphlo.pub<i1>>, tensor<100x!pphlo.pub<f32>>, tensor<100x!pphlo.pub<f32>>) -> tensor<100x!pphlo.pub<f32>>
    %9 = "pphlo.multiply"(%8, %0) : (tensor<100x!pphlo.pub<f32>>, tensor<100x!pphlo.pub<f32>>) -> tensor<100x!pphlo.pub<f32>>
    return %9 : tensor<100x!pphlo.pub<f32>>
  }
}
