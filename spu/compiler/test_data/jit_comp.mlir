module @a_inference_comp_27__.20 {
  func @main(%arg0: tensor<2x2x!pphlo.pub<i32>>, %arg1: tensor<2x2x!pphlo.pub<i32>>) -> (tensor<2x2x!pphlo.pub<i1>>, tensor<2x2x!pphlo.pub<i1>>, tensor<2x2x!pphlo.pub<i1>>, tensor<2x2x!pphlo.pub<i1>>, tensor<2x2x!pphlo.pub<i1>>, tensor<2x2x!pphlo.pub<i1>>, tensor<2x2x!pphlo.pub<i1>>) {
    %0 = "pphlo.constant"() {value = dense<true> : tensor<2x2xi1>} : () -> tensor<2x2x!pphlo.pub<i1>>
    %1 = "pphlo.equal"(%arg0, %arg1) : (tensor<2x2x!pphlo.pub<i32>>, tensor<2x2x!pphlo.pub<i32>>) -> tensor<2x2x!pphlo.pub<i1>>
    %2 = "pphlo.not"(%1) : (tensor<2x2x!pphlo.pub<i1>>) -> tensor<2x2x!pphlo.pub<i1>>
    %3 = "pphlo.less"(%arg0, %arg1) : (tensor<2x2x!pphlo.pub<i32>>, tensor<2x2x!pphlo.pub<i32>>) -> tensor<2x2x!pphlo.pub<i1>>
    %4 = "pphlo.greater"(%arg0, %arg1) : (tensor<2x2x!pphlo.pub<i32>>, tensor<2x2x!pphlo.pub<i32>>) -> tensor<2x2x!pphlo.pub<i1>>
    %5 = "pphlo.not"(%4) : (tensor<2x2x!pphlo.pub<i1>>) -> tensor<2x2x!pphlo.pub<i1>>
    %6 = "pphlo.not"(%3) : (tensor<2x2x!pphlo.pub<i1>>) -> tensor<2x2x!pphlo.pub<i1>>
    return %1, %2, %3, %4, %5, %6, %0 : tensor<2x2x!pphlo.pub<i1>>, tensor<2x2x!pphlo.pub<i1>>, tensor<2x2x!pphlo.pub<i1>>, tensor<2x2x!pphlo.pub<i1>>, tensor<2x2x!pphlo.pub<i1>>, tensor<2x2x!pphlo.pub<i1>>, tensor<2x2x!pphlo.pub<i1>>
  }
}
