module @cluster_0__XlaCompiledKernel_true__XlaHasReferenceVars_false__XlaNumConstantArgs_6__XlaNumResourceArgs_1_.73 {
  func @main(%arg0: tensor<100x!pphlo.pub<i32>>, %arg1: tensor<100x1x!pphlo.pub<f32>>, %arg2: tensor<!pphlo.pub<f32>>) -> (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) {
    %0 = "pphlo.constant"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<!pphlo.pub<f32>>
    %1 = "pphlo.constant"() {value = dense<0.000000e+00> : tensor<100x1xf32>} : () -> tensor<100x1x!pphlo.pub<f32>>
    %2 = "pphlo.constant"() {value = dense<1.000000e+02> : tensor<f32>} : () -> tensor<!pphlo.pub<f32>>
    %3 = "pphlo.less"(%arg1, %1) : (tensor<100x1x!pphlo.pub<f32>>, tensor<100x1x!pphlo.pub<f32>>) -> tensor<100x1x!pphlo.pub<i1>>
    %4 = "pphlo.not"(%3) : (tensor<100x1x!pphlo.pub<i1>>) -> tensor<100x1x!pphlo.pub<i1>>
    %5 = "pphlo.negate"(%arg1) : (tensor<100x1x!pphlo.pub<f32>>) -> tensor<100x1x!pphlo.pub<f32>>
    %6 = "pphlo.select"(%4, %5, %arg1) : (tensor<100x1x!pphlo.pub<i1>>, tensor<100x1x!pphlo.pub<f32>>, tensor<100x1x!pphlo.pub<f32>>) -> tensor<100x1x!pphlo.pub<f32>>
    %7 = "pphlo.exponential"(%6) : (tensor<100x1x!pphlo.pub<f32>>) -> tensor<100x1x!pphlo.pub<f32>>
    %8 = "pphlo.log_plus_one"(%7) : (tensor<100x1x!pphlo.pub<f32>>) -> tensor<100x1x!pphlo.pub<f32>>
    %9 = "pphlo.select"(%4, %arg1, %1) : (tensor<100x1x!pphlo.pub<i1>>, tensor<100x1x!pphlo.pub<f32>>, tensor<100x1x!pphlo.pub<f32>>) -> tensor<100x1x!pphlo.pub<f32>>
    %10 = "pphlo.convert"(%arg0) : (tensor<100x!pphlo.pub<i32>>) -> tensor<100x!pphlo.pub<f32>>
    %11 = "pphlo.reshape"(%10) : (tensor<100x!pphlo.pub<f32>>) -> tensor<100x1x!pphlo.pub<f32>>
    %12 = "pphlo.multiply"(%11, %arg1) : (tensor<100x1x!pphlo.pub<f32>>, tensor<100x1x!pphlo.pub<f32>>) -> tensor<100x1x!pphlo.pub<f32>>
    %13 = "pphlo.subtract"(%9, %12) : (tensor<100x1x!pphlo.pub<f32>>, tensor<100x1x!pphlo.pub<f32>>) -> tensor<100x1x!pphlo.pub<f32>>
    %14 = "pphlo.add"(%8, %13) : (tensor<100x1x!pphlo.pub<f32>>, tensor<100x1x!pphlo.pub<f32>>) -> tensor<100x1x!pphlo.pub<f32>>
    %15 = "pphlo.reduce"(%14, %0) ({
    ^bb0(%arg3: tensor<!pphlo.pub<f32>>, %arg4: tensor<!pphlo.pub<f32>>):
      %17 = "pphlo.add"(%arg3, %arg4) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
      "pphlo.return"(%17) : (tensor<!pphlo.pub<f32>>) -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<100x1x!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    %16 = "pphlo.add"(%arg2, %15) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    return %2, %16 : tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>
  }
}
