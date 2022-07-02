module @cluster_0__XlaCompiledKernel_true__XlaHasReferenceVars_false__XlaNumConstantArgs_21__XlaNumResourceArgs_5_.200 {
  func @main(%arg0: tensor<100x!pphlo.pub<i32>>, %arg1: tensor<100x1x!pphlo.pub<f32>>, %arg2: tensor<!pphlo.pub<f32>>, %arg3: tensor<1x!pphlo.pub<f32>>, %arg4: tensor<1x!pphlo.pub<f32>>, %arg5: tensor<1x!pphlo.pub<f32>>, %arg6: tensor<1x!pphlo.pub<f32>>) -> (tensor<!pphlo.pub<i1>>, tensor<100x1x!pphlo.pub<f32>>, tensor<!pphlo.pub<i1>>, tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>) {
    %0 = "pphlo.constant"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<!pphlo.pub<f32>>
    %1 = "pphlo.constant"() {value = dense<0.000000e+00> : tensor<100x1xf32>} : () -> tensor<100x1x!pphlo.pub<f32>>
    %2 = "pphlo.constant"() {value = dense<5.000000e-01> : tensor<100x1xf32>} : () -> tensor<100x1x!pphlo.pub<f32>>
    %3 = "pphlo.constant"() {value = dense<1.000000e+02> : tensor<f32>} : () -> tensor<!pphlo.pub<f32>>
    %4 = "pphlo.constant"() {value = dense<true> : tensor<i1>} : () -> tensor<!pphlo.pub<i1>>
    %5 = "pphlo.constant"() {value = dense<1.000000e+00> : tensor<100x1xf32>} : () -> tensor<100x1x!pphlo.pub<f32>>
    %6 = "pphlo.logistic"(%arg1) : (tensor<100x1x!pphlo.pub<f32>>) -> tensor<100x1x!pphlo.pub<f32>>
    %7 = "pphlo.less"(%6, %1) : (tensor<100x1x!pphlo.pub<f32>>, tensor<100x1x!pphlo.pub<f32>>) -> tensor<100x1x!pphlo.pub<i1>>
    %8 = "pphlo.not"(%7) : (tensor<100x1x!pphlo.pub<i1>>) -> tensor<100x1x!pphlo.pub<i1>>
    %9 = "pphlo.reduce"(%8, %4) ({
    ^bb0(%arg7: tensor<!pphlo.pub<i1>>, %arg8: tensor<!pphlo.pub<i1>>):
      %47 = "pphlo.and"(%arg7, %arg8) : (tensor<!pphlo.pub<i1>>, tensor<!pphlo.pub<i1>>) -> tensor<!pphlo.pub<i1>>
      "pphlo.return"(%47) : (tensor<!pphlo.pub<i1>>) -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<100x1x!pphlo.pub<i1>>, tensor<!pphlo.pub<i1>>) -> tensor<!pphlo.pub<i1>>
    %10 = "pphlo.greater"(%6, %5) : (tensor<100x1x!pphlo.pub<f32>>, tensor<100x1x!pphlo.pub<f32>>) -> tensor<100x1x!pphlo.pub<i1>>
    %11 = "pphlo.not"(%10) : (tensor<100x1x!pphlo.pub<i1>>) -> tensor<100x1x!pphlo.pub<i1>>
    %12 = "pphlo.reduce"(%11, %4) ({
    ^bb0(%arg7: tensor<!pphlo.pub<i1>>, %arg8: tensor<!pphlo.pub<i1>>):
      %47 = "pphlo.and"(%arg7, %arg8) : (tensor<!pphlo.pub<i1>>, tensor<!pphlo.pub<i1>>) -> tensor<!pphlo.pub<i1>>
      "pphlo.return"(%47) : (tensor<!pphlo.pub<i1>>) -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<100x1x!pphlo.pub<i1>>, tensor<!pphlo.pub<i1>>) -> tensor<!pphlo.pub<i1>>
    %13 = "pphlo.less"(%arg1, %1) : (tensor<100x1x!pphlo.pub<f32>>, tensor<100x1x!pphlo.pub<f32>>) -> tensor<100x1x!pphlo.pub<i1>>
    %14 = "pphlo.not"(%13) : (tensor<100x1x!pphlo.pub<i1>>) -> tensor<100x1x!pphlo.pub<i1>>
    %15 = "pphlo.negate"(%arg1) : (tensor<100x1x!pphlo.pub<f32>>) -> tensor<100x1x!pphlo.pub<f32>>
    %16 = "pphlo.select"(%14, %15, %arg1) : (tensor<100x1x!pphlo.pub<i1>>, tensor<100x1x!pphlo.pub<f32>>, tensor<100x1x!pphlo.pub<f32>>) -> tensor<100x1x!pphlo.pub<f32>>
    %17 = "pphlo.exponential"(%16) : (tensor<100x1x!pphlo.pub<f32>>) -> tensor<100x1x!pphlo.pub<f32>>
    %18 = "pphlo.log_plus_one"(%17) : (tensor<100x1x!pphlo.pub<f32>>) -> tensor<100x1x!pphlo.pub<f32>>
    %19 = "pphlo.select"(%14, %arg1, %1) : (tensor<100x1x!pphlo.pub<i1>>, tensor<100x1x!pphlo.pub<f32>>, tensor<100x1x!pphlo.pub<f32>>) -> tensor<100x1x!pphlo.pub<f32>>
    %20 = "pphlo.convert"(%arg0) : (tensor<100x!pphlo.pub<i32>>) -> tensor<100x!pphlo.pub<f32>>
    %21 = "pphlo.reshape"(%20) : (tensor<100x!pphlo.pub<f32>>) -> tensor<100x1x!pphlo.pub<f32>>
    %22 = "pphlo.multiply"(%21, %arg1) : (tensor<100x1x!pphlo.pub<f32>>, tensor<100x1x!pphlo.pub<f32>>) -> tensor<100x1x!pphlo.pub<f32>>
    %23 = "pphlo.subtract"(%19, %22) : (tensor<100x1x!pphlo.pub<f32>>, tensor<100x1x!pphlo.pub<f32>>) -> tensor<100x1x!pphlo.pub<f32>>
    %24 = "pphlo.add"(%18, %23) : (tensor<100x1x!pphlo.pub<f32>>, tensor<100x1x!pphlo.pub<f32>>) -> tensor<100x1x!pphlo.pub<f32>>
    %25 = "pphlo.reduce"(%24, %0) ({
    ^bb0(%arg7: tensor<!pphlo.pub<f32>>, %arg8: tensor<!pphlo.pub<f32>>):
      %47 = "pphlo.add"(%arg7, %arg8) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
      "pphlo.return"(%47) : (tensor<!pphlo.pub<f32>>) -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<100x1x!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    %26 = "pphlo.add"(%arg2, %25) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    %27 = "pphlo.greater"(%6, %2) : (tensor<100x1x!pphlo.pub<f32>>, tensor<100x1x!pphlo.pub<f32>>) -> tensor<100x1x!pphlo.pub<i1>>
    %28 = "pphlo.reshape"(%27) : (tensor<100x1x!pphlo.pub<i1>>) -> tensor<1x100x!pphlo.pub<i1>>
    %29 = "pphlo.equal"(%21, %1) : (tensor<100x1x!pphlo.pub<f32>>, tensor<100x1x!pphlo.pub<f32>>) -> tensor<100x1x!pphlo.pub<i1>>
    %30 = "pphlo.not"(%29) : (tensor<100x1x!pphlo.pub<i1>>) -> tensor<100x1x!pphlo.pub<i1>>
    %31 = "pphlo.reshape"(%30) : (tensor<100x1x!pphlo.pub<i1>>) -> tensor<1x100x!pphlo.pub<i1>>
    %32 = "pphlo.and"(%28, %31) : (tensor<1x100x!pphlo.pub<i1>>, tensor<1x100x!pphlo.pub<i1>>) -> tensor<1x100x!pphlo.pub<i1>>
    %33 = "pphlo.convert"(%32) : (tensor<1x100x!pphlo.pub<i1>>) -> tensor<1x100x!pphlo.pub<f32>>
    %34 = "pphlo.reduce"(%33, %0) ({
    ^bb0(%arg7: tensor<!pphlo.pub<f32>>, %arg8: tensor<!pphlo.pub<f32>>):
      %47 = "pphlo.add"(%arg7, %arg8) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
      "pphlo.return"(%47) : (tensor<!pphlo.pub<f32>>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<1x100x!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<1x!pphlo.pub<f32>>
    %35 = "pphlo.add"(%arg3, %34) : (tensor<1x!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>) -> tensor<1x!pphlo.pub<f32>>
    %36 = "pphlo.add"(%arg4, %34) : (tensor<1x!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>) -> tensor<1x!pphlo.pub<f32>>
    %37 = "pphlo.not"(%31) : (tensor<1x100x!pphlo.pub<i1>>) -> tensor<1x100x!pphlo.pub<i1>>
    %38 = "pphlo.and"(%28, %37) : (tensor<1x100x!pphlo.pub<i1>>, tensor<1x100x!pphlo.pub<i1>>) -> tensor<1x100x!pphlo.pub<i1>>
    %39 = "pphlo.convert"(%38) : (tensor<1x100x!pphlo.pub<i1>>) -> tensor<1x100x!pphlo.pub<f32>>
    %40 = "pphlo.reduce"(%39, %0) ({
    ^bb0(%arg7: tensor<!pphlo.pub<f32>>, %arg8: tensor<!pphlo.pub<f32>>):
      %47 = "pphlo.add"(%arg7, %arg8) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
      "pphlo.return"(%47) : (tensor<!pphlo.pub<f32>>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<1x100x!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<1x!pphlo.pub<f32>>
    %41 = "pphlo.add"(%arg5, %40) : (tensor<1x!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>) -> tensor<1x!pphlo.pub<f32>>
    %42 = "pphlo.not"(%28) : (tensor<1x100x!pphlo.pub<i1>>) -> tensor<1x100x!pphlo.pub<i1>>
    %43 = "pphlo.and"(%42, %31) : (tensor<1x100x!pphlo.pub<i1>>, tensor<1x100x!pphlo.pub<i1>>) -> tensor<1x100x!pphlo.pub<i1>>
    %44 = "pphlo.convert"(%43) : (tensor<1x100x!pphlo.pub<i1>>) -> tensor<1x100x!pphlo.pub<f32>>
    %45 = "pphlo.reduce"(%44, %0) ({
    ^bb0(%arg7: tensor<!pphlo.pub<f32>>, %arg8: tensor<!pphlo.pub<f32>>):
      %47 = "pphlo.add"(%arg7, %arg8) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
      "pphlo.return"(%47) : (tensor<!pphlo.pub<f32>>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<1x100x!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<1x!pphlo.pub<f32>>
    %46 = "pphlo.add"(%arg6, %45) : (tensor<1x!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>) -> tensor<1x!pphlo.pub<f32>>
    return %9, %6, %12, %3, %26, %35, %36, %41, %46 : tensor<!pphlo.pub<i1>>, tensor<100x1x!pphlo.pub<f32>>, tensor<!pphlo.pub<i1>>, tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>
  }
}
