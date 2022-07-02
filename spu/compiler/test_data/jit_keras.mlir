module @a_inference_train_step_2176__.301 {
  func @main(%arg0: tensor<1024x16x!pphlo.pub<f64>>, %arg1: tensor<1024x7x!pphlo.pub<f64>>, %arg2: tensor<1024x1x!pphlo.pub<f64>>, %arg3: tensor<16x!pphlo.pub<f32>>, %arg4: tensor<16x!pphlo.pub<f32>>, %arg5: tensor<7x!pphlo.pub<f32>>, %arg6: tensor<7x!pphlo.pub<f32>>, %arg7: tensor<23x1x!pphlo.pub<f32>>, %arg8: tensor<1x!pphlo.pub<f32>>, %arg9: tensor<!pphlo.pub<f32>>, %arg10: tensor<!pphlo.pub<f32>>, %arg11: tensor<!pphlo.pub<i64>>, %arg12: tensor<!pphlo.pub<f32>>) -> (tensor<!pphlo.pub<f32>>, tensor<23x1x!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<i64>>) {
    %0 = "pphlo.constant"() {value = dense<1> : tensor<i64>} : () -> tensor<!pphlo.pub<i64>>
    %1 = "pphlo.constant"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<!pphlo.pub<f32>>
    %2 = "pphlo.constant"() {value = dense<0.000000e+00> : tensor<1024x1xf32>} : () -> tensor<1024x1x!pphlo.pub<f32>>
    %3 = "pphlo.constant"() {value = dense<1.000000e+00> : tensor<1024x1xf32>} : () -> tensor<1024x1x!pphlo.pub<f32>>
    %4 = "pphlo.constant"() {value = dense<1.000000e+00> : tensor<f32>} : () -> tensor<!pphlo.pub<f32>>
    %5 = "pphlo.constant"() {value = dense<9.765625E-4> : tensor<1024x1xf32>} : () -> tensor<1024x1x!pphlo.pub<f32>>
    %6 = "pphlo.constant"() {value = dense<-9.765625E-4> : tensor<1024x1xf32>} : () -> tensor<1024x1x!pphlo.pub<f32>>
    %7 = "pphlo.constant"() {value = dense<1.000000e-01> : tensor<f32>} : () -> tensor<!pphlo.pub<f32>>
    %8 = "pphlo.constant"() {value = dense<0.899999976> : tensor<f32>} : () -> tensor<!pphlo.pub<f32>>
    %9 = "pphlo.constant"() {value = dense<-1.000000e+00> : tensor<f32>} : () -> tensor<!pphlo.pub<f32>>
    %10 = "pphlo.constant"() {value = dense<2.000000e+00> : tensor<f32>} : () -> tensor<!pphlo.pub<f32>>
    %11 = "pphlo.constant"() {value = dense<0.0018248175> : tensor<f32>} : () -> tensor<!pphlo.pub<f32>>
    %12 = "pphlo.constant"() {value = dense<0.00364963501> : tensor<f32>} : () -> tensor<!pphlo.pub<f32>>
    %13 = "pphlo.constant"() {value = dense<1.000000e-07> : tensor<7xf32>} : () -> tensor<7x!pphlo.pub<f32>>
    %14 = "pphlo.constant"() {value = dense<1.000000e-07> : tensor<16xf32>} : () -> tensor<16x!pphlo.pub<f32>>
    %15 = "pphlo.constant"() {value = dense<1.024000e+03> : tensor<f32>} : () -> tensor<!pphlo.pub<f32>>
    %16 = "pphlo.add"(%arg10, %15) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    %17 = "pphlo.equal"(%16, %1) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<i1>>
    %18 = "pphlo.convert"(%arg0) : (tensor<1024x16x!pphlo.pub<f64>>) -> tensor<1024x16x!pphlo.pub<f32>>
    %19 = "pphlo.broadcast"(%arg3) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<16x!pphlo.pub<f32>>) -> tensor<1024x16x!pphlo.pub<f32>>
    %20 = "pphlo.subtract"(%18, %19) : (tensor<1024x16x!pphlo.pub<f32>>, tensor<1024x16x!pphlo.pub<f32>>) -> tensor<1024x16x!pphlo.pub<f32>>
    %21 = "pphlo.constant"() {value = dense<5.000000e-01> : tensor<16xf32>} : () -> tensor<16x!pphlo.pub<f32>>
    %22 = "pphlo.power"(%arg4, %21) : (tensor<16x!pphlo.pub<f32>>, tensor<16x!pphlo.pub<f32>>) -> tensor<16x!pphlo.pub<f32>>
    %23 = "pphlo.maximum"(%22, %14) : (tensor<16x!pphlo.pub<f32>>, tensor<16x!pphlo.pub<f32>>) -> tensor<16x!pphlo.pub<f32>>
    %24 = "pphlo.broadcast"(%23) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<16x!pphlo.pub<f32>>) -> tensor<1024x16x!pphlo.pub<f32>>
    %25 = "pphlo.divide"(%20, %24) : (tensor<1024x16x!pphlo.pub<f32>>, tensor<1024x16x!pphlo.pub<f32>>) -> tensor<1024x16x!pphlo.pub<f32>>
    %26 = "pphlo.convert"(%arg1) : (tensor<1024x7x!pphlo.pub<f64>>) -> tensor<1024x7x!pphlo.pub<f32>>
    %27 = "pphlo.broadcast"(%arg5) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<7x!pphlo.pub<f32>>) -> tensor<1024x7x!pphlo.pub<f32>>
    %28 = "pphlo.subtract"(%26, %27) : (tensor<1024x7x!pphlo.pub<f32>>, tensor<1024x7x!pphlo.pub<f32>>) -> tensor<1024x7x!pphlo.pub<f32>>
    %29 = "pphlo.constant"() {value = dense<5.000000e-01> : tensor<7xf32>} : () -> tensor<7x!pphlo.pub<f32>>
    %30 = "pphlo.power"(%arg6, %29) : (tensor<7x!pphlo.pub<f32>>, tensor<7x!pphlo.pub<f32>>) -> tensor<7x!pphlo.pub<f32>>
    %31 = "pphlo.maximum"(%30, %13) : (tensor<7x!pphlo.pub<f32>>, tensor<7x!pphlo.pub<f32>>) -> tensor<7x!pphlo.pub<f32>>
    %32 = "pphlo.broadcast"(%31) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<7x!pphlo.pub<f32>>) -> tensor<1024x7x!pphlo.pub<f32>>
    %33 = "pphlo.divide"(%28, %32) : (tensor<1024x7x!pphlo.pub<f32>>, tensor<1024x7x!pphlo.pub<f32>>) -> tensor<1024x7x!pphlo.pub<f32>>
    %34 = "pphlo.concatenate"(%25, %33) {dimension = 1 : i64} : (tensor<1024x16x!pphlo.pub<f32>>, tensor<1024x7x!pphlo.pub<f32>>) -> tensor<1024x23x!pphlo.pub<f32>>
    %35 = "pphlo.reshape"(%arg7) : (tensor<23x1x!pphlo.pub<f32>>) -> tensor<23x!pphlo.pub<f32>>
    %36 = "pphlo.broadcast"(%35) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<23x!pphlo.pub<f32>>) -> tensor<1024x23x!pphlo.pub<f32>>
    %37 = "pphlo.multiply"(%34, %36) : (tensor<1024x23x!pphlo.pub<f32>>, tensor<1024x23x!pphlo.pub<f32>>) -> tensor<1024x23x!pphlo.pub<f32>>
    %38 = "pphlo.reduce"(%37, %1) ({
    ^bb0(%arg13: tensor<!pphlo.pub<f32>>, %arg14: tensor<!pphlo.pub<f32>>):
      %101 = "pphlo.add"(%arg13, %arg14) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
      "pphlo.return"(%101) : (tensor<!pphlo.pub<f32>>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<1024x23x!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<1024x!pphlo.pub<f32>>
    %39 = "pphlo.reshape"(%arg8) : (tensor<1x!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    %40 = "pphlo.broadcast"(%39) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pub<f32>>) -> tensor<1024x!pphlo.pub<f32>>
    %41 = "pphlo.add"(%38, %40) : (tensor<1024x!pphlo.pub<f32>>, tensor<1024x!pphlo.pub<f32>>) -> tensor<1024x!pphlo.pub<f32>>
    %42 = "pphlo.reshape"(%41) : (tensor<1024x!pphlo.pub<f32>>) -> tensor<1024x1x!pphlo.pub<f32>>
    %43 = "pphlo.less"(%42, %2) : (tensor<1024x1x!pphlo.pub<f32>>, tensor<1024x1x!pphlo.pub<f32>>) -> tensor<1024x1x!pphlo.pub<i1>>
    %44 = "pphlo.not"(%43) : (tensor<1024x1x!pphlo.pub<i1>>) -> tensor<1024x1x!pphlo.pub<i1>>
    %45 = "pphlo.select"(%44, %42, %2) : (tensor<1024x1x!pphlo.pub<i1>>, tensor<1024x1x!pphlo.pub<f32>>, tensor<1024x1x!pphlo.pub<f32>>) -> tensor<1024x1x!pphlo.pub<f32>>
    %46 = "pphlo.convert"(%arg2) : (tensor<1024x1x!pphlo.pub<f64>>) -> tensor<1024x1x!pphlo.pub<f32>>
    %47 = "pphlo.multiply"(%42, %46) : (tensor<1024x1x!pphlo.pub<f32>>, tensor<1024x1x!pphlo.pub<f32>>) -> tensor<1024x1x!pphlo.pub<f32>>
    %48 = "pphlo.subtract"(%45, %47) : (tensor<1024x1x!pphlo.pub<f32>>, tensor<1024x1x!pphlo.pub<f32>>) -> tensor<1024x1x!pphlo.pub<f32>>
    %49 = "pphlo.negate"(%42) : (tensor<1024x1x!pphlo.pub<f32>>) -> tensor<1024x1x!pphlo.pub<f32>>
    %50 = "pphlo.select"(%44, %49, %42) : (tensor<1024x1x!pphlo.pub<i1>>, tensor<1024x1x!pphlo.pub<f32>>, tensor<1024x1x!pphlo.pub<f32>>) -> tensor<1024x1x!pphlo.pub<f32>>
    %51 = "pphlo.exponential"(%50) : (tensor<1024x1x!pphlo.pub<f32>>) -> tensor<1024x1x!pphlo.pub<f32>>
    %52 = "pphlo.log_plus_one"(%51) : (tensor<1024x1x!pphlo.pub<f32>>) -> tensor<1024x1x!pphlo.pub<f32>>
    %53 = "pphlo.add"(%48, %52) : (tensor<1024x1x!pphlo.pub<f32>>, tensor<1024x1x!pphlo.pub<f32>>) -> tensor<1024x1x!pphlo.pub<f32>>
    %54 = "pphlo.reduce"(%53, %1) ({
    ^bb0(%arg13: tensor<!pphlo.pub<f32>>, %arg14: tensor<!pphlo.pub<f32>>):
      %101 = "pphlo.add"(%arg13, %arg14) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
      "pphlo.return"(%101) : (tensor<!pphlo.pub<f32>>) -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1024x1x!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    %55 = "pphlo.add"(%arg9, %54) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    %56 = "pphlo.divide"(%55, %16) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    %57 = "pphlo.select"(%17, %1, %56) : (tensor<!pphlo.pub<i1>>, tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    %58 = "pphlo.convert"(%arg11) : (tensor<!pphlo.pub<i64>>) -> tensor<!pphlo.pub<f32>>
    %59 = "pphlo.multiply"(%58, %12) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    %60 = "pphlo.multiply"(%58, %11) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    %61 = "pphlo.add"(%60, %4) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    %62 = "pphlo.floor"(%61) : (tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    %63 = "pphlo.multiply"(%62, %10) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    %64 = "pphlo.subtract"(%59, %63) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    %65 = "pphlo.add"(%64, %4) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    %66 = "pphlo.abs"(%65) : (tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    %67 = "pphlo.subtract"(%4, %66) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    %68 = "pphlo.maximum"(%67, %1) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    %69 = "pphlo.add"(%62, %9) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    %70 = "pphlo.negate"(%69) : (tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    %71 = "pphlo.power"(%10, %70) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    %72 = "pphlo.multiply"(%68, %71) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    %73 = "pphlo.multiply"(%72, %8) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    %74 = "pphlo.add"(%73, %7) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    %75 = "pphlo.broadcast"(%74) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pub<f32>>) -> tensor<23x!pphlo.pub<f32>>
    %76 = "pphlo.transpose"(%34) {permutation = dense<[1, 0]> : tensor<2xi64>, xla_shape = "f32[23,1024]{0,1}"} : (tensor<1024x23x!pphlo.pub<f32>>) -> tensor<23x1024x!pphlo.pub<f32>>
    %77 = "pphlo.select"(%44, %5, %2) : (tensor<1024x1x!pphlo.pub<i1>>, tensor<1024x1x!pphlo.pub<f32>>, tensor<1024x1x!pphlo.pub<f32>>) -> tensor<1024x1x!pphlo.pub<f32>>
    %78 = "pphlo.multiply"(%46, %6) : (tensor<1024x1x!pphlo.pub<f32>>, tensor<1024x1x!pphlo.pub<f32>>) -> tensor<1024x1x!pphlo.pub<f32>>
    %79 = "pphlo.add"(%77, %78) : (tensor<1024x1x!pphlo.pub<f32>>, tensor<1024x1x!pphlo.pub<f32>>) -> tensor<1024x1x!pphlo.pub<f32>>
    %80 = "pphlo.add"(%51, %3) : (tensor<1024x1x!pphlo.pub<f32>>, tensor<1024x1x!pphlo.pub<f32>>) -> tensor<1024x1x!pphlo.pub<f32>>
    %81 = "pphlo.divide"(%3, %80) : (tensor<1024x1x!pphlo.pub<f32>>, tensor<1024x1x!pphlo.pub<f32>>) -> tensor<1024x1x!pphlo.pub<f32>>
    %82 = "pphlo.multiply"(%81, %5) : (tensor<1024x1x!pphlo.pub<f32>>, tensor<1024x1x!pphlo.pub<f32>>) -> tensor<1024x1x!pphlo.pub<f32>>
    %83 = "pphlo.multiply"(%82, %51) : (tensor<1024x1x!pphlo.pub<f32>>, tensor<1024x1x!pphlo.pub<f32>>) -> tensor<1024x1x!pphlo.pub<f32>>
    %84 = "pphlo.select"(%44, %2, %83) : (tensor<1024x1x!pphlo.pub<i1>>, tensor<1024x1x!pphlo.pub<f32>>, tensor<1024x1x!pphlo.pub<f32>>) -> tensor<1024x1x!pphlo.pub<f32>>
    %85 = "pphlo.add"(%79, %84) : (tensor<1024x1x!pphlo.pub<f32>>, tensor<1024x1x!pphlo.pub<f32>>) -> tensor<1024x1x!pphlo.pub<f32>>
    %86 = "pphlo.select"(%44, %83, %2) : (tensor<1024x1x!pphlo.pub<i1>>, tensor<1024x1x!pphlo.pub<f32>>, tensor<1024x1x!pphlo.pub<f32>>) -> tensor<1024x1x!pphlo.pub<f32>>
    %87 = "pphlo.negate"(%86) : (tensor<1024x1x!pphlo.pub<f32>>) -> tensor<1024x1x!pphlo.pub<f32>>
    %88 = "pphlo.add"(%85, %87) : (tensor<1024x1x!pphlo.pub<f32>>, tensor<1024x1x!pphlo.pub<f32>>) -> tensor<1024x1x!pphlo.pub<f32>>
    %89 = "pphlo.reshape"(%88) : (tensor<1024x1x!pphlo.pub<f32>>) -> tensor<1024x!pphlo.pub<f32>>
    %90 = "pphlo.broadcast"(%89) {broadcast_dimensions = dense<1> : tensor<1xi64>, xla_shape = "f32[23,1024]{0,1}"} : (tensor<1024x!pphlo.pub<f32>>) -> tensor<23x1024x!pphlo.pub<f32>>
    %91 = "pphlo.multiply"(%76, %90) {xla_shape = "f32[23,1024]{0,1}"} : (tensor<23x1024x!pphlo.pub<f32>>, tensor<23x1024x!pphlo.pub<f32>>) -> tensor<23x1024x!pphlo.pub<f32>>
    %92 = "pphlo.reduce"(%91, %1) ({
    ^bb0(%arg13: tensor<!pphlo.pub<f32>>, %arg14: tensor<!pphlo.pub<f32>>):
      %101 = "pphlo.add"(%arg13, %arg14) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
      "pphlo.return"(%101) : (tensor<!pphlo.pub<f32>>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<23x1024x!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<23x!pphlo.pub<f32>>
    %93 = "pphlo.multiply"(%75, %92) : (tensor<23x!pphlo.pub<f32>>, tensor<23x!pphlo.pub<f32>>) -> tensor<23x!pphlo.pub<f32>>
    %94 = "pphlo.reshape"(%93) : (tensor<23x!pphlo.pub<f32>>) -> tensor<23x1x!pphlo.pub<f32>>
    %95 = "pphlo.subtract"(%arg7, %94) : (tensor<23x1x!pphlo.pub<f32>>, tensor<23x1x!pphlo.pub<f32>>) -> tensor<23x1x!pphlo.pub<f32>>
    %96 = "pphlo.reshape"(%74) : (tensor<!pphlo.pub<f32>>) -> tensor<1x!pphlo.pub<f32>>
    %97 = "pphlo.reduce"(%88, %1) ({
    ^bb0(%arg13: tensor<!pphlo.pub<f32>>, %arg14: tensor<!pphlo.pub<f32>>):
      %101 = "pphlo.add"(%arg13, %arg14) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
      "pphlo.return"(%101) : (tensor<!pphlo.pub<f32>>) -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<1024x1x!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<1x!pphlo.pub<f32>>
    %98 = "pphlo.multiply"(%96, %97) : (tensor<1x!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>) -> tensor<1x!pphlo.pub<f32>>
    %99 = "pphlo.subtract"(%arg8, %98) : (tensor<1x!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>) -> tensor<1x!pphlo.pub<f32>>
    %100 = "pphlo.add"(%arg11, %0) : (tensor<!pphlo.pub<i64>>, tensor<!pphlo.pub<i64>>) -> tensor<!pphlo.pub<i64>>
    return %57, %95, %99, %55, %16, %100 : tensor<!pphlo.pub<f32>>, tensor<23x1x!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<i64>>
  }
}
