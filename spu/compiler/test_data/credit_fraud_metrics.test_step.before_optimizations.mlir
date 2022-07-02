module @a_inference_test_step_172960__XlaMustCompile_true_config_proto___n_007_n_003CPU_020_001_n_007_n_003GPU_020_0002_002J_0008_001_202_001_000__executor_type____.407 {
  func @main(%arg0: tensor<100x3x!pphlo.pub<f32>>, %arg1: tensor<100x26x!pphlo.pub<f32>>, %arg2: tensor<100x!pphlo.pub<i32>>, %arg3: tensor<29x16x!pphlo.pub<f32>>, %arg4: tensor<16x!pphlo.pub<f32>>, %arg5: tensor<16x24x!pphlo.pub<f32>>, %arg6: tensor<24x!pphlo.pub<f32>>, %arg7: tensor<24x20x!pphlo.pub<f32>>, %arg8: tensor<20x!pphlo.pub<f32>>, %arg9: tensor<20x24x!pphlo.pub<f32>>, %arg10: tensor<24x!pphlo.pub<f32>>, %arg11: tensor<24x1x!pphlo.pub<f32>>, %arg12: tensor<1x!pphlo.pub<f32>>, %arg13: tensor<!pphlo.pub<f32>>, %arg14: tensor<!pphlo.pub<f32>>, %arg15: tensor<1x!pphlo.pub<f32>>, %arg16: tensor<1x!pphlo.pub<f32>>, %arg17: tensor<1x!pphlo.pub<f32>>, %arg18: tensor<1x!pphlo.pub<f32>>) -> (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>) {
    %0 = "pphlo.constant"() {value = dense<0.000000e+00> : tensor<1xf32>} : () -> tensor<1x!pphlo.pub<f32>>
    %1 = "pphlo.constant"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<!pphlo.pub<f32>>
    %2 = "pphlo.constant"() {value = dense<5.000000e-01> : tensor<100x1xf32>} : () -> tensor<100x1x!pphlo.pub<f32>>
    %3 = "pphlo.constant"() {value = dense<0.000000e+00> : tensor<100xf32>} : () -> tensor<100x!pphlo.pub<f32>>
    %4 = "pphlo.constant"() {value = dense<0.000000e+00> : tensor<100x1xf32>} : () -> tensor<100x1x!pphlo.pub<f32>>
    %5 = "pphlo.constant"() {value = dense<0.000000e+00> : tensor<100x24xf32>} : () -> tensor<100x24x!pphlo.pub<f32>>
    %6 = "pphlo.constant"() {value = dense<0.000000e+00> : tensor<100x20xf32>} : () -> tensor<100x20x!pphlo.pub<f32>>
    %7 = "pphlo.constant"() {value = dense<0.000000e+00> : tensor<100x16xf32>} : () -> tensor<100x16x!pphlo.pub<f32>>
    %8 = "pphlo.constant"() {value = dense<1.000000e+02> : tensor<f32>} : () -> tensor<!pphlo.pub<f32>>
    %9 = "pphlo.add"(%arg14, %8) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    %10 = "pphlo.equal"(%9, %1) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<i1>>
    %11 = "pphlo.concatenate"(%arg1, %arg0) {dimension = 1 : i64} : (tensor<100x26x!pphlo.pub<f32>>, tensor<100x3x!pphlo.pub<f32>>) -> tensor<100x29x!pphlo.pub<f32>>
    %12 = "pphlo.dot"(%11, %arg3) : (tensor<100x29x!pphlo.pub<f32>>, tensor<29x16x!pphlo.pub<f32>>) -> tensor<100x16x!pphlo.pub<f32>>
    %13 = "pphlo.broadcast"(%arg4) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<16x!pphlo.pub<f32>>) -> tensor<100x16x!pphlo.pub<f32>>
    %14 = "pphlo.add"(%12, %13) : (tensor<100x16x!pphlo.pub<f32>>, tensor<100x16x!pphlo.pub<f32>>) -> tensor<100x16x!pphlo.pub<f32>>
    %15 = "pphlo.maximum"(%14, %7) : (tensor<100x16x!pphlo.pub<f32>>, tensor<100x16x!pphlo.pub<f32>>) -> tensor<100x16x!pphlo.pub<f32>>
    %16 = "pphlo.dot"(%15, %arg5) : (tensor<100x16x!pphlo.pub<f32>>, tensor<16x24x!pphlo.pub<f32>>) -> tensor<100x24x!pphlo.pub<f32>>
    %17 = "pphlo.broadcast"(%arg6) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<24x!pphlo.pub<f32>>) -> tensor<100x24x!pphlo.pub<f32>>
    %18 = "pphlo.add"(%16, %17) : (tensor<100x24x!pphlo.pub<f32>>, tensor<100x24x!pphlo.pub<f32>>) -> tensor<100x24x!pphlo.pub<f32>>
    %19 = "pphlo.maximum"(%18, %5) : (tensor<100x24x!pphlo.pub<f32>>, tensor<100x24x!pphlo.pub<f32>>) -> tensor<100x24x!pphlo.pub<f32>>
    %20 = "pphlo.dot"(%19, %arg7) : (tensor<100x24x!pphlo.pub<f32>>, tensor<24x20x!pphlo.pub<f32>>) -> tensor<100x20x!pphlo.pub<f32>>
    %21 = "pphlo.broadcast"(%arg8) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<20x!pphlo.pub<f32>>) -> tensor<100x20x!pphlo.pub<f32>>
    %22 = "pphlo.add"(%20, %21) : (tensor<100x20x!pphlo.pub<f32>>, tensor<100x20x!pphlo.pub<f32>>) -> tensor<100x20x!pphlo.pub<f32>>
    %23 = "pphlo.maximum"(%22, %6) : (tensor<100x20x!pphlo.pub<f32>>, tensor<100x20x!pphlo.pub<f32>>) -> tensor<100x20x!pphlo.pub<f32>>
    %24 = "pphlo.dot"(%23, %arg9) : (tensor<100x20x!pphlo.pub<f32>>, tensor<20x24x!pphlo.pub<f32>>) -> tensor<100x24x!pphlo.pub<f32>>
    %25 = "pphlo.broadcast"(%arg10) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<24x!pphlo.pub<f32>>) -> tensor<100x24x!pphlo.pub<f32>>
    %26 = "pphlo.add"(%24, %25) : (tensor<100x24x!pphlo.pub<f32>>, tensor<100x24x!pphlo.pub<f32>>) -> tensor<100x24x!pphlo.pub<f32>>
    %27 = "pphlo.maximum"(%26, %5) : (tensor<100x24x!pphlo.pub<f32>>, tensor<100x24x!pphlo.pub<f32>>) -> tensor<100x24x!pphlo.pub<f32>>
    %28 = "pphlo.reshape"(%arg11) : (tensor<24x1x!pphlo.pub<f32>>) -> tensor<24x!pphlo.pub<f32>>
    %29 = "pphlo.broadcast"(%28) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<24x!pphlo.pub<f32>>) -> tensor<100x24x!pphlo.pub<f32>>
    %30 = "pphlo.multiply"(%27, %29) : (tensor<100x24x!pphlo.pub<f32>>, tensor<100x24x!pphlo.pub<f32>>) -> tensor<100x24x!pphlo.pub<f32>>
    %31 = "pphlo.reduce"(%30, %1) ({
    ^bb0(%arg19: tensor<!pphlo.pub<f32>>, %arg20: tensor<!pphlo.pub<f32>>):
      %84 = "pphlo.add"(%arg19, %arg20) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
      "pphlo.return"(%84) : (tensor<!pphlo.pub<f32>>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<100x24x!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<100x!pphlo.pub<f32>>
    %32 = "pphlo.reshape"(%arg12) : (tensor<1x!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    %33 = "pphlo.broadcast"(%32) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pub<f32>>) -> tensor<100x!pphlo.pub<f32>>
    %34 = "pphlo.add"(%31, %33) : (tensor<100x!pphlo.pub<f32>>, tensor<100x!pphlo.pub<f32>>) -> tensor<100x!pphlo.pub<f32>>
    %35 = "pphlo.reshape"(%34) : (tensor<100x!pphlo.pub<f32>>) -> tensor<100x1x!pphlo.pub<f32>>
    %36 = "pphlo.less"(%35, %4) : (tensor<100x1x!pphlo.pub<f32>>, tensor<100x1x!pphlo.pub<f32>>) -> tensor<100x1x!pphlo.pub<i1>>
    %37 = "pphlo.not"(%36) : (tensor<100x1x!pphlo.pub<i1>>) -> tensor<100x1x!pphlo.pub<i1>>
    %38 = "pphlo.select"(%37, %35, %4) : (tensor<100x1x!pphlo.pub<i1>>, tensor<100x1x!pphlo.pub<f32>>, tensor<100x1x!pphlo.pub<f32>>) -> tensor<100x1x!pphlo.pub<f32>>
    %39 = "pphlo.convert"(%arg2) : (tensor<100x!pphlo.pub<i32>>) -> tensor<100x!pphlo.pub<f32>>
    %40 = "pphlo.reshape"(%39) : (tensor<100x!pphlo.pub<f32>>) -> tensor<100x1x!pphlo.pub<f32>>
    %41 = "pphlo.multiply"(%35, %40) : (tensor<100x1x!pphlo.pub<f32>>, tensor<100x1x!pphlo.pub<f32>>) -> tensor<100x1x!pphlo.pub<f32>>
    %42 = "pphlo.subtract"(%38, %41) : (tensor<100x1x!pphlo.pub<f32>>, tensor<100x1x!pphlo.pub<f32>>) -> tensor<100x1x!pphlo.pub<f32>>
    %43 = "pphlo.negate"(%35) : (tensor<100x1x!pphlo.pub<f32>>) -> tensor<100x1x!pphlo.pub<f32>>
    %44 = "pphlo.select"(%37, %43, %35) : (tensor<100x1x!pphlo.pub<i1>>, tensor<100x1x!pphlo.pub<f32>>, tensor<100x1x!pphlo.pub<f32>>) -> tensor<100x1x!pphlo.pub<f32>>
    %45 = "pphlo.exponential"(%44) : (tensor<100x1x!pphlo.pub<f32>>) -> tensor<100x1x!pphlo.pub<f32>>
    %46 = "pphlo.log_plus_one"(%45) : (tensor<100x1x!pphlo.pub<f32>>) -> tensor<100x1x!pphlo.pub<f32>>
    %47 = "pphlo.add"(%42, %46) : (tensor<100x1x!pphlo.pub<f32>>, tensor<100x1x!pphlo.pub<f32>>) -> tensor<100x1x!pphlo.pub<f32>>
    %48 = "pphlo.reduce"(%47, %1) ({
    ^bb0(%arg19: tensor<!pphlo.pub<f32>>, %arg20: tensor<!pphlo.pub<f32>>):
      %84 = "pphlo.add"(%arg19, %arg20) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
      "pphlo.return"(%84) : (tensor<!pphlo.pub<f32>>) -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<100x1x!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    %49 = "pphlo.add"(%arg13, %48) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    %50 = "pphlo.divide"(%49, %9) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    %51 = "pphlo.select"(%10, %1, %50) : (tensor<!pphlo.pub<i1>>, tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    %52 = "pphlo.equal"(%39, %3) : (tensor<100x!pphlo.pub<f32>>, tensor<100x!pphlo.pub<f32>>) -> tensor<100x!pphlo.pub<i1>>
    %53 = "pphlo.not"(%52) : (tensor<100x!pphlo.pub<i1>>) -> tensor<100x!pphlo.pub<i1>>
    %54 = "pphlo.reshape"(%53) : (tensor<100x!pphlo.pub<i1>>) -> tensor<1x100x!pphlo.pub<i1>>
    %55 = "pphlo.logistic"(%35) : (tensor<100x1x!pphlo.pub<f32>>) -> tensor<100x1x!pphlo.pub<f32>>
    %56 = "pphlo.greater"(%55, %2) : (tensor<100x1x!pphlo.pub<f32>>, tensor<100x1x!pphlo.pub<f32>>) -> tensor<100x1x!pphlo.pub<i1>>
    %57 = "pphlo.reshape"(%56) : (tensor<100x1x!pphlo.pub<i1>>) -> tensor<1x100x!pphlo.pub<i1>>
    %58 = "pphlo.and"(%54, %57) : (tensor<1x100x!pphlo.pub<i1>>, tensor<1x100x!pphlo.pub<i1>>) -> tensor<1x100x!pphlo.pub<i1>>
    %59 = "pphlo.convert"(%58) : (tensor<1x100x!pphlo.pub<i1>>) -> tensor<1x100x!pphlo.pub<f32>>
    %60 = "pphlo.reduce"(%59, %1) ({
    ^bb0(%arg19: tensor<!pphlo.pub<f32>>, %arg20: tensor<!pphlo.pub<f32>>):
      %84 = "pphlo.add"(%arg19, %arg20) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
      "pphlo.return"(%84) : (tensor<!pphlo.pub<f32>>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<1x100x!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<1x!pphlo.pub<f32>>
    %61 = "pphlo.add"(%arg15, %60) : (tensor<1x!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>) -> tensor<1x!pphlo.pub<f32>>
    %62 = "pphlo.not"(%54) : (tensor<1x100x!pphlo.pub<i1>>) -> tensor<1x100x!pphlo.pub<i1>>
    %63 = "pphlo.and"(%62, %57) : (tensor<1x100x!pphlo.pub<i1>>, tensor<1x100x!pphlo.pub<i1>>) -> tensor<1x100x!pphlo.pub<i1>>
    %64 = "pphlo.convert"(%63) : (tensor<1x100x!pphlo.pub<i1>>) -> tensor<1x100x!pphlo.pub<f32>>
    %65 = "pphlo.reduce"(%64, %1) ({
    ^bb0(%arg19: tensor<!pphlo.pub<f32>>, %arg20: tensor<!pphlo.pub<f32>>):
      %84 = "pphlo.add"(%arg19, %arg20) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
      "pphlo.return"(%84) : (tensor<!pphlo.pub<f32>>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<1x100x!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<1x!pphlo.pub<f32>>
    %66 = "pphlo.add"(%arg16, %65) : (tensor<1x!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>) -> tensor<1x!pphlo.pub<f32>>
    %67 = "pphlo.add"(%61, %66) : (tensor<1x!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>) -> tensor<1x!pphlo.pub<f32>>
    %68 = "pphlo.equal"(%67, %0) : (tensor<1x!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>) -> tensor<1x!pphlo.pub<i1>>
    %69 = "pphlo.divide"(%61, %67) : (tensor<1x!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>) -> tensor<1x!pphlo.pub<f32>>
    %70 = "pphlo.select"(%68, %0, %69) : (tensor<1x!pphlo.pub<i1>>, tensor<1x!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>) -> tensor<1x!pphlo.pub<f32>>
    %71 = "pphlo.reshape"(%70) : (tensor<1x!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    %72 = "pphlo.reduce"(%59, %1) ({
    ^bb0(%arg19: tensor<!pphlo.pub<f32>>, %arg20: tensor<!pphlo.pub<f32>>):
      %84 = "pphlo.add"(%arg19, %arg20) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
      "pphlo.return"(%84) : (tensor<!pphlo.pub<f32>>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<1x100x!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<1x!pphlo.pub<f32>>
    %73 = "pphlo.add"(%arg17, %72) : (tensor<1x!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>) -> tensor<1x!pphlo.pub<f32>>
    %74 = "pphlo.not"(%57) : (tensor<1x100x!pphlo.pub<i1>>) -> tensor<1x100x!pphlo.pub<i1>>
    %75 = "pphlo.and"(%54, %74) : (tensor<1x100x!pphlo.pub<i1>>, tensor<1x100x!pphlo.pub<i1>>) -> tensor<1x100x!pphlo.pub<i1>>
    %76 = "pphlo.convert"(%75) : (tensor<1x100x!pphlo.pub<i1>>) -> tensor<1x100x!pphlo.pub<f32>>
    %77 = "pphlo.reduce"(%76, %1) ({
    ^bb0(%arg19: tensor<!pphlo.pub<f32>>, %arg20: tensor<!pphlo.pub<f32>>):
      %84 = "pphlo.add"(%arg19, %arg20) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
      "pphlo.return"(%84) : (tensor<!pphlo.pub<f32>>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<1x100x!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<1x!pphlo.pub<f32>>
    %78 = "pphlo.add"(%arg18, %77) : (tensor<1x!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>) -> tensor<1x!pphlo.pub<f32>>
    %79 = "pphlo.add"(%73, %78) : (tensor<1x!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>) -> tensor<1x!pphlo.pub<f32>>
    %80 = "pphlo.equal"(%79, %0) : (tensor<1x!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>) -> tensor<1x!pphlo.pub<i1>>
    %81 = "pphlo.divide"(%73, %79) : (tensor<1x!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>) -> tensor<1x!pphlo.pub<f32>>
    %82 = "pphlo.select"(%80, %0, %81) : (tensor<1x!pphlo.pub<i1>>, tensor<1x!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>) -> tensor<1x!pphlo.pub<f32>>
    %83 = "pphlo.reshape"(%82) : (tensor<1x!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    return %51, %71, %83, %49, %9, %61, %66, %73, %78 : tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>
  }
}
