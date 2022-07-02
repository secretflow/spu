module @a_inference_train_step_2047__XlaMustCompile_true_config_proto___n_007_n_003CPU_020_001_n_007_n_003GPU_020_0002_002J_0008_001_202_001_000__executor_type____.970 {
  func @main(%arg0: tensor<15x3x!pphlo.pub<f32>>, %arg1: tensor<15x26x!pphlo.pub<f32>>, %arg2: tensor<15x!pphlo.pub<i32>>, %arg3: tensor<15x!pphlo.pub<i32>>, %arg4: tensor<29x16x!pphlo.pub<f32>>, %arg5: tensor<16x!pphlo.pub<f32>>, %arg6: tensor<16x24x!pphlo.pub<f32>>, %arg7: tensor<24x!pphlo.pub<f32>>, %arg8: tensor<24x20x!pphlo.pub<f32>>, %arg9: tensor<20x!pphlo.pub<f32>>, %arg10: tensor<20x24x!pphlo.pub<f32>>, %arg11: tensor<24x!pphlo.pub<f32>>, %arg12: tensor<24x1x!pphlo.pub<f32>>, %arg13: tensor<1x!pphlo.pub<f32>>, %arg14: tensor<!pphlo.pub<f32>>, %arg15: tensor<!pphlo.pub<f32>>, %arg16: tensor<!pphlo.pub<f32>>, %arg17: tensor<!pphlo.pub<i64>>, %arg18: tensor<!pphlo.pub<f32>>, %arg19: tensor<!pphlo.pub<f32>>, %arg20: tensor<29x16x!pphlo.pub<f32>>, %arg21: tensor<29x16x!pphlo.pub<f32>>, %arg22: tensor<16x!pphlo.pub<f32>>, %arg23: tensor<16x!pphlo.pub<f32>>, %arg24: tensor<16x24x!pphlo.pub<f32>>, %arg25: tensor<16x24x!pphlo.pub<f32>>, %arg26: tensor<24x!pphlo.pub<f32>>, %arg27: tensor<24x!pphlo.pub<f32>>, %arg28: tensor<24x20x!pphlo.pub<f32>>, %arg29: tensor<24x20x!pphlo.pub<f32>>, %arg30: tensor<20x!pphlo.pub<f32>>, %arg31: tensor<20x!pphlo.pub<f32>>, %arg32: tensor<20x24x!pphlo.pub<f32>>, %arg33: tensor<20x24x!pphlo.pub<f32>>, %arg34: tensor<24x!pphlo.pub<f32>>, %arg35: tensor<24x!pphlo.pub<f32>>, %arg36: tensor<24x1x!pphlo.pub<f32>>, %arg37: tensor<24x1x!pphlo.pub<f32>>, %arg38: tensor<1x!pphlo.pub<f32>>, %arg39: tensor<1x!pphlo.pub<f32>>, %arg40: tensor<1x!pphlo.pub<f32>>, %arg41: tensor<1x!pphlo.pub<f32>>, %arg42: tensor<1x!pphlo.pub<f32>>, %arg43: tensor<1x!pphlo.pub<f32>>) -> (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>, tensor<29x16x!pphlo.pub<f32>>, tensor<16x!pphlo.pub<f32>>, tensor<16x24x!pphlo.pub<f32>>, tensor<24x!pphlo.pub<f32>>, tensor<24x20x!pphlo.pub<f32>>, tensor<20x!pphlo.pub<f32>>, tensor<20x24x!pphlo.pub<f32>>, tensor<24x!pphlo.pub<f32>>, tensor<24x1x!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<i64>>, tensor<29x16x!pphlo.pub<f32>>, tensor<29x16x!pphlo.pub<f32>>, tensor<16x!pphlo.pub<f32>>, tensor<16x!pphlo.pub<f32>>, tensor<16x24x!pphlo.pub<f32>>, tensor<16x24x!pphlo.pub<f32>>, tensor<24x!pphlo.pub<f32>>, tensor<24x!pphlo.pub<f32>>, tensor<24x20x!pphlo.pub<f32>>, tensor<24x20x!pphlo.pub<f32>>, tensor<20x!pphlo.pub<f32>>, tensor<20x!pphlo.pub<f32>>, tensor<20x24x!pphlo.pub<f32>>, tensor<20x24x!pphlo.pub<f32>>, tensor<24x!pphlo.pub<f32>>, tensor<24x!pphlo.pub<f32>>, tensor<24x1x!pphlo.pub<f32>>, tensor<24x1x!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>) {
    %0 = "pphlo.constant"() {value = dense<1> : tensor<i64>} : () -> tensor<!pphlo.pub<i64>>
    %1 = "pphlo.constant"() {value = dense<1.000000e-07> : tensor<1xf32>} : () -> tensor<1x!pphlo.pub<f32>>
    %2 = "pphlo.constant"() {value = dense<1.000000e+00> : tensor<f32>} : () -> tensor<!pphlo.pub<f32>>
    %3 = "pphlo.constant"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<!pphlo.pub<f32>>
    %4 = "pphlo.constant"() {value = dense<1.000000e-07> : tensor<24x1xf32>} : () -> tensor<24x1x!pphlo.pub<f32>>
    %5 = "pphlo.constant"() {value = dense<1.000000e-07> : tensor<24xf32>} : () -> tensor<24x!pphlo.pub<f32>>
    %6 = "pphlo.constant"() {value = dense<1.000000e-07> : tensor<20x24xf32>} : () -> tensor<20x24x!pphlo.pub<f32>>
    %7 = "pphlo.constant"() {value = dense<1.000000e-07> : tensor<20xf32>} : () -> tensor<20x!pphlo.pub<f32>>
    %8 = "pphlo.constant"() {value = dense<1.000000e-07> : tensor<24x20xf32>} : () -> tensor<24x20x!pphlo.pub<f32>>
    %9 = "pphlo.constant"() {value = dense<1.000000e-07> : tensor<16x24xf32>} : () -> tensor<16x24x!pphlo.pub<f32>>
    %10 = "pphlo.constant"() {value = dense<1.000000e-07> : tensor<16xf32>} : () -> tensor<16x!pphlo.pub<f32>>
    %11 = "pphlo.constant"() {value = dense<1.000000e-07> : tensor<29x16xf32>} : () -> tensor<29x16x!pphlo.pub<f32>>
    %12 = "pphlo.constant"() {value = dense<2.000000e+00> : tensor<15x24xf32>} : () -> tensor<15x24x!pphlo.pub<f32>>
    %13 = "pphlo.constant"() {value = dense<0.000000e+00> : tensor<15x24xf32>} : () -> tensor<15x24x!pphlo.pub<f32>>
    %14 = "pphlo.constant"() {value = dense<1.000000e+00> : tensor<15x1xf32>} : () -> tensor<15x1x!pphlo.pub<f32>>
    %15 = "pphlo.constant"() {value = dense<0.000000e+00> : tensor<15x1xf32>} : () -> tensor<15x1x!pphlo.pub<f32>>
    %16 = "pphlo.constant"() {value = dense<0.0666666701> : tensor<15xf32>} : () -> tensor<15x!pphlo.pub<f32>>
    %17 = "pphlo.constant"() {value = dense<0.000000e+00> : tensor<15x20xf32>} : () -> tensor<15x20x!pphlo.pub<f32>>
    %18 = "pphlo.constant"() {value = dense<0.000000e+00> : tensor<15x16xf32>} : () -> tensor<15x16x!pphlo.pub<f32>>
    %19 = "pphlo.constant"() {value = dense<0.000000e+00> : tensor<1xf32>} : () -> tensor<1x!pphlo.pub<f32>>
    %20 = "pphlo.constant"() {value = dense<5.000000e-01> : tensor<15x1xf32>} : () -> tensor<15x1x!pphlo.pub<f32>>
    %21 = "pphlo.constant"() {value = dense<0.000000e+00> : tensor<15xf32>} : () -> tensor<15x!pphlo.pub<f32>>
    %22 = "pphlo.constant"() {value = dense<5.000000e-01> : tensor<15x24xf32>} : () -> tensor<15x24x!pphlo.pub<f32>>
    %23 = "pphlo.constant"() {value = dense<1.500000e+01> : tensor<f32>} : () -> tensor<!pphlo.pub<f32>>
    %24 = "pphlo.add"(%arg15, %23) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    %25 = "pphlo.equal"(%24, %3) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<i1>>
    %26 = "pphlo.rng_uniform"(%3, %2) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<15x24x!pphlo.pub<f32>>
    %27 = "pphlo.less"(%26, %22) : (tensor<15x24x!pphlo.pub<f32>>, tensor<15x24x!pphlo.pub<f32>>) -> tensor<15x24x!pphlo.pub<i1>>
    %28 = "pphlo.not"(%27) : (tensor<15x24x!pphlo.pub<i1>>) -> tensor<15x24x!pphlo.pub<i1>>
    %29 = "pphlo.concatenate"(%arg1, %arg0) {dimension = 1 : i64} : (tensor<15x26x!pphlo.pub<f32>>, tensor<15x3x!pphlo.pub<f32>>) -> tensor<15x29x!pphlo.pub<f32>>
    %30 = "pphlo.dot"(%29, %arg4) : (tensor<15x29x!pphlo.pub<f32>>, tensor<29x16x!pphlo.pub<f32>>) -> tensor<15x16x!pphlo.pub<f32>>
    %31 = "pphlo.broadcast"(%arg5) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<16x!pphlo.pub<f32>>) -> tensor<15x16x!pphlo.pub<f32>>
    %32 = "pphlo.add"(%30, %31) : (tensor<15x16x!pphlo.pub<f32>>, tensor<15x16x!pphlo.pub<f32>>) -> tensor<15x16x!pphlo.pub<f32>>
    %33 = "pphlo.maximum"(%32, %18) : (tensor<15x16x!pphlo.pub<f32>>, tensor<15x16x!pphlo.pub<f32>>) -> tensor<15x16x!pphlo.pub<f32>>
    %34 = "pphlo.dot"(%33, %arg6) : (tensor<15x16x!pphlo.pub<f32>>, tensor<16x24x!pphlo.pub<f32>>) -> tensor<15x24x!pphlo.pub<f32>>
    %35 = "pphlo.broadcast"(%arg7) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<24x!pphlo.pub<f32>>) -> tensor<15x24x!pphlo.pub<f32>>
    %36 = "pphlo.add"(%34, %35) : (tensor<15x24x!pphlo.pub<f32>>, tensor<15x24x!pphlo.pub<f32>>) -> tensor<15x24x!pphlo.pub<f32>>
    %37 = "pphlo.maximum"(%36, %13) : (tensor<15x24x!pphlo.pub<f32>>, tensor<15x24x!pphlo.pub<f32>>) -> tensor<15x24x!pphlo.pub<f32>>
    %38 = "pphlo.multiply"(%37, %12) : (tensor<15x24x!pphlo.pub<f32>>, tensor<15x24x!pphlo.pub<f32>>) -> tensor<15x24x!pphlo.pub<f32>>
    %39 = "pphlo.select"(%28, %38, %13) : (tensor<15x24x!pphlo.pub<i1>>, tensor<15x24x!pphlo.pub<f32>>, tensor<15x24x!pphlo.pub<f32>>) -> tensor<15x24x!pphlo.pub<f32>>
    %40 = "pphlo.dot"(%39, %arg8) : (tensor<15x24x!pphlo.pub<f32>>, tensor<24x20x!pphlo.pub<f32>>) -> tensor<15x20x!pphlo.pub<f32>>
    %41 = "pphlo.broadcast"(%arg9) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<20x!pphlo.pub<f32>>) -> tensor<15x20x!pphlo.pub<f32>>
    %42 = "pphlo.add"(%40, %41) : (tensor<15x20x!pphlo.pub<f32>>, tensor<15x20x!pphlo.pub<f32>>) -> tensor<15x20x!pphlo.pub<f32>>
    %43 = "pphlo.maximum"(%42, %17) : (tensor<15x20x!pphlo.pub<f32>>, tensor<15x20x!pphlo.pub<f32>>) -> tensor<15x20x!pphlo.pub<f32>>
    %44 = "pphlo.dot"(%43, %arg10) : (tensor<15x20x!pphlo.pub<f32>>, tensor<20x24x!pphlo.pub<f32>>) -> tensor<15x24x!pphlo.pub<f32>>
    %45 = "pphlo.broadcast"(%arg11) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<24x!pphlo.pub<f32>>) -> tensor<15x24x!pphlo.pub<f32>>
    %46 = "pphlo.add"(%44, %45) : (tensor<15x24x!pphlo.pub<f32>>, tensor<15x24x!pphlo.pub<f32>>) -> tensor<15x24x!pphlo.pub<f32>>
    %47 = "pphlo.maximum"(%46, %13) : (tensor<15x24x!pphlo.pub<f32>>, tensor<15x24x!pphlo.pub<f32>>) -> tensor<15x24x!pphlo.pub<f32>>
    %48 = "pphlo.reshape"(%arg12) : (tensor<24x1x!pphlo.pub<f32>>) -> tensor<24x!pphlo.pub<f32>>
    %49 = "pphlo.broadcast"(%48) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<24x!pphlo.pub<f32>>) -> tensor<15x24x!pphlo.pub<f32>>
    %50 = "pphlo.multiply"(%47, %49) : (tensor<15x24x!pphlo.pub<f32>>, tensor<15x24x!pphlo.pub<f32>>) -> tensor<15x24x!pphlo.pub<f32>>
    %51 = "pphlo.reduce"(%50, %3) ({
    ^bb0(%arg44: tensor<!pphlo.pub<f32>>, %arg45: tensor<!pphlo.pub<f32>>):
      %327 = "pphlo.add"(%arg44, %arg45) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
      "pphlo.return"(%327) : (tensor<!pphlo.pub<f32>>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<15x24x!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<15x!pphlo.pub<f32>>
    %52 = "pphlo.reshape"(%arg13) : (tensor<1x!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    %53 = "pphlo.broadcast"(%52) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pub<f32>>) -> tensor<15x!pphlo.pub<f32>>
    %54 = "pphlo.add"(%51, %53) : (tensor<15x!pphlo.pub<f32>>, tensor<15x!pphlo.pub<f32>>) -> tensor<15x!pphlo.pub<f32>>
    %55 = "pphlo.reshape"(%54) : (tensor<15x!pphlo.pub<f32>>) -> tensor<15x1x!pphlo.pub<f32>>
    %56 = "pphlo.less"(%55, %15) : (tensor<15x1x!pphlo.pub<f32>>, tensor<15x1x!pphlo.pub<f32>>) -> tensor<15x1x!pphlo.pub<i1>>
    %57 = "pphlo.not"(%56) : (tensor<15x1x!pphlo.pub<i1>>) -> tensor<15x1x!pphlo.pub<i1>>
    %58 = "pphlo.select"(%57, %55, %15) : (tensor<15x1x!pphlo.pub<i1>>, tensor<15x1x!pphlo.pub<f32>>, tensor<15x1x!pphlo.pub<f32>>) -> tensor<15x1x!pphlo.pub<f32>>
    %59 = "pphlo.convert"(%arg2) : (tensor<15x!pphlo.pub<i32>>) -> tensor<15x!pphlo.pub<f32>>
    %60 = "pphlo.reshape"(%59) : (tensor<15x!pphlo.pub<f32>>) -> tensor<15x1x!pphlo.pub<f32>>
    %61 = "pphlo.multiply"(%55, %60) : (tensor<15x1x!pphlo.pub<f32>>, tensor<15x1x!pphlo.pub<f32>>) -> tensor<15x1x!pphlo.pub<f32>>
    %62 = "pphlo.subtract"(%58, %61) : (tensor<15x1x!pphlo.pub<f32>>, tensor<15x1x!pphlo.pub<f32>>) -> tensor<15x1x!pphlo.pub<f32>>
    %63 = "pphlo.negate"(%55) : (tensor<15x1x!pphlo.pub<f32>>) -> tensor<15x1x!pphlo.pub<f32>>
    %64 = "pphlo.select"(%57, %63, %55) : (tensor<15x1x!pphlo.pub<i1>>, tensor<15x1x!pphlo.pub<f32>>, tensor<15x1x!pphlo.pub<f32>>) -> tensor<15x1x!pphlo.pub<f32>>
    %65 = "pphlo.exponential"(%64) : (tensor<15x1x!pphlo.pub<f32>>) -> tensor<15x1x!pphlo.pub<f32>>
    %66 = "pphlo.log_plus_one"(%65) : (tensor<15x1x!pphlo.pub<f32>>) -> tensor<15x1x!pphlo.pub<f32>>
    %67 = "pphlo.add"(%62, %66) : (tensor<15x1x!pphlo.pub<f32>>, tensor<15x1x!pphlo.pub<f32>>) -> tensor<15x1x!pphlo.pub<f32>>
    %68 = "pphlo.reshape"(%67) : (tensor<15x1x!pphlo.pub<f32>>) -> tensor<15x!pphlo.pub<f32>>
    %69 = "pphlo.convert"(%arg3) : (tensor<15x!pphlo.pub<i32>>) -> tensor<15x!pphlo.pub<f32>>
    %70 = "pphlo.multiply"(%68, %69) : (tensor<15x!pphlo.pub<f32>>, tensor<15x!pphlo.pub<f32>>) -> tensor<15x!pphlo.pub<f32>>
    %71 = "pphlo.reduce"(%70, %3) ({
    ^bb0(%arg44: tensor<!pphlo.pub<f32>>, %arg45: tensor<!pphlo.pub<f32>>):
      %327 = "pphlo.add"(%arg44, %arg45) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
      "pphlo.return"(%327) : (tensor<!pphlo.pub<f32>>) -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<15x!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    %72 = "pphlo.add"(%arg14, %71) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    %73 = "pphlo.divide"(%72, %24) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    %74 = "pphlo.select"(%25, %3, %73) : (tensor<!pphlo.pub<i1>>, tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    %75 = "pphlo.equal"(%59, %21) : (tensor<15x!pphlo.pub<f32>>, tensor<15x!pphlo.pub<f32>>) -> tensor<15x!pphlo.pub<i1>>
    %76 = "pphlo.not"(%75) : (tensor<15x!pphlo.pub<i1>>) -> tensor<15x!pphlo.pub<i1>>
    %77 = "pphlo.reshape"(%76) : (tensor<15x!pphlo.pub<i1>>) -> tensor<1x15x!pphlo.pub<i1>>
    %78 = "pphlo.logistic"(%55) : (tensor<15x1x!pphlo.pub<f32>>) -> tensor<15x1x!pphlo.pub<f32>>
    %79 = "pphlo.greater"(%78, %20) : (tensor<15x1x!pphlo.pub<f32>>, tensor<15x1x!pphlo.pub<f32>>) -> tensor<15x1x!pphlo.pub<i1>>
    %80 = "pphlo.reshape"(%79) : (tensor<15x1x!pphlo.pub<i1>>) -> tensor<1x15x!pphlo.pub<i1>>
    %81 = "pphlo.and"(%77, %80) : (tensor<1x15x!pphlo.pub<i1>>, tensor<1x15x!pphlo.pub<i1>>) -> tensor<1x15x!pphlo.pub<i1>>
    %82 = "pphlo.convert"(%81) : (tensor<1x15x!pphlo.pub<i1>>) -> tensor<1x15x!pphlo.pub<f32>>
    %83 = "pphlo.reduce"(%82, %3) ({
    ^bb0(%arg44: tensor<!pphlo.pub<f32>>, %arg45: tensor<!pphlo.pub<f32>>):
      %327 = "pphlo.add"(%arg44, %arg45) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
      "pphlo.return"(%327) : (tensor<!pphlo.pub<f32>>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<1x15x!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<1x!pphlo.pub<f32>>
    %84 = "pphlo.add"(%arg40, %83) : (tensor<1x!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>) -> tensor<1x!pphlo.pub<f32>>
    %85 = "pphlo.not"(%77) : (tensor<1x15x!pphlo.pub<i1>>) -> tensor<1x15x!pphlo.pub<i1>>
    %86 = "pphlo.and"(%85, %80) : (tensor<1x15x!pphlo.pub<i1>>, tensor<1x15x!pphlo.pub<i1>>) -> tensor<1x15x!pphlo.pub<i1>>
    %87 = "pphlo.convert"(%86) : (tensor<1x15x!pphlo.pub<i1>>) -> tensor<1x15x!pphlo.pub<f32>>
    %88 = "pphlo.reduce"(%87, %3) ({
    ^bb0(%arg44: tensor<!pphlo.pub<f32>>, %arg45: tensor<!pphlo.pub<f32>>):
      %327 = "pphlo.add"(%arg44, %arg45) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
      "pphlo.return"(%327) : (tensor<!pphlo.pub<f32>>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<1x15x!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<1x!pphlo.pub<f32>>
    %89 = "pphlo.add"(%arg41, %88) : (tensor<1x!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>) -> tensor<1x!pphlo.pub<f32>>
    %90 = "pphlo.add"(%84, %89) : (tensor<1x!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>) -> tensor<1x!pphlo.pub<f32>>
    %91 = "pphlo.equal"(%90, %19) : (tensor<1x!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>) -> tensor<1x!pphlo.pub<i1>>
    %92 = "pphlo.divide"(%84, %90) : (tensor<1x!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>) -> tensor<1x!pphlo.pub<f32>>
    %93 = "pphlo.select"(%91, %19, %92) : (tensor<1x!pphlo.pub<i1>>, tensor<1x!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>) -> tensor<1x!pphlo.pub<f32>>
    %94 = "pphlo.reshape"(%93) : (tensor<1x!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    %95 = "pphlo.reduce"(%82, %3) ({
    ^bb0(%arg44: tensor<!pphlo.pub<f32>>, %arg45: tensor<!pphlo.pub<f32>>):
      %327 = "pphlo.add"(%arg44, %arg45) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
      "pphlo.return"(%327) : (tensor<!pphlo.pub<f32>>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<1x15x!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<1x!pphlo.pub<f32>>
    %96 = "pphlo.add"(%arg42, %95) : (tensor<1x!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>) -> tensor<1x!pphlo.pub<f32>>
    %97 = "pphlo.not"(%80) : (tensor<1x15x!pphlo.pub<i1>>) -> tensor<1x15x!pphlo.pub<i1>>
    %98 = "pphlo.and"(%77, %97) : (tensor<1x15x!pphlo.pub<i1>>, tensor<1x15x!pphlo.pub<i1>>) -> tensor<1x15x!pphlo.pub<i1>>
    %99 = "pphlo.convert"(%98) : (tensor<1x15x!pphlo.pub<i1>>) -> tensor<1x15x!pphlo.pub<f32>>
    %100 = "pphlo.reduce"(%99, %3) ({
    ^bb0(%arg44: tensor<!pphlo.pub<f32>>, %arg45: tensor<!pphlo.pub<f32>>):
      %327 = "pphlo.add"(%arg44, %arg45) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
      "pphlo.return"(%327) : (tensor<!pphlo.pub<f32>>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<1x15x!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<1x!pphlo.pub<f32>>
    %101 = "pphlo.add"(%arg43, %100) : (tensor<1x!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>) -> tensor<1x!pphlo.pub<f32>>
    %102 = "pphlo.add"(%96, %101) : (tensor<1x!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>) -> tensor<1x!pphlo.pub<f32>>
    %103 = "pphlo.equal"(%102, %19) : (tensor<1x!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>) -> tensor<1x!pphlo.pub<i1>>
    %104 = "pphlo.divide"(%96, %102) : (tensor<1x!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>) -> tensor<1x!pphlo.pub<f32>>
    %105 = "pphlo.select"(%103, %19, %104) : (tensor<1x!pphlo.pub<i1>>, tensor<1x!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>) -> tensor<1x!pphlo.pub<f32>>
    %106 = "pphlo.reshape"(%105) : (tensor<1x!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    %107 = "pphlo.greater"(%33, %18) : (tensor<15x16x!pphlo.pub<f32>>, tensor<15x16x!pphlo.pub<f32>>) -> tensor<15x16x!pphlo.pub<i1>>
    %108 = "pphlo.greater"(%37, %13) : (tensor<15x24x!pphlo.pub<f32>>, tensor<15x24x!pphlo.pub<f32>>) -> tensor<15x24x!pphlo.pub<i1>>
    %109 = "pphlo.greater"(%43, %17) : (tensor<15x20x!pphlo.pub<f32>>, tensor<15x20x!pphlo.pub<f32>>) -> tensor<15x20x!pphlo.pub<i1>>
    %110 = "pphlo.greater"(%47, %13) : (tensor<15x24x!pphlo.pub<f32>>, tensor<15x24x!pphlo.pub<f32>>) -> tensor<15x24x!pphlo.pub<i1>>
    %111 = "pphlo.multiply"(%69, %16) : (tensor<15x!pphlo.pub<f32>>, tensor<15x!pphlo.pub<f32>>) -> tensor<15x!pphlo.pub<f32>>
    %112 = "pphlo.reshape"(%111) : (tensor<15x!pphlo.pub<f32>>) -> tensor<15x1x!pphlo.pub<f32>>
    %113 = "pphlo.select"(%57, %112, %15) : (tensor<15x1x!pphlo.pub<i1>>, tensor<15x1x!pphlo.pub<f32>>, tensor<15x1x!pphlo.pub<f32>>) -> tensor<15x1x!pphlo.pub<f32>>
    %114 = "pphlo.negate"(%112) : (tensor<15x1x!pphlo.pub<f32>>) -> tensor<15x1x!pphlo.pub<f32>>
    %115 = "pphlo.multiply"(%114, %60) : (tensor<15x1x!pphlo.pub<f32>>, tensor<15x1x!pphlo.pub<f32>>) -> tensor<15x1x!pphlo.pub<f32>>
    %116 = "pphlo.add"(%113, %115) : (tensor<15x1x!pphlo.pub<f32>>, tensor<15x1x!pphlo.pub<f32>>) -> tensor<15x1x!pphlo.pub<f32>>
    %117 = "pphlo.add"(%65, %14) : (tensor<15x1x!pphlo.pub<f32>>, tensor<15x1x!pphlo.pub<f32>>) -> tensor<15x1x!pphlo.pub<f32>>
    %118 = "pphlo.divide"(%14, %117) : (tensor<15x1x!pphlo.pub<f32>>, tensor<15x1x!pphlo.pub<f32>>) -> tensor<15x1x!pphlo.pub<f32>>
    %119 = "pphlo.multiply"(%112, %118) : (tensor<15x1x!pphlo.pub<f32>>, tensor<15x1x!pphlo.pub<f32>>) -> tensor<15x1x!pphlo.pub<f32>>
    %120 = "pphlo.multiply"(%119, %65) : (tensor<15x1x!pphlo.pub<f32>>, tensor<15x1x!pphlo.pub<f32>>) -> tensor<15x1x!pphlo.pub<f32>>
    %121 = "pphlo.select"(%57, %15, %120) : (tensor<15x1x!pphlo.pub<i1>>, tensor<15x1x!pphlo.pub<f32>>, tensor<15x1x!pphlo.pub<f32>>) -> tensor<15x1x!pphlo.pub<f32>>
    %122 = "pphlo.add"(%116, %121) : (tensor<15x1x!pphlo.pub<f32>>, tensor<15x1x!pphlo.pub<f32>>) -> tensor<15x1x!pphlo.pub<f32>>
    %123 = "pphlo.select"(%57, %120, %15) : (tensor<15x1x!pphlo.pub<i1>>, tensor<15x1x!pphlo.pub<f32>>, tensor<15x1x!pphlo.pub<f32>>) -> tensor<15x1x!pphlo.pub<f32>>
    %124 = "pphlo.negate"(%123) : (tensor<15x1x!pphlo.pub<f32>>) -> tensor<15x1x!pphlo.pub<f32>>
    %125 = "pphlo.add"(%122, %124) : (tensor<15x1x!pphlo.pub<f32>>, tensor<15x1x!pphlo.pub<f32>>) -> tensor<15x1x!pphlo.pub<f32>>
    %126 = "pphlo.reshape"(%125) : (tensor<15x1x!pphlo.pub<f32>>) -> tensor<15x!pphlo.pub<f32>>
    %127 = "pphlo.broadcast"(%126) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<15x!pphlo.pub<f32>>) -> tensor<15x24x!pphlo.pub<f32>>
    %128 = "pphlo.multiply"(%127, %49) : (tensor<15x24x!pphlo.pub<f32>>, tensor<15x24x!pphlo.pub<f32>>) -> tensor<15x24x!pphlo.pub<f32>>
    %129 = "pphlo.select"(%110, %128, %13) : (tensor<15x24x!pphlo.pub<i1>>, tensor<15x24x!pphlo.pub<f32>>, tensor<15x24x!pphlo.pub<f32>>) -> tensor<15x24x!pphlo.pub<f32>>
    %130 = "pphlo.transpose"(%arg10) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<20x24x!pphlo.pub<f32>>) -> tensor<24x20x!pphlo.pub<f32>>
    %131 = "pphlo.dot"(%129, %130) : (tensor<15x24x!pphlo.pub<f32>>, tensor<24x20x!pphlo.pub<f32>>) -> tensor<15x20x!pphlo.pub<f32>>
    %132 = "pphlo.select"(%109, %131, %17) : (tensor<15x20x!pphlo.pub<i1>>, tensor<15x20x!pphlo.pub<f32>>, tensor<15x20x!pphlo.pub<f32>>) -> tensor<15x20x!pphlo.pub<f32>>
    %133 = "pphlo.transpose"(%arg8) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<24x20x!pphlo.pub<f32>>) -> tensor<20x24x!pphlo.pub<f32>>
    %134 = "pphlo.dot"(%132, %133) : (tensor<15x20x!pphlo.pub<f32>>, tensor<20x24x!pphlo.pub<f32>>) -> tensor<15x24x!pphlo.pub<f32>>
    %135 = "pphlo.select"(%28, %134, %13) : (tensor<15x24x!pphlo.pub<i1>>, tensor<15x24x!pphlo.pub<f32>>, tensor<15x24x!pphlo.pub<f32>>) -> tensor<15x24x!pphlo.pub<f32>>
    %136 = "pphlo.multiply"(%135, %12) : (tensor<15x24x!pphlo.pub<f32>>, tensor<15x24x!pphlo.pub<f32>>) -> tensor<15x24x!pphlo.pub<f32>>
    %137 = "pphlo.select"(%108, %136, %13) : (tensor<15x24x!pphlo.pub<i1>>, tensor<15x24x!pphlo.pub<f32>>, tensor<15x24x!pphlo.pub<f32>>) -> tensor<15x24x!pphlo.pub<f32>>
    %138 = "pphlo.transpose"(%arg6) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<16x24x!pphlo.pub<f32>>) -> tensor<24x16x!pphlo.pub<f32>>
    %139 = "pphlo.dot"(%137, %138) : (tensor<15x24x!pphlo.pub<f32>>, tensor<24x16x!pphlo.pub<f32>>) -> tensor<15x16x!pphlo.pub<f32>>
    %140 = "pphlo.select"(%107, %139, %18) : (tensor<15x16x!pphlo.pub<i1>>, tensor<15x16x!pphlo.pub<f32>>, tensor<15x16x!pphlo.pub<f32>>) -> tensor<15x16x!pphlo.pub<f32>>
    %141 = "pphlo.transpose"(%29) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<15x29x!pphlo.pub<f32>>) -> tensor<29x15x!pphlo.pub<f32>>
    %142 = "pphlo.dot"(%141, %140) : (tensor<29x15x!pphlo.pub<f32>>, tensor<15x16x!pphlo.pub<f32>>) -> tensor<29x16x!pphlo.pub<f32>>
    %143 = "pphlo.subtract"(%142, %arg20) : (tensor<29x16x!pphlo.pub<f32>>, tensor<29x16x!pphlo.pub<f32>>) -> tensor<29x16x!pphlo.pub<f32>>
    %144 = "pphlo.subtract"(%2, %arg18) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    %145 = "pphlo.broadcast"(%144) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pub<f32>>) -> tensor<29x16x!pphlo.pub<f32>>
    %146 = "pphlo.multiply"(%143, %145) : (tensor<29x16x!pphlo.pub<f32>>, tensor<29x16x!pphlo.pub<f32>>) -> tensor<29x16x!pphlo.pub<f32>>
    %147 = "pphlo.add"(%arg20, %146) : (tensor<29x16x!pphlo.pub<f32>>, tensor<29x16x!pphlo.pub<f32>>) -> tensor<29x16x!pphlo.pub<f32>>
    %148 = "pphlo.add"(%arg17, %0) : (tensor<!pphlo.pub<i64>>, tensor<!pphlo.pub<i64>>) -> tensor<!pphlo.pub<i64>>
    %149 = "pphlo.convert"(%148) : (tensor<!pphlo.pub<i64>>) -> tensor<!pphlo.pub<f32>>
    %150 = "pphlo.power"(%arg19, %149) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    %151 = "pphlo.subtract"(%2, %150) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    %152 = "pphlo.constant"() {value = dense<5.000000e-01> : tensor<f32>} : () -> tensor<!pphlo.pub<f32>>
    %153 = "pphlo.power"(%151, %152) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    %154 = "pphlo.multiply"(%arg16, %153) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    %155 = "pphlo.power"(%arg18, %149) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    %156 = "pphlo.subtract"(%2, %155) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    %157 = "pphlo.divide"(%154, %156) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    %158 = "pphlo.broadcast"(%157) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pub<f32>>) -> tensor<29x16x!pphlo.pub<f32>>
    %159 = "pphlo.multiply"(%147, %158) : (tensor<29x16x!pphlo.pub<f32>>, tensor<29x16x!pphlo.pub<f32>>) -> tensor<29x16x!pphlo.pub<f32>>
    %160 = "pphlo.multiply"(%142, %142) : (tensor<29x16x!pphlo.pub<f32>>, tensor<29x16x!pphlo.pub<f32>>) -> tensor<29x16x!pphlo.pub<f32>>
    %161 = "pphlo.subtract"(%160, %arg21) : (tensor<29x16x!pphlo.pub<f32>>, tensor<29x16x!pphlo.pub<f32>>) -> tensor<29x16x!pphlo.pub<f32>>
    %162 = "pphlo.subtract"(%2, %arg19) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    %163 = "pphlo.broadcast"(%162) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pub<f32>>) -> tensor<29x16x!pphlo.pub<f32>>
    %164 = "pphlo.multiply"(%161, %163) : (tensor<29x16x!pphlo.pub<f32>>, tensor<29x16x!pphlo.pub<f32>>) -> tensor<29x16x!pphlo.pub<f32>>
    %165 = "pphlo.add"(%arg21, %164) : (tensor<29x16x!pphlo.pub<f32>>, tensor<29x16x!pphlo.pub<f32>>) -> tensor<29x16x!pphlo.pub<f32>>
    %166 = "pphlo.constant"() {value = dense<5.000000e-01> : tensor<29x16xf32>} : () -> tensor<29x16x!pphlo.pub<f32>>
    %167 = "pphlo.power"(%165, %166) : (tensor<29x16x!pphlo.pub<f32>>, tensor<29x16x!pphlo.pub<f32>>) -> tensor<29x16x!pphlo.pub<f32>>
    %168 = "pphlo.add"(%167, %11) : (tensor<29x16x!pphlo.pub<f32>>, tensor<29x16x!pphlo.pub<f32>>) -> tensor<29x16x!pphlo.pub<f32>>
    %169 = "pphlo.divide"(%159, %168) : (tensor<29x16x!pphlo.pub<f32>>, tensor<29x16x!pphlo.pub<f32>>) -> tensor<29x16x!pphlo.pub<f32>>
    %170 = "pphlo.subtract"(%arg4, %169) : (tensor<29x16x!pphlo.pub<f32>>, tensor<29x16x!pphlo.pub<f32>>) -> tensor<29x16x!pphlo.pub<f32>>
    %171 = "pphlo.reduce"(%140, %3) ({
    ^bb0(%arg44: tensor<!pphlo.pub<f32>>, %arg45: tensor<!pphlo.pub<f32>>):
      %327 = "pphlo.add"(%arg44, %arg45) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
      "pphlo.return"(%327) : (tensor<!pphlo.pub<f32>>) -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<15x16x!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<16x!pphlo.pub<f32>>
    %172 = "pphlo.subtract"(%171, %arg22) : (tensor<16x!pphlo.pub<f32>>, tensor<16x!pphlo.pub<f32>>) -> tensor<16x!pphlo.pub<f32>>
    %173 = "pphlo.broadcast"(%144) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pub<f32>>) -> tensor<16x!pphlo.pub<f32>>
    %174 = "pphlo.multiply"(%172, %173) : (tensor<16x!pphlo.pub<f32>>, tensor<16x!pphlo.pub<f32>>) -> tensor<16x!pphlo.pub<f32>>
    %175 = "pphlo.add"(%arg22, %174) : (tensor<16x!pphlo.pub<f32>>, tensor<16x!pphlo.pub<f32>>) -> tensor<16x!pphlo.pub<f32>>
    %176 = "pphlo.broadcast"(%157) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pub<f32>>) -> tensor<16x!pphlo.pub<f32>>
    %177 = "pphlo.multiply"(%175, %176) : (tensor<16x!pphlo.pub<f32>>, tensor<16x!pphlo.pub<f32>>) -> tensor<16x!pphlo.pub<f32>>
    %178 = "pphlo.multiply"(%171, %171) : (tensor<16x!pphlo.pub<f32>>, tensor<16x!pphlo.pub<f32>>) -> tensor<16x!pphlo.pub<f32>>
    %179 = "pphlo.subtract"(%178, %arg23) : (tensor<16x!pphlo.pub<f32>>, tensor<16x!pphlo.pub<f32>>) -> tensor<16x!pphlo.pub<f32>>
    %180 = "pphlo.broadcast"(%162) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pub<f32>>) -> tensor<16x!pphlo.pub<f32>>
    %181 = "pphlo.multiply"(%179, %180) : (tensor<16x!pphlo.pub<f32>>, tensor<16x!pphlo.pub<f32>>) -> tensor<16x!pphlo.pub<f32>>
    %182 = "pphlo.add"(%arg23, %181) : (tensor<16x!pphlo.pub<f32>>, tensor<16x!pphlo.pub<f32>>) -> tensor<16x!pphlo.pub<f32>>
    %183 = "pphlo.constant"() {value = dense<5.000000e-01> : tensor<16xf32>} : () -> tensor<16x!pphlo.pub<f32>>
    %184 = "pphlo.power"(%182, %183) : (tensor<16x!pphlo.pub<f32>>, tensor<16x!pphlo.pub<f32>>) -> tensor<16x!pphlo.pub<f32>>
    %185 = "pphlo.add"(%184, %10) : (tensor<16x!pphlo.pub<f32>>, tensor<16x!pphlo.pub<f32>>) -> tensor<16x!pphlo.pub<f32>>
    %186 = "pphlo.divide"(%177, %185) : (tensor<16x!pphlo.pub<f32>>, tensor<16x!pphlo.pub<f32>>) -> tensor<16x!pphlo.pub<f32>>
    %187 = "pphlo.subtract"(%arg5, %186) : (tensor<16x!pphlo.pub<f32>>, tensor<16x!pphlo.pub<f32>>) -> tensor<16x!pphlo.pub<f32>>
    %188 = "pphlo.transpose"(%33) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<15x16x!pphlo.pub<f32>>) -> tensor<16x15x!pphlo.pub<f32>>
    %189 = "pphlo.dot"(%188, %137) : (tensor<16x15x!pphlo.pub<f32>>, tensor<15x24x!pphlo.pub<f32>>) -> tensor<16x24x!pphlo.pub<f32>>
    %190 = "pphlo.subtract"(%189, %arg24) : (tensor<16x24x!pphlo.pub<f32>>, tensor<16x24x!pphlo.pub<f32>>) -> tensor<16x24x!pphlo.pub<f32>>
    %191 = "pphlo.broadcast"(%144) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pub<f32>>) -> tensor<16x24x!pphlo.pub<f32>>
    %192 = "pphlo.multiply"(%190, %191) : (tensor<16x24x!pphlo.pub<f32>>, tensor<16x24x!pphlo.pub<f32>>) -> tensor<16x24x!pphlo.pub<f32>>
    %193 = "pphlo.add"(%arg24, %192) : (tensor<16x24x!pphlo.pub<f32>>, tensor<16x24x!pphlo.pub<f32>>) -> tensor<16x24x!pphlo.pub<f32>>
    %194 = "pphlo.broadcast"(%157) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pub<f32>>) -> tensor<16x24x!pphlo.pub<f32>>
    %195 = "pphlo.multiply"(%193, %194) : (tensor<16x24x!pphlo.pub<f32>>, tensor<16x24x!pphlo.pub<f32>>) -> tensor<16x24x!pphlo.pub<f32>>
    %196 = "pphlo.multiply"(%189, %189) : (tensor<16x24x!pphlo.pub<f32>>, tensor<16x24x!pphlo.pub<f32>>) -> tensor<16x24x!pphlo.pub<f32>>
    %197 = "pphlo.subtract"(%196, %arg25) : (tensor<16x24x!pphlo.pub<f32>>, tensor<16x24x!pphlo.pub<f32>>) -> tensor<16x24x!pphlo.pub<f32>>
    %198 = "pphlo.broadcast"(%162) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pub<f32>>) -> tensor<16x24x!pphlo.pub<f32>>
    %199 = "pphlo.multiply"(%197, %198) : (tensor<16x24x!pphlo.pub<f32>>, tensor<16x24x!pphlo.pub<f32>>) -> tensor<16x24x!pphlo.pub<f32>>
    %200 = "pphlo.add"(%arg25, %199) : (tensor<16x24x!pphlo.pub<f32>>, tensor<16x24x!pphlo.pub<f32>>) -> tensor<16x24x!pphlo.pub<f32>>
    %201 = "pphlo.constant"() {value = dense<5.000000e-01> : tensor<16x24xf32>} : () -> tensor<16x24x!pphlo.pub<f32>>
    %202 = "pphlo.power"(%200, %201) : (tensor<16x24x!pphlo.pub<f32>>, tensor<16x24x!pphlo.pub<f32>>) -> tensor<16x24x!pphlo.pub<f32>>
    %203 = "pphlo.add"(%202, %9) : (tensor<16x24x!pphlo.pub<f32>>, tensor<16x24x!pphlo.pub<f32>>) -> tensor<16x24x!pphlo.pub<f32>>
    %204 = "pphlo.divide"(%195, %203) : (tensor<16x24x!pphlo.pub<f32>>, tensor<16x24x!pphlo.pub<f32>>) -> tensor<16x24x!pphlo.pub<f32>>
    %205 = "pphlo.subtract"(%arg6, %204) : (tensor<16x24x!pphlo.pub<f32>>, tensor<16x24x!pphlo.pub<f32>>) -> tensor<16x24x!pphlo.pub<f32>>
    %206 = "pphlo.reduce"(%137, %3) ({
    ^bb0(%arg44: tensor<!pphlo.pub<f32>>, %arg45: tensor<!pphlo.pub<f32>>):
      %327 = "pphlo.add"(%arg44, %arg45) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
      "pphlo.return"(%327) : (tensor<!pphlo.pub<f32>>) -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<15x24x!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<24x!pphlo.pub<f32>>
    %207 = "pphlo.subtract"(%206, %arg26) : (tensor<24x!pphlo.pub<f32>>, tensor<24x!pphlo.pub<f32>>) -> tensor<24x!pphlo.pub<f32>>
    %208 = "pphlo.broadcast"(%144) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pub<f32>>) -> tensor<24x!pphlo.pub<f32>>
    %209 = "pphlo.multiply"(%207, %208) : (tensor<24x!pphlo.pub<f32>>, tensor<24x!pphlo.pub<f32>>) -> tensor<24x!pphlo.pub<f32>>
    %210 = "pphlo.add"(%arg26, %209) : (tensor<24x!pphlo.pub<f32>>, tensor<24x!pphlo.pub<f32>>) -> tensor<24x!pphlo.pub<f32>>
    %211 = "pphlo.broadcast"(%157) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pub<f32>>) -> tensor<24x!pphlo.pub<f32>>
    %212 = "pphlo.multiply"(%210, %211) : (tensor<24x!pphlo.pub<f32>>, tensor<24x!pphlo.pub<f32>>) -> tensor<24x!pphlo.pub<f32>>
    %213 = "pphlo.multiply"(%206, %206) : (tensor<24x!pphlo.pub<f32>>, tensor<24x!pphlo.pub<f32>>) -> tensor<24x!pphlo.pub<f32>>
    %214 = "pphlo.subtract"(%213, %arg27) : (tensor<24x!pphlo.pub<f32>>, tensor<24x!pphlo.pub<f32>>) -> tensor<24x!pphlo.pub<f32>>
    %215 = "pphlo.broadcast"(%162) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pub<f32>>) -> tensor<24x!pphlo.pub<f32>>
    %216 = "pphlo.multiply"(%214, %215) : (tensor<24x!pphlo.pub<f32>>, tensor<24x!pphlo.pub<f32>>) -> tensor<24x!pphlo.pub<f32>>
    %217 = "pphlo.add"(%arg27, %216) : (tensor<24x!pphlo.pub<f32>>, tensor<24x!pphlo.pub<f32>>) -> tensor<24x!pphlo.pub<f32>>
    %218 = "pphlo.constant"() {value = dense<5.000000e-01> : tensor<24xf32>} : () -> tensor<24x!pphlo.pub<f32>>
    %219 = "pphlo.power"(%217, %218) : (tensor<24x!pphlo.pub<f32>>, tensor<24x!pphlo.pub<f32>>) -> tensor<24x!pphlo.pub<f32>>
    %220 = "pphlo.add"(%219, %5) : (tensor<24x!pphlo.pub<f32>>, tensor<24x!pphlo.pub<f32>>) -> tensor<24x!pphlo.pub<f32>>
    %221 = "pphlo.divide"(%212, %220) : (tensor<24x!pphlo.pub<f32>>, tensor<24x!pphlo.pub<f32>>) -> tensor<24x!pphlo.pub<f32>>
    %222 = "pphlo.subtract"(%arg7, %221) : (tensor<24x!pphlo.pub<f32>>, tensor<24x!pphlo.pub<f32>>) -> tensor<24x!pphlo.pub<f32>>
    %223 = "pphlo.transpose"(%39) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<15x24x!pphlo.pub<f32>>) -> tensor<24x15x!pphlo.pub<f32>>
    %224 = "pphlo.dot"(%223, %132) : (tensor<24x15x!pphlo.pub<f32>>, tensor<15x20x!pphlo.pub<f32>>) -> tensor<24x20x!pphlo.pub<f32>>
    %225 = "pphlo.subtract"(%224, %arg28) : (tensor<24x20x!pphlo.pub<f32>>, tensor<24x20x!pphlo.pub<f32>>) -> tensor<24x20x!pphlo.pub<f32>>
    %226 = "pphlo.broadcast"(%144) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pub<f32>>) -> tensor<24x20x!pphlo.pub<f32>>
    %227 = "pphlo.multiply"(%225, %226) : (tensor<24x20x!pphlo.pub<f32>>, tensor<24x20x!pphlo.pub<f32>>) -> tensor<24x20x!pphlo.pub<f32>>
    %228 = "pphlo.add"(%arg28, %227) : (tensor<24x20x!pphlo.pub<f32>>, tensor<24x20x!pphlo.pub<f32>>) -> tensor<24x20x!pphlo.pub<f32>>
    %229 = "pphlo.broadcast"(%157) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pub<f32>>) -> tensor<24x20x!pphlo.pub<f32>>
    %230 = "pphlo.multiply"(%228, %229) : (tensor<24x20x!pphlo.pub<f32>>, tensor<24x20x!pphlo.pub<f32>>) -> tensor<24x20x!pphlo.pub<f32>>
    %231 = "pphlo.multiply"(%224, %224) : (tensor<24x20x!pphlo.pub<f32>>, tensor<24x20x!pphlo.pub<f32>>) -> tensor<24x20x!pphlo.pub<f32>>
    %232 = "pphlo.subtract"(%231, %arg29) : (tensor<24x20x!pphlo.pub<f32>>, tensor<24x20x!pphlo.pub<f32>>) -> tensor<24x20x!pphlo.pub<f32>>
    %233 = "pphlo.broadcast"(%162) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pub<f32>>) -> tensor<24x20x!pphlo.pub<f32>>
    %234 = "pphlo.multiply"(%232, %233) : (tensor<24x20x!pphlo.pub<f32>>, tensor<24x20x!pphlo.pub<f32>>) -> tensor<24x20x!pphlo.pub<f32>>
    %235 = "pphlo.add"(%arg29, %234) : (tensor<24x20x!pphlo.pub<f32>>, tensor<24x20x!pphlo.pub<f32>>) -> tensor<24x20x!pphlo.pub<f32>>
    %236 = "pphlo.constant"() {value = dense<5.000000e-01> : tensor<24x20xf32>} : () -> tensor<24x20x!pphlo.pub<f32>>
    %237 = "pphlo.power"(%235, %236) : (tensor<24x20x!pphlo.pub<f32>>, tensor<24x20x!pphlo.pub<f32>>) -> tensor<24x20x!pphlo.pub<f32>>
    %238 = "pphlo.add"(%237, %8) : (tensor<24x20x!pphlo.pub<f32>>, tensor<24x20x!pphlo.pub<f32>>) -> tensor<24x20x!pphlo.pub<f32>>
    %239 = "pphlo.divide"(%230, %238) : (tensor<24x20x!pphlo.pub<f32>>, tensor<24x20x!pphlo.pub<f32>>) -> tensor<24x20x!pphlo.pub<f32>>
    %240 = "pphlo.subtract"(%arg8, %239) : (tensor<24x20x!pphlo.pub<f32>>, tensor<24x20x!pphlo.pub<f32>>) -> tensor<24x20x!pphlo.pub<f32>>
    %241 = "pphlo.reduce"(%132, %3) ({
    ^bb0(%arg44: tensor<!pphlo.pub<f32>>, %arg45: tensor<!pphlo.pub<f32>>):
      %327 = "pphlo.add"(%arg44, %arg45) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
      "pphlo.return"(%327) : (tensor<!pphlo.pub<f32>>) -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<15x20x!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<20x!pphlo.pub<f32>>
    %242 = "pphlo.subtract"(%241, %arg30) : (tensor<20x!pphlo.pub<f32>>, tensor<20x!pphlo.pub<f32>>) -> tensor<20x!pphlo.pub<f32>>
    %243 = "pphlo.broadcast"(%144) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pub<f32>>) -> tensor<20x!pphlo.pub<f32>>
    %244 = "pphlo.multiply"(%242, %243) : (tensor<20x!pphlo.pub<f32>>, tensor<20x!pphlo.pub<f32>>) -> tensor<20x!pphlo.pub<f32>>
    %245 = "pphlo.add"(%arg30, %244) : (tensor<20x!pphlo.pub<f32>>, tensor<20x!pphlo.pub<f32>>) -> tensor<20x!pphlo.pub<f32>>
    %246 = "pphlo.broadcast"(%157) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pub<f32>>) -> tensor<20x!pphlo.pub<f32>>
    %247 = "pphlo.multiply"(%245, %246) : (tensor<20x!pphlo.pub<f32>>, tensor<20x!pphlo.pub<f32>>) -> tensor<20x!pphlo.pub<f32>>
    %248 = "pphlo.multiply"(%241, %241) : (tensor<20x!pphlo.pub<f32>>, tensor<20x!pphlo.pub<f32>>) -> tensor<20x!pphlo.pub<f32>>
    %249 = "pphlo.subtract"(%248, %arg31) : (tensor<20x!pphlo.pub<f32>>, tensor<20x!pphlo.pub<f32>>) -> tensor<20x!pphlo.pub<f32>>
    %250 = "pphlo.broadcast"(%162) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pub<f32>>) -> tensor<20x!pphlo.pub<f32>>
    %251 = "pphlo.multiply"(%249, %250) : (tensor<20x!pphlo.pub<f32>>, tensor<20x!pphlo.pub<f32>>) -> tensor<20x!pphlo.pub<f32>>
    %252 = "pphlo.add"(%arg31, %251) : (tensor<20x!pphlo.pub<f32>>, tensor<20x!pphlo.pub<f32>>) -> tensor<20x!pphlo.pub<f32>>
    %253 = "pphlo.constant"() {value = dense<5.000000e-01> : tensor<20xf32>} : () -> tensor<20x!pphlo.pub<f32>>
    %254 = "pphlo.power"(%252, %253) : (tensor<20x!pphlo.pub<f32>>, tensor<20x!pphlo.pub<f32>>) -> tensor<20x!pphlo.pub<f32>>
    %255 = "pphlo.add"(%254, %7) : (tensor<20x!pphlo.pub<f32>>, tensor<20x!pphlo.pub<f32>>) -> tensor<20x!pphlo.pub<f32>>
    %256 = "pphlo.divide"(%247, %255) : (tensor<20x!pphlo.pub<f32>>, tensor<20x!pphlo.pub<f32>>) -> tensor<20x!pphlo.pub<f32>>
    %257 = "pphlo.subtract"(%arg9, %256) : (tensor<20x!pphlo.pub<f32>>, tensor<20x!pphlo.pub<f32>>) -> tensor<20x!pphlo.pub<f32>>
    %258 = "pphlo.transpose"(%43) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<15x20x!pphlo.pub<f32>>) -> tensor<20x15x!pphlo.pub<f32>>
    %259 = "pphlo.dot"(%258, %129) : (tensor<20x15x!pphlo.pub<f32>>, tensor<15x24x!pphlo.pub<f32>>) -> tensor<20x24x!pphlo.pub<f32>>
    %260 = "pphlo.subtract"(%259, %arg32) : (tensor<20x24x!pphlo.pub<f32>>, tensor<20x24x!pphlo.pub<f32>>) -> tensor<20x24x!pphlo.pub<f32>>
    %261 = "pphlo.broadcast"(%144) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pub<f32>>) -> tensor<20x24x!pphlo.pub<f32>>
    %262 = "pphlo.multiply"(%260, %261) : (tensor<20x24x!pphlo.pub<f32>>, tensor<20x24x!pphlo.pub<f32>>) -> tensor<20x24x!pphlo.pub<f32>>
    %263 = "pphlo.add"(%arg32, %262) : (tensor<20x24x!pphlo.pub<f32>>, tensor<20x24x!pphlo.pub<f32>>) -> tensor<20x24x!pphlo.pub<f32>>
    %264 = "pphlo.broadcast"(%157) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pub<f32>>) -> tensor<20x24x!pphlo.pub<f32>>
    %265 = "pphlo.multiply"(%263, %264) : (tensor<20x24x!pphlo.pub<f32>>, tensor<20x24x!pphlo.pub<f32>>) -> tensor<20x24x!pphlo.pub<f32>>
    %266 = "pphlo.multiply"(%259, %259) : (tensor<20x24x!pphlo.pub<f32>>, tensor<20x24x!pphlo.pub<f32>>) -> tensor<20x24x!pphlo.pub<f32>>
    %267 = "pphlo.subtract"(%266, %arg33) : (tensor<20x24x!pphlo.pub<f32>>, tensor<20x24x!pphlo.pub<f32>>) -> tensor<20x24x!pphlo.pub<f32>>
    %268 = "pphlo.broadcast"(%162) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pub<f32>>) -> tensor<20x24x!pphlo.pub<f32>>
    %269 = "pphlo.multiply"(%267, %268) : (tensor<20x24x!pphlo.pub<f32>>, tensor<20x24x!pphlo.pub<f32>>) -> tensor<20x24x!pphlo.pub<f32>>
    %270 = "pphlo.add"(%arg33, %269) : (tensor<20x24x!pphlo.pub<f32>>, tensor<20x24x!pphlo.pub<f32>>) -> tensor<20x24x!pphlo.pub<f32>>
    %271 = "pphlo.constant"() {value = dense<5.000000e-01> : tensor<20x24xf32>} : () -> tensor<20x24x!pphlo.pub<f32>>
    %272 = "pphlo.power"(%270, %271) : (tensor<20x24x!pphlo.pub<f32>>, tensor<20x24x!pphlo.pub<f32>>) -> tensor<20x24x!pphlo.pub<f32>>
    %273 = "pphlo.add"(%272, %6) : (tensor<20x24x!pphlo.pub<f32>>, tensor<20x24x!pphlo.pub<f32>>) -> tensor<20x24x!pphlo.pub<f32>>
    %274 = "pphlo.divide"(%265, %273) : (tensor<20x24x!pphlo.pub<f32>>, tensor<20x24x!pphlo.pub<f32>>) -> tensor<20x24x!pphlo.pub<f32>>
    %275 = "pphlo.subtract"(%arg10, %274) : (tensor<20x24x!pphlo.pub<f32>>, tensor<20x24x!pphlo.pub<f32>>) -> tensor<20x24x!pphlo.pub<f32>>
    %276 = "pphlo.reduce"(%129, %3) ({
    ^bb0(%arg44: tensor<!pphlo.pub<f32>>, %arg45: tensor<!pphlo.pub<f32>>):
      %327 = "pphlo.add"(%arg44, %arg45) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
      "pphlo.return"(%327) : (tensor<!pphlo.pub<f32>>) -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<15x24x!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<24x!pphlo.pub<f32>>
    %277 = "pphlo.subtract"(%276, %arg34) : (tensor<24x!pphlo.pub<f32>>, tensor<24x!pphlo.pub<f32>>) -> tensor<24x!pphlo.pub<f32>>
    %278 = "pphlo.multiply"(%277, %208) : (tensor<24x!pphlo.pub<f32>>, tensor<24x!pphlo.pub<f32>>) -> tensor<24x!pphlo.pub<f32>>
    %279 = "pphlo.add"(%arg34, %278) : (tensor<24x!pphlo.pub<f32>>, tensor<24x!pphlo.pub<f32>>) -> tensor<24x!pphlo.pub<f32>>
    %280 = "pphlo.multiply"(%279, %211) : (tensor<24x!pphlo.pub<f32>>, tensor<24x!pphlo.pub<f32>>) -> tensor<24x!pphlo.pub<f32>>
    %281 = "pphlo.multiply"(%276, %276) : (tensor<24x!pphlo.pub<f32>>, tensor<24x!pphlo.pub<f32>>) -> tensor<24x!pphlo.pub<f32>>
    %282 = "pphlo.subtract"(%281, %arg35) : (tensor<24x!pphlo.pub<f32>>, tensor<24x!pphlo.pub<f32>>) -> tensor<24x!pphlo.pub<f32>>
    %283 = "pphlo.multiply"(%282, %215) : (tensor<24x!pphlo.pub<f32>>, tensor<24x!pphlo.pub<f32>>) -> tensor<24x!pphlo.pub<f32>>
    %284 = "pphlo.add"(%arg35, %283) : (tensor<24x!pphlo.pub<f32>>, tensor<24x!pphlo.pub<f32>>) -> tensor<24x!pphlo.pub<f32>>
    %285 = "pphlo.power"(%284, %218) : (tensor<24x!pphlo.pub<f32>>, tensor<24x!pphlo.pub<f32>>) -> tensor<24x!pphlo.pub<f32>>
    %286 = "pphlo.add"(%285, %5) : (tensor<24x!pphlo.pub<f32>>, tensor<24x!pphlo.pub<f32>>) -> tensor<24x!pphlo.pub<f32>>
    %287 = "pphlo.divide"(%280, %286) : (tensor<24x!pphlo.pub<f32>>, tensor<24x!pphlo.pub<f32>>) -> tensor<24x!pphlo.pub<f32>>
    %288 = "pphlo.subtract"(%arg11, %287) : (tensor<24x!pphlo.pub<f32>>, tensor<24x!pphlo.pub<f32>>) -> tensor<24x!pphlo.pub<f32>>
    %289 = "pphlo.transpose"(%47) {permutation = dense<[1, 0]> : tensor<2xi64>, xla_shape = "f32[24,15]{0,1}"} : (tensor<15x24x!pphlo.pub<f32>>) -> tensor<24x15x!pphlo.pub<f32>>
    %290 = "pphlo.broadcast"(%126) {broadcast_dimensions = dense<1> : tensor<1xi64>, xla_shape = "f32[24,15]{0,1}"} : (tensor<15x!pphlo.pub<f32>>) -> tensor<24x15x!pphlo.pub<f32>>
    %291 = "pphlo.multiply"(%289, %290) {xla_shape = "f32[24,15]{0,1}"} : (tensor<24x15x!pphlo.pub<f32>>, tensor<24x15x!pphlo.pub<f32>>) -> tensor<24x15x!pphlo.pub<f32>>
    %292 = "pphlo.reduce"(%291, %3) ({
    ^bb0(%arg44: tensor<!pphlo.pub<f32>>, %arg45: tensor<!pphlo.pub<f32>>):
      %327 = "pphlo.add"(%arg44, %arg45) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
      "pphlo.return"(%327) : (tensor<!pphlo.pub<f32>>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<24x15x!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<24x!pphlo.pub<f32>>
    %293 = "pphlo.reshape"(%292) : (tensor<24x!pphlo.pub<f32>>) -> tensor<24x1x!pphlo.pub<f32>>
    %294 = "pphlo.subtract"(%293, %arg36) : (tensor<24x1x!pphlo.pub<f32>>, tensor<24x1x!pphlo.pub<f32>>) -> tensor<24x1x!pphlo.pub<f32>>
    %295 = "pphlo.broadcast"(%144) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pub<f32>>) -> tensor<24x1x!pphlo.pub<f32>>
    %296 = "pphlo.multiply"(%294, %295) : (tensor<24x1x!pphlo.pub<f32>>, tensor<24x1x!pphlo.pub<f32>>) -> tensor<24x1x!pphlo.pub<f32>>
    %297 = "pphlo.add"(%arg36, %296) : (tensor<24x1x!pphlo.pub<f32>>, tensor<24x1x!pphlo.pub<f32>>) -> tensor<24x1x!pphlo.pub<f32>>
    %298 = "pphlo.broadcast"(%157) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pub<f32>>) -> tensor<24x1x!pphlo.pub<f32>>
    %299 = "pphlo.multiply"(%297, %298) : (tensor<24x1x!pphlo.pub<f32>>, tensor<24x1x!pphlo.pub<f32>>) -> tensor<24x1x!pphlo.pub<f32>>
    %300 = "pphlo.multiply"(%293, %293) : (tensor<24x1x!pphlo.pub<f32>>, tensor<24x1x!pphlo.pub<f32>>) -> tensor<24x1x!pphlo.pub<f32>>
    %301 = "pphlo.subtract"(%300, %arg37) : (tensor<24x1x!pphlo.pub<f32>>, tensor<24x1x!pphlo.pub<f32>>) -> tensor<24x1x!pphlo.pub<f32>>
    %302 = "pphlo.broadcast"(%162) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pub<f32>>) -> tensor<24x1x!pphlo.pub<f32>>
    %303 = "pphlo.multiply"(%301, %302) : (tensor<24x1x!pphlo.pub<f32>>, tensor<24x1x!pphlo.pub<f32>>) -> tensor<24x1x!pphlo.pub<f32>>
    %304 = "pphlo.add"(%arg37, %303) : (tensor<24x1x!pphlo.pub<f32>>, tensor<24x1x!pphlo.pub<f32>>) -> tensor<24x1x!pphlo.pub<f32>>
    %305 = "pphlo.constant"() {value = dense<5.000000e-01> : tensor<24x1xf32>} : () -> tensor<24x1x!pphlo.pub<f32>>
    %306 = "pphlo.power"(%304, %305) : (tensor<24x1x!pphlo.pub<f32>>, tensor<24x1x!pphlo.pub<f32>>) -> tensor<24x1x!pphlo.pub<f32>>
    %307 = "pphlo.add"(%306, %4) : (tensor<24x1x!pphlo.pub<f32>>, tensor<24x1x!pphlo.pub<f32>>) -> tensor<24x1x!pphlo.pub<f32>>
    %308 = "pphlo.divide"(%299, %307) : (tensor<24x1x!pphlo.pub<f32>>, tensor<24x1x!pphlo.pub<f32>>) -> tensor<24x1x!pphlo.pub<f32>>
    %309 = "pphlo.subtract"(%arg12, %308) : (tensor<24x1x!pphlo.pub<f32>>, tensor<24x1x!pphlo.pub<f32>>) -> tensor<24x1x!pphlo.pub<f32>>
    %310 = "pphlo.reduce"(%125, %3) ({
    ^bb0(%arg44: tensor<!pphlo.pub<f32>>, %arg45: tensor<!pphlo.pub<f32>>):
      %327 = "pphlo.add"(%arg44, %arg45) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
      "pphlo.return"(%327) : (tensor<!pphlo.pub<f32>>) -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<15x1x!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<1x!pphlo.pub<f32>>
    %311 = "pphlo.subtract"(%310, %arg38) : (tensor<1x!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>) -> tensor<1x!pphlo.pub<f32>>
    %312 = "pphlo.reshape"(%144) : (tensor<!pphlo.pub<f32>>) -> tensor<1x!pphlo.pub<f32>>
    %313 = "pphlo.multiply"(%311, %312) : (tensor<1x!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>) -> tensor<1x!pphlo.pub<f32>>
    %314 = "pphlo.add"(%arg38, %313) : (tensor<1x!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>) -> tensor<1x!pphlo.pub<f32>>
    %315 = "pphlo.reshape"(%157) : (tensor<!pphlo.pub<f32>>) -> tensor<1x!pphlo.pub<f32>>
    %316 = "pphlo.multiply"(%314, %315) : (tensor<1x!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>) -> tensor<1x!pphlo.pub<f32>>
    %317 = "pphlo.multiply"(%310, %310) : (tensor<1x!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>) -> tensor<1x!pphlo.pub<f32>>
    %318 = "pphlo.subtract"(%317, %arg39) : (tensor<1x!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>) -> tensor<1x!pphlo.pub<f32>>
    %319 = "pphlo.reshape"(%162) : (tensor<!pphlo.pub<f32>>) -> tensor<1x!pphlo.pub<f32>>
    %320 = "pphlo.multiply"(%318, %319) : (tensor<1x!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>) -> tensor<1x!pphlo.pub<f32>>
    %321 = "pphlo.add"(%arg39, %320) : (tensor<1x!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>) -> tensor<1x!pphlo.pub<f32>>
    %322 = "pphlo.constant"() {value = dense<5.000000e-01> : tensor<1xf32>} : () -> tensor<1x!pphlo.pub<f32>>
    %323 = "pphlo.power"(%321, %322) : (tensor<1x!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>) -> tensor<1x!pphlo.pub<f32>>
    %324 = "pphlo.add"(%323, %1) : (tensor<1x!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>) -> tensor<1x!pphlo.pub<f32>>
    %325 = "pphlo.divide"(%316, %324) : (tensor<1x!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>) -> tensor<1x!pphlo.pub<f32>>
    %326 = "pphlo.subtract"(%arg13, %325) : (tensor<1x!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>) -> tensor<1x!pphlo.pub<f32>>
    return %74, %94, %106, %170, %187, %205, %222, %240, %257, %275, %288, %309, %326, %72, %24, %148, %147, %165, %175, %182, %193, %200, %210, %217, %228, %235, %245, %252, %263, %270, %279, %284, %297, %304, %314, %321, %84, %89, %96, %101 : tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>, tensor<29x16x!pphlo.pub<f32>>, tensor<16x!pphlo.pub<f32>>, tensor<16x24x!pphlo.pub<f32>>, tensor<24x!pphlo.pub<f32>>, tensor<24x20x!pphlo.pub<f32>>, tensor<20x!pphlo.pub<f32>>, tensor<20x24x!pphlo.pub<f32>>, tensor<24x!pphlo.pub<f32>>, tensor<24x1x!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<i64>>, tensor<29x16x!pphlo.pub<f32>>, tensor<29x16x!pphlo.pub<f32>>, tensor<16x!pphlo.pub<f32>>, tensor<16x!pphlo.pub<f32>>, tensor<16x24x!pphlo.pub<f32>>, tensor<16x24x!pphlo.pub<f32>>, tensor<24x!pphlo.pub<f32>>, tensor<24x!pphlo.pub<f32>>, tensor<24x20x!pphlo.pub<f32>>, tensor<24x20x!pphlo.pub<f32>>, tensor<20x!pphlo.pub<f32>>, tensor<20x!pphlo.pub<f32>>, tensor<20x24x!pphlo.pub<f32>>, tensor<20x24x!pphlo.pub<f32>>, tensor<24x!pphlo.pub<f32>>, tensor<24x!pphlo.pub<f32>>, tensor<24x1x!pphlo.pub<f32>>, tensor<24x1x!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>
  }
}
