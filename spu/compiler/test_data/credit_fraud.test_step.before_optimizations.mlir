module @a_inference_test_step_123449__XlaMustCompile_true_config_proto___n_007_n_003CPU_020_001_n_007_n_003GPU_020_0002_002J_0008_001_202_001_000__executor_type____.121 {
  func @main(%arg0: tensor<100x3x!pphlo.pub<f32>>, %arg1: tensor<100x26x!pphlo.pub<f32>>, %arg2: tensor<100x!pphlo.pub<i32>>, %arg3: tensor<29x16x!pphlo.pub<f32>>, %arg4: tensor<16x!pphlo.pub<f32>>, %arg5: tensor<16x24x!pphlo.pub<f32>>, %arg6: tensor<24x!pphlo.pub<f32>>, %arg7: tensor<24x20x!pphlo.pub<f32>>, %arg8: tensor<20x!pphlo.pub<f32>>, %arg9: tensor<20x24x!pphlo.pub<f32>>, %arg10: tensor<24x!pphlo.pub<f32>>, %arg11: tensor<24x1x!pphlo.pub<f32>>, %arg12: tensor<1x!pphlo.pub<f32>>, %arg13: tensor<!pphlo.pub<f32>>, %arg14: tensor<!pphlo.pub<f32>>) -> (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) {
    %0 = "pphlo.constant"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<!pphlo.pub<f32>>
    %1 = "pphlo.constant"() {value = dense<0.000000e+00> : tensor<100x1xf32>} : () -> tensor<100x1x!pphlo.pub<f32>>
    %2 = "pphlo.constant"() {value = dense<0.000000e+00> : tensor<100x24xf32>} : () -> tensor<100x24x!pphlo.pub<f32>>
    %3 = "pphlo.constant"() {value = dense<0.000000e+00> : tensor<100x20xf32>} : () -> tensor<100x20x!pphlo.pub<f32>>
    %4 = "pphlo.constant"() {value = dense<0.000000e+00> : tensor<100x16xf32>} : () -> tensor<100x16x!pphlo.pub<f32>>
    %5 = "pphlo.constant"() {value = dense<1.000000e+02> : tensor<f32>} : () -> tensor<!pphlo.pub<f32>>
    %6 = "pphlo.add"(%arg14, %5) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    %7 = "pphlo.equal"(%6, %0) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<i1>>
    %8 = "pphlo.concatenate"(%arg1, %arg0) {dimension = 1 : i64} : (tensor<100x26x!pphlo.pub<f32>>, tensor<100x3x!pphlo.pub<f32>>) -> tensor<100x29x!pphlo.pub<f32>>
    %9 = "pphlo.dot"(%8, %arg3) : (tensor<100x29x!pphlo.pub<f32>>, tensor<29x16x!pphlo.pub<f32>>) -> tensor<100x16x!pphlo.pub<f32>>
    %10 = "pphlo.broadcast"(%arg4) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<16x!pphlo.pub<f32>>) -> tensor<100x16x!pphlo.pub<f32>>
    %11 = "pphlo.add"(%9, %10) : (tensor<100x16x!pphlo.pub<f32>>, tensor<100x16x!pphlo.pub<f32>>) -> tensor<100x16x!pphlo.pub<f32>>
    %12 = "pphlo.maximum"(%11, %4) : (tensor<100x16x!pphlo.pub<f32>>, tensor<100x16x!pphlo.pub<f32>>) -> tensor<100x16x!pphlo.pub<f32>>
    %13 = "pphlo.dot"(%12, %arg5) : (tensor<100x16x!pphlo.pub<f32>>, tensor<16x24x!pphlo.pub<f32>>) -> tensor<100x24x!pphlo.pub<f32>>
    %14 = "pphlo.broadcast"(%arg6) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<24x!pphlo.pub<f32>>) -> tensor<100x24x!pphlo.pub<f32>>
    %15 = "pphlo.add"(%13, %14) : (tensor<100x24x!pphlo.pub<f32>>, tensor<100x24x!pphlo.pub<f32>>) -> tensor<100x24x!pphlo.pub<f32>>
    %16 = "pphlo.maximum"(%15, %2) : (tensor<100x24x!pphlo.pub<f32>>, tensor<100x24x!pphlo.pub<f32>>) -> tensor<100x24x!pphlo.pub<f32>>
    %17 = "pphlo.dot"(%16, %arg7) : (tensor<100x24x!pphlo.pub<f32>>, tensor<24x20x!pphlo.pub<f32>>) -> tensor<100x20x!pphlo.pub<f32>>
    %18 = "pphlo.broadcast"(%arg8) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<20x!pphlo.pub<f32>>) -> tensor<100x20x!pphlo.pub<f32>>
    %19 = "pphlo.add"(%17, %18) : (tensor<100x20x!pphlo.pub<f32>>, tensor<100x20x!pphlo.pub<f32>>) -> tensor<100x20x!pphlo.pub<f32>>
    %20 = "pphlo.maximum"(%19, %3) : (tensor<100x20x!pphlo.pub<f32>>, tensor<100x20x!pphlo.pub<f32>>) -> tensor<100x20x!pphlo.pub<f32>>
    %21 = "pphlo.dot"(%20, %arg9) : (tensor<100x20x!pphlo.pub<f32>>, tensor<20x24x!pphlo.pub<f32>>) -> tensor<100x24x!pphlo.pub<f32>>
    %22 = "pphlo.broadcast"(%arg10) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<24x!pphlo.pub<f32>>) -> tensor<100x24x!pphlo.pub<f32>>
    %23 = "pphlo.add"(%21, %22) : (tensor<100x24x!pphlo.pub<f32>>, tensor<100x24x!pphlo.pub<f32>>) -> tensor<100x24x!pphlo.pub<f32>>
    %24 = "pphlo.maximum"(%23, %2) : (tensor<100x24x!pphlo.pub<f32>>, tensor<100x24x!pphlo.pub<f32>>) -> tensor<100x24x!pphlo.pub<f32>>
    %25 = "pphlo.reshape"(%arg11) : (tensor<24x1x!pphlo.pub<f32>>) -> tensor<24x!pphlo.pub<f32>>
    %26 = "pphlo.broadcast"(%25) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<24x!pphlo.pub<f32>>) -> tensor<100x24x!pphlo.pub<f32>>
    %27 = "pphlo.multiply"(%24, %26) : (tensor<100x24x!pphlo.pub<f32>>, tensor<100x24x!pphlo.pub<f32>>) -> tensor<100x24x!pphlo.pub<f32>>
    %28 = "pphlo.reduce"(%27, %0) ({
    ^bb0(%arg15: tensor<!pphlo.pub<f32>>, %arg16: tensor<!pphlo.pub<f32>>):
      %49 = "pphlo.add"(%arg15, %arg16) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
      "pphlo.return"(%49) : (tensor<!pphlo.pub<f32>>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<100x24x!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<100x!pphlo.pub<f32>>
    %29 = "pphlo.reshape"(%arg12) : (tensor<1x!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    %30 = "pphlo.broadcast"(%29) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pub<f32>>) -> tensor<100x!pphlo.pub<f32>>
    %31 = "pphlo.add"(%28, %30) : (tensor<100x!pphlo.pub<f32>>, tensor<100x!pphlo.pub<f32>>) -> tensor<100x!pphlo.pub<f32>>
    %32 = "pphlo.reshape"(%31) : (tensor<100x!pphlo.pub<f32>>) -> tensor<100x1x!pphlo.pub<f32>>
    %33 = "pphlo.less"(%32, %1) : (tensor<100x1x!pphlo.pub<f32>>, tensor<100x1x!pphlo.pub<f32>>) -> tensor<100x1x!pphlo.pub<i1>>
    %34 = "pphlo.not"(%33) : (tensor<100x1x!pphlo.pub<i1>>) -> tensor<100x1x!pphlo.pub<i1>>
    %35 = "pphlo.select"(%34, %32, %1) : (tensor<100x1x!pphlo.pub<i1>>, tensor<100x1x!pphlo.pub<f32>>, tensor<100x1x!pphlo.pub<f32>>) -> tensor<100x1x!pphlo.pub<f32>>
    %36 = "pphlo.convert"(%arg2) : (tensor<100x!pphlo.pub<i32>>) -> tensor<100x!pphlo.pub<f32>>
    %37 = "pphlo.reshape"(%36) : (tensor<100x!pphlo.pub<f32>>) -> tensor<100x1x!pphlo.pub<f32>>
    %38 = "pphlo.multiply"(%32, %37) : (tensor<100x1x!pphlo.pub<f32>>, tensor<100x1x!pphlo.pub<f32>>) -> tensor<100x1x!pphlo.pub<f32>>
    %39 = "pphlo.subtract"(%35, %38) : (tensor<100x1x!pphlo.pub<f32>>, tensor<100x1x!pphlo.pub<f32>>) -> tensor<100x1x!pphlo.pub<f32>>
    %40 = "pphlo.negate"(%32) : (tensor<100x1x!pphlo.pub<f32>>) -> tensor<100x1x!pphlo.pub<f32>>
    %41 = "pphlo.select"(%34, %40, %32) : (tensor<100x1x!pphlo.pub<i1>>, tensor<100x1x!pphlo.pub<f32>>, tensor<100x1x!pphlo.pub<f32>>) -> tensor<100x1x!pphlo.pub<f32>>
    %42 = "pphlo.exponential"(%41) : (tensor<100x1x!pphlo.pub<f32>>) -> tensor<100x1x!pphlo.pub<f32>>
    %43 = "pphlo.log_plus_one"(%42) : (tensor<100x1x!pphlo.pub<f32>>) -> tensor<100x1x!pphlo.pub<f32>>
    %44 = "pphlo.add"(%39, %43) : (tensor<100x1x!pphlo.pub<f32>>, tensor<100x1x!pphlo.pub<f32>>) -> tensor<100x1x!pphlo.pub<f32>>
    %45 = "pphlo.reduce"(%44, %0) ({
    ^bb0(%arg15: tensor<!pphlo.pub<f32>>, %arg16: tensor<!pphlo.pub<f32>>):
      %49 = "pphlo.add"(%arg15, %arg16) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
      "pphlo.return"(%49) : (tensor<!pphlo.pub<f32>>) -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<100x1x!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    %46 = "pphlo.add"(%arg13, %45) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    %47 = "pphlo.divide"(%46, %6) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    %48 = "pphlo.select"(%7, %0, %47) : (tensor<!pphlo.pub<i1>>, tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    return %48, %46, %6 : tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>
  }
}
