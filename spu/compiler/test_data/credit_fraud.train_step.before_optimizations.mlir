module @a_inference_train_step_1585__XlaMustCompile_true_config_proto___n_007_n_003CPU_020_001_n_007_n_003GPU_020_0002_002J_0008_001_202_001_000__executor_type____.683 {
  func @main(%arg0: tensor<15x3x!pphlo.pub<f32>>, %arg1: tensor<15x26x!pphlo.pub<f32>>, %arg2: tensor<15x!pphlo.pub<i32>>, %arg3: tensor<15x!pphlo.pub<i32>>, %arg4: tensor<29x16x!pphlo.pub<f32>>, %arg5: tensor<16x!pphlo.pub<f32>>, %arg6: tensor<16x24x!pphlo.pub<f32>>, %arg7: tensor<24x!pphlo.pub<f32>>, %arg8: tensor<24x20x!pphlo.pub<f32>>, %arg9: tensor<20x!pphlo.pub<f32>>, %arg10: tensor<20x24x!pphlo.pub<f32>>, %arg11: tensor<24x!pphlo.pub<f32>>, %arg12: tensor<24x1x!pphlo.pub<f32>>, %arg13: tensor<1x!pphlo.pub<f32>>, %arg14: tensor<!pphlo.pub<f32>>, %arg15: tensor<!pphlo.pub<f32>>, %arg16: tensor<!pphlo.pub<f32>>, %arg17: tensor<!pphlo.pub<i64>>, %arg18: tensor<!pphlo.pub<f32>>, %arg19: tensor<!pphlo.pub<f32>>, %arg20: tensor<29x16x!pphlo.pub<f32>>, %arg21: tensor<29x16x!pphlo.pub<f32>>, %arg22: tensor<16x!pphlo.pub<f32>>, %arg23: tensor<16x!pphlo.pub<f32>>, %arg24: tensor<16x24x!pphlo.pub<f32>>, %arg25: tensor<16x24x!pphlo.pub<f32>>, %arg26: tensor<24x!pphlo.pub<f32>>, %arg27: tensor<24x!pphlo.pub<f32>>, %arg28: tensor<24x20x!pphlo.pub<f32>>, %arg29: tensor<24x20x!pphlo.pub<f32>>, %arg30: tensor<20x!pphlo.pub<f32>>, %arg31: tensor<20x!pphlo.pub<f32>>, %arg32: tensor<20x24x!pphlo.pub<f32>>, %arg33: tensor<20x24x!pphlo.pub<f32>>, %arg34: tensor<24x!pphlo.pub<f32>>, %arg35: tensor<24x!pphlo.pub<f32>>, %arg36: tensor<24x1x!pphlo.pub<f32>>, %arg37: tensor<24x1x!pphlo.pub<f32>>, %arg38: tensor<1x!pphlo.pub<f32>>, %arg39: tensor<1x!pphlo.pub<f32>>) -> (tensor<!pphlo.pub<f32>>, tensor<29x16x!pphlo.pub<f32>>, tensor<16x!pphlo.pub<f32>>, tensor<16x24x!pphlo.pub<f32>>, tensor<24x!pphlo.pub<f32>>, tensor<24x20x!pphlo.pub<f32>>, tensor<20x!pphlo.pub<f32>>, tensor<20x24x!pphlo.pub<f32>>, tensor<24x!pphlo.pub<f32>>, tensor<24x1x!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<i64>>, tensor<29x16x!pphlo.pub<f32>>, tensor<29x16x!pphlo.pub<f32>>, tensor<16x!pphlo.pub<f32>>, tensor<16x!pphlo.pub<f32>>, tensor<16x24x!pphlo.pub<f32>>, tensor<16x24x!pphlo.pub<f32>>, tensor<24x!pphlo.pub<f32>>, tensor<24x!pphlo.pub<f32>>, tensor<24x20x!pphlo.pub<f32>>, tensor<24x20x!pphlo.pub<f32>>, tensor<20x!pphlo.pub<f32>>, tensor<20x!pphlo.pub<f32>>, tensor<20x24x!pphlo.pub<f32>>, tensor<20x24x!pphlo.pub<f32>>, tensor<24x!pphlo.pub<f32>>, tensor<24x!pphlo.pub<f32>>, tensor<24x1x!pphlo.pub<f32>>, tensor<24x1x!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>) {
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
    %19 = "pphlo.constant"() {value = dense<5.000000e-01> : tensor<15x24xf32>} : () -> tensor<15x24x!pphlo.pub<f32>>
    %20 = "pphlo.constant"() {value = dense<1.500000e+01> : tensor<f32>} : () -> tensor<!pphlo.pub<f32>>
    %21 = "pphlo.add"(%arg15, %20) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    %22 = "pphlo.equal"(%21, %3) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<i1>>
    %23 = "pphlo.rng_uniform"(%3, %2) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<15x24x!pphlo.pub<f32>>
    %24 = "pphlo.less"(%23, %19) : (tensor<15x24x!pphlo.pub<f32>>, tensor<15x24x!pphlo.pub<f32>>) -> tensor<15x24x!pphlo.pub<i1>>
    %25 = "pphlo.not"(%24) : (tensor<15x24x!pphlo.pub<i1>>) -> tensor<15x24x!pphlo.pub<i1>>
    %26 = "pphlo.concatenate"(%arg1, %arg0) {dimension = 1 : i64} : (tensor<15x26x!pphlo.pub<f32>>, tensor<15x3x!pphlo.pub<f32>>) -> tensor<15x29x!pphlo.pub<f32>>
    %27 = "pphlo.dot"(%26, %arg4) : (tensor<15x29x!pphlo.pub<f32>>, tensor<29x16x!pphlo.pub<f32>>) -> tensor<15x16x!pphlo.pub<f32>>
    %28 = "pphlo.broadcast"(%arg5) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<16x!pphlo.pub<f32>>) -> tensor<15x16x!pphlo.pub<f32>>
    %29 = "pphlo.add"(%27, %28) : (tensor<15x16x!pphlo.pub<f32>>, tensor<15x16x!pphlo.pub<f32>>) -> tensor<15x16x!pphlo.pub<f32>>
    %30 = "pphlo.maximum"(%29, %18) : (tensor<15x16x!pphlo.pub<f32>>, tensor<15x16x!pphlo.pub<f32>>) -> tensor<15x16x!pphlo.pub<f32>>
    %31 = "pphlo.dot"(%30, %arg6) : (tensor<15x16x!pphlo.pub<f32>>, tensor<16x24x!pphlo.pub<f32>>) -> tensor<15x24x!pphlo.pub<f32>>
    %32 = "pphlo.broadcast"(%arg7) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<24x!pphlo.pub<f32>>) -> tensor<15x24x!pphlo.pub<f32>>
    %33 = "pphlo.add"(%31, %32) : (tensor<15x24x!pphlo.pub<f32>>, tensor<15x24x!pphlo.pub<f32>>) -> tensor<15x24x!pphlo.pub<f32>>
    %34 = "pphlo.maximum"(%33, %13) : (tensor<15x24x!pphlo.pub<f32>>, tensor<15x24x!pphlo.pub<f32>>) -> tensor<15x24x!pphlo.pub<f32>>
    %35 = "pphlo.multiply"(%34, %12) : (tensor<15x24x!pphlo.pub<f32>>, tensor<15x24x!pphlo.pub<f32>>) -> tensor<15x24x!pphlo.pub<f32>>
    %36 = "pphlo.select"(%25, %35, %13) : (tensor<15x24x!pphlo.pub<i1>>, tensor<15x24x!pphlo.pub<f32>>, tensor<15x24x!pphlo.pub<f32>>) -> tensor<15x24x!pphlo.pub<f32>>
    %37 = "pphlo.dot"(%36, %arg8) : (tensor<15x24x!pphlo.pub<f32>>, tensor<24x20x!pphlo.pub<f32>>) -> tensor<15x20x!pphlo.pub<f32>>
    %38 = "pphlo.broadcast"(%arg9) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<20x!pphlo.pub<f32>>) -> tensor<15x20x!pphlo.pub<f32>>
    %39 = "pphlo.add"(%37, %38) : (tensor<15x20x!pphlo.pub<f32>>, tensor<15x20x!pphlo.pub<f32>>) -> tensor<15x20x!pphlo.pub<f32>>
    %40 = "pphlo.maximum"(%39, %17) : (tensor<15x20x!pphlo.pub<f32>>, tensor<15x20x!pphlo.pub<f32>>) -> tensor<15x20x!pphlo.pub<f32>>
    %41 = "pphlo.dot"(%40, %arg10) : (tensor<15x20x!pphlo.pub<f32>>, tensor<20x24x!pphlo.pub<f32>>) -> tensor<15x24x!pphlo.pub<f32>>
    %42 = "pphlo.broadcast"(%arg11) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<24x!pphlo.pub<f32>>) -> tensor<15x24x!pphlo.pub<f32>>
    %43 = "pphlo.add"(%41, %42) : (tensor<15x24x!pphlo.pub<f32>>, tensor<15x24x!pphlo.pub<f32>>) -> tensor<15x24x!pphlo.pub<f32>>
    %44 = "pphlo.maximum"(%43, %13) : (tensor<15x24x!pphlo.pub<f32>>, tensor<15x24x!pphlo.pub<f32>>) -> tensor<15x24x!pphlo.pub<f32>>
    %45 = "pphlo.reshape"(%arg12) : (tensor<24x1x!pphlo.pub<f32>>) -> tensor<24x!pphlo.pub<f32>>
    %46 = "pphlo.broadcast"(%45) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<24x!pphlo.pub<f32>>) -> tensor<15x24x!pphlo.pub<f32>>
    %47 = "pphlo.multiply"(%44, %46) : (tensor<15x24x!pphlo.pub<f32>>, tensor<15x24x!pphlo.pub<f32>>) -> tensor<15x24x!pphlo.pub<f32>>
    %48 = "pphlo.reduce"(%47, %3) ({
    ^bb0(%arg40: tensor<!pphlo.pub<f32>>, %arg41: tensor<!pphlo.pub<f32>>):
      %292 = "pphlo.add"(%arg40, %arg41) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
      "pphlo.return"(%292) : (tensor<!pphlo.pub<f32>>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<15x24x!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<15x!pphlo.pub<f32>>
    %49 = "pphlo.reshape"(%arg13) : (tensor<1x!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    %50 = "pphlo.broadcast"(%49) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pub<f32>>) -> tensor<15x!pphlo.pub<f32>>
    %51 = "pphlo.add"(%48, %50) : (tensor<15x!pphlo.pub<f32>>, tensor<15x!pphlo.pub<f32>>) -> tensor<15x!pphlo.pub<f32>>
    %52 = "pphlo.reshape"(%51) : (tensor<15x!pphlo.pub<f32>>) -> tensor<15x1x!pphlo.pub<f32>>
    %53 = "pphlo.less"(%52, %15) : (tensor<15x1x!pphlo.pub<f32>>, tensor<15x1x!pphlo.pub<f32>>) -> tensor<15x1x!pphlo.pub<i1>>
    %54 = "pphlo.not"(%53) : (tensor<15x1x!pphlo.pub<i1>>) -> tensor<15x1x!pphlo.pub<i1>>
    %55 = "pphlo.select"(%54, %52, %15) : (tensor<15x1x!pphlo.pub<i1>>, tensor<15x1x!pphlo.pub<f32>>, tensor<15x1x!pphlo.pub<f32>>) -> tensor<15x1x!pphlo.pub<f32>>
    %56 = "pphlo.convert"(%arg2) : (tensor<15x!pphlo.pub<i32>>) -> tensor<15x!pphlo.pub<f32>>
    %57 = "pphlo.reshape"(%56) : (tensor<15x!pphlo.pub<f32>>) -> tensor<15x1x!pphlo.pub<f32>>
    %58 = "pphlo.multiply"(%52, %57) : (tensor<15x1x!pphlo.pub<f32>>, tensor<15x1x!pphlo.pub<f32>>) -> tensor<15x1x!pphlo.pub<f32>>
    %59 = "pphlo.subtract"(%55, %58) : (tensor<15x1x!pphlo.pub<f32>>, tensor<15x1x!pphlo.pub<f32>>) -> tensor<15x1x!pphlo.pub<f32>>
    %60 = "pphlo.negate"(%52) : (tensor<15x1x!pphlo.pub<f32>>) -> tensor<15x1x!pphlo.pub<f32>>
    %61 = "pphlo.select"(%54, %60, %52) : (tensor<15x1x!pphlo.pub<i1>>, tensor<15x1x!pphlo.pub<f32>>, tensor<15x1x!pphlo.pub<f32>>) -> tensor<15x1x!pphlo.pub<f32>>
    %62 = "pphlo.exponential"(%61) : (tensor<15x1x!pphlo.pub<f32>>) -> tensor<15x1x!pphlo.pub<f32>>
    %63 = "pphlo.log_plus_one"(%62) : (tensor<15x1x!pphlo.pub<f32>>) -> tensor<15x1x!pphlo.pub<f32>>
    %64 = "pphlo.add"(%59, %63) : (tensor<15x1x!pphlo.pub<f32>>, tensor<15x1x!pphlo.pub<f32>>) -> tensor<15x1x!pphlo.pub<f32>>
    %65 = "pphlo.reshape"(%64) : (tensor<15x1x!pphlo.pub<f32>>) -> tensor<15x!pphlo.pub<f32>>
    %66 = "pphlo.convert"(%arg3) : (tensor<15x!pphlo.pub<i32>>) -> tensor<15x!pphlo.pub<f32>>
    %67 = "pphlo.multiply"(%65, %66) : (tensor<15x!pphlo.pub<f32>>, tensor<15x!pphlo.pub<f32>>) -> tensor<15x!pphlo.pub<f32>>
    %68 = "pphlo.reduce"(%67, %3) ({
    ^bb0(%arg40: tensor<!pphlo.pub<f32>>, %arg41: tensor<!pphlo.pub<f32>>):
      %292 = "pphlo.add"(%arg40, %arg41) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
      "pphlo.return"(%292) : (tensor<!pphlo.pub<f32>>) -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<15x!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    %69 = "pphlo.add"(%arg14, %68) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    %70 = "pphlo.divide"(%69, %21) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    %71 = "pphlo.select"(%22, %3, %70) : (tensor<!pphlo.pub<i1>>, tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    %72 = "pphlo.greater"(%30, %18) : (tensor<15x16x!pphlo.pub<f32>>, tensor<15x16x!pphlo.pub<f32>>) -> tensor<15x16x!pphlo.pub<i1>>
    %73 = "pphlo.greater"(%34, %13) : (tensor<15x24x!pphlo.pub<f32>>, tensor<15x24x!pphlo.pub<f32>>) -> tensor<15x24x!pphlo.pub<i1>>
    %74 = "pphlo.greater"(%40, %17) : (tensor<15x20x!pphlo.pub<f32>>, tensor<15x20x!pphlo.pub<f32>>) -> tensor<15x20x!pphlo.pub<i1>>
    %75 = "pphlo.greater"(%44, %13) : (tensor<15x24x!pphlo.pub<f32>>, tensor<15x24x!pphlo.pub<f32>>) -> tensor<15x24x!pphlo.pub<i1>>
    %76 = "pphlo.multiply"(%66, %16) : (tensor<15x!pphlo.pub<f32>>, tensor<15x!pphlo.pub<f32>>) -> tensor<15x!pphlo.pub<f32>>
    %77 = "pphlo.reshape"(%76) : (tensor<15x!pphlo.pub<f32>>) -> tensor<15x1x!pphlo.pub<f32>>
    %78 = "pphlo.select"(%54, %77, %15) : (tensor<15x1x!pphlo.pub<i1>>, tensor<15x1x!pphlo.pub<f32>>, tensor<15x1x!pphlo.pub<f32>>) -> tensor<15x1x!pphlo.pub<f32>>
    %79 = "pphlo.negate"(%77) : (tensor<15x1x!pphlo.pub<f32>>) -> tensor<15x1x!pphlo.pub<f32>>
    %80 = "pphlo.multiply"(%79, %57) : (tensor<15x1x!pphlo.pub<f32>>, tensor<15x1x!pphlo.pub<f32>>) -> tensor<15x1x!pphlo.pub<f32>>
    %81 = "pphlo.add"(%78, %80) : (tensor<15x1x!pphlo.pub<f32>>, tensor<15x1x!pphlo.pub<f32>>) -> tensor<15x1x!pphlo.pub<f32>>
    %82 = "pphlo.add"(%62, %14) : (tensor<15x1x!pphlo.pub<f32>>, tensor<15x1x!pphlo.pub<f32>>) -> tensor<15x1x!pphlo.pub<f32>>
    %83 = "pphlo.divide"(%14, %82) : (tensor<15x1x!pphlo.pub<f32>>, tensor<15x1x!pphlo.pub<f32>>) -> tensor<15x1x!pphlo.pub<f32>>
    %84 = "pphlo.multiply"(%77, %83) : (tensor<15x1x!pphlo.pub<f32>>, tensor<15x1x!pphlo.pub<f32>>) -> tensor<15x1x!pphlo.pub<f32>>
    %85 = "pphlo.multiply"(%84, %62) : (tensor<15x1x!pphlo.pub<f32>>, tensor<15x1x!pphlo.pub<f32>>) -> tensor<15x1x!pphlo.pub<f32>>
    %86 = "pphlo.select"(%54, %15, %85) : (tensor<15x1x!pphlo.pub<i1>>, tensor<15x1x!pphlo.pub<f32>>, tensor<15x1x!pphlo.pub<f32>>) -> tensor<15x1x!pphlo.pub<f32>>
    %87 = "pphlo.add"(%81, %86) : (tensor<15x1x!pphlo.pub<f32>>, tensor<15x1x!pphlo.pub<f32>>) -> tensor<15x1x!pphlo.pub<f32>>
    %88 = "pphlo.select"(%54, %85, %15) : (tensor<15x1x!pphlo.pub<i1>>, tensor<15x1x!pphlo.pub<f32>>, tensor<15x1x!pphlo.pub<f32>>) -> tensor<15x1x!pphlo.pub<f32>>
    %89 = "pphlo.negate"(%88) : (tensor<15x1x!pphlo.pub<f32>>) -> tensor<15x1x!pphlo.pub<f32>>
    %90 = "pphlo.add"(%87, %89) : (tensor<15x1x!pphlo.pub<f32>>, tensor<15x1x!pphlo.pub<f32>>) -> tensor<15x1x!pphlo.pub<f32>>
    %91 = "pphlo.reshape"(%90) : (tensor<15x1x!pphlo.pub<f32>>) -> tensor<15x!pphlo.pub<f32>>
    %92 = "pphlo.broadcast"(%91) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<15x!pphlo.pub<f32>>) -> tensor<15x24x!pphlo.pub<f32>>
    %93 = "pphlo.multiply"(%92, %46) : (tensor<15x24x!pphlo.pub<f32>>, tensor<15x24x!pphlo.pub<f32>>) -> tensor<15x24x!pphlo.pub<f32>>
    %94 = "pphlo.select"(%75, %93, %13) : (tensor<15x24x!pphlo.pub<i1>>, tensor<15x24x!pphlo.pub<f32>>, tensor<15x24x!pphlo.pub<f32>>) -> tensor<15x24x!pphlo.pub<f32>>
    %95 = "pphlo.transpose"(%arg10) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<20x24x!pphlo.pub<f32>>) -> tensor<24x20x!pphlo.pub<f32>>
    %96 = "pphlo.dot"(%94, %95) : (tensor<15x24x!pphlo.pub<f32>>, tensor<24x20x!pphlo.pub<f32>>) -> tensor<15x20x!pphlo.pub<f32>>
    %97 = "pphlo.select"(%74, %96, %17) : (tensor<15x20x!pphlo.pub<i1>>, tensor<15x20x!pphlo.pub<f32>>, tensor<15x20x!pphlo.pub<f32>>) -> tensor<15x20x!pphlo.pub<f32>>
    %98 = "pphlo.transpose"(%arg8) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<24x20x!pphlo.pub<f32>>) -> tensor<20x24x!pphlo.pub<f32>>
    %99 = "pphlo.dot"(%97, %98) : (tensor<15x20x!pphlo.pub<f32>>, tensor<20x24x!pphlo.pub<f32>>) -> tensor<15x24x!pphlo.pub<f32>>
    %100 = "pphlo.select"(%25, %99, %13) : (tensor<15x24x!pphlo.pub<i1>>, tensor<15x24x!pphlo.pub<f32>>, tensor<15x24x!pphlo.pub<f32>>) -> tensor<15x24x!pphlo.pub<f32>>
    %101 = "pphlo.multiply"(%100, %12) : (tensor<15x24x!pphlo.pub<f32>>, tensor<15x24x!pphlo.pub<f32>>) -> tensor<15x24x!pphlo.pub<f32>>
    %102 = "pphlo.select"(%73, %101, %13) : (tensor<15x24x!pphlo.pub<i1>>, tensor<15x24x!pphlo.pub<f32>>, tensor<15x24x!pphlo.pub<f32>>) -> tensor<15x24x!pphlo.pub<f32>>
    %103 = "pphlo.transpose"(%arg6) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<16x24x!pphlo.pub<f32>>) -> tensor<24x16x!pphlo.pub<f32>>
    %104 = "pphlo.dot"(%102, %103) : (tensor<15x24x!pphlo.pub<f32>>, tensor<24x16x!pphlo.pub<f32>>) -> tensor<15x16x!pphlo.pub<f32>>
    %105 = "pphlo.select"(%72, %104, %18) : (tensor<15x16x!pphlo.pub<i1>>, tensor<15x16x!pphlo.pub<f32>>, tensor<15x16x!pphlo.pub<f32>>) -> tensor<15x16x!pphlo.pub<f32>>
    %106 = "pphlo.transpose"(%26) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<15x29x!pphlo.pub<f32>>) -> tensor<29x15x!pphlo.pub<f32>>
    %107 = "pphlo.dot"(%106, %105) : (tensor<29x15x!pphlo.pub<f32>>, tensor<15x16x!pphlo.pub<f32>>) -> tensor<29x16x!pphlo.pub<f32>>
    %108 = "pphlo.subtract"(%107, %arg20) : (tensor<29x16x!pphlo.pub<f32>>, tensor<29x16x!pphlo.pub<f32>>) -> tensor<29x16x!pphlo.pub<f32>>
    %109 = "pphlo.subtract"(%2, %arg18) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    %110 = "pphlo.broadcast"(%109) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pub<f32>>) -> tensor<29x16x!pphlo.pub<f32>>
    %111 = "pphlo.multiply"(%108, %110) : (tensor<29x16x!pphlo.pub<f32>>, tensor<29x16x!pphlo.pub<f32>>) -> tensor<29x16x!pphlo.pub<f32>>
    %112 = "pphlo.add"(%arg20, %111) : (tensor<29x16x!pphlo.pub<f32>>, tensor<29x16x!pphlo.pub<f32>>) -> tensor<29x16x!pphlo.pub<f32>>
    %113 = "pphlo.add"(%arg17, %0) : (tensor<!pphlo.pub<i64>>, tensor<!pphlo.pub<i64>>) -> tensor<!pphlo.pub<i64>>
    %114 = "pphlo.convert"(%113) : (tensor<!pphlo.pub<i64>>) -> tensor<!pphlo.pub<f32>>
    %115 = "pphlo.power"(%arg19, %114) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    %116 = "pphlo.subtract"(%2, %115) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    %117 = "pphlo.constant"() {value = dense<5.000000e-01> : tensor<f32>} : () -> tensor<!pphlo.pub<f32>>
    %118 = "pphlo.power"(%116, %117) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    %119 = "pphlo.multiply"(%arg16, %118) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    %120 = "pphlo.power"(%arg18, %114) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    %121 = "pphlo.subtract"(%2, %120) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    %122 = "pphlo.divide"(%119, %121) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    %123 = "pphlo.broadcast"(%122) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pub<f32>>) -> tensor<29x16x!pphlo.pub<f32>>
    %124 = "pphlo.multiply"(%112, %123) : (tensor<29x16x!pphlo.pub<f32>>, tensor<29x16x!pphlo.pub<f32>>) -> tensor<29x16x!pphlo.pub<f32>>
    %125 = "pphlo.multiply"(%107, %107) : (tensor<29x16x!pphlo.pub<f32>>, tensor<29x16x!pphlo.pub<f32>>) -> tensor<29x16x!pphlo.pub<f32>>
    %126 = "pphlo.subtract"(%125, %arg21) : (tensor<29x16x!pphlo.pub<f32>>, tensor<29x16x!pphlo.pub<f32>>) -> tensor<29x16x!pphlo.pub<f32>>
    %127 = "pphlo.subtract"(%2, %arg19) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    %128 = "pphlo.broadcast"(%127) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pub<f32>>) -> tensor<29x16x!pphlo.pub<f32>>
    %129 = "pphlo.multiply"(%126, %128) : (tensor<29x16x!pphlo.pub<f32>>, tensor<29x16x!pphlo.pub<f32>>) -> tensor<29x16x!pphlo.pub<f32>>
    %130 = "pphlo.add"(%arg21, %129) : (tensor<29x16x!pphlo.pub<f32>>, tensor<29x16x!pphlo.pub<f32>>) -> tensor<29x16x!pphlo.pub<f32>>
    %131 = "pphlo.constant"() {value = dense<5.000000e-01> : tensor<29x16xf32>} : () -> tensor<29x16x!pphlo.pub<f32>>
    %132 = "pphlo.power"(%130, %131) : (tensor<29x16x!pphlo.pub<f32>>, tensor<29x16x!pphlo.pub<f32>>) -> tensor<29x16x!pphlo.pub<f32>>
    %133 = "pphlo.add"(%132, %11) : (tensor<29x16x!pphlo.pub<f32>>, tensor<29x16x!pphlo.pub<f32>>) -> tensor<29x16x!pphlo.pub<f32>>
    %134 = "pphlo.divide"(%124, %133) : (tensor<29x16x!pphlo.pub<f32>>, tensor<29x16x!pphlo.pub<f32>>) -> tensor<29x16x!pphlo.pub<f32>>
    %135 = "pphlo.subtract"(%arg4, %134) : (tensor<29x16x!pphlo.pub<f32>>, tensor<29x16x!pphlo.pub<f32>>) -> tensor<29x16x!pphlo.pub<f32>>
    %136 = "pphlo.reduce"(%105, %3) ({
    ^bb0(%arg40: tensor<!pphlo.pub<f32>>, %arg41: tensor<!pphlo.pub<f32>>):
      %292 = "pphlo.add"(%arg40, %arg41) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
      "pphlo.return"(%292) : (tensor<!pphlo.pub<f32>>) -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<15x16x!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<16x!pphlo.pub<f32>>
    %137 = "pphlo.subtract"(%136, %arg22) : (tensor<16x!pphlo.pub<f32>>, tensor<16x!pphlo.pub<f32>>) -> tensor<16x!pphlo.pub<f32>>
    %138 = "pphlo.broadcast"(%109) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pub<f32>>) -> tensor<16x!pphlo.pub<f32>>
    %139 = "pphlo.multiply"(%137, %138) : (tensor<16x!pphlo.pub<f32>>, tensor<16x!pphlo.pub<f32>>) -> tensor<16x!pphlo.pub<f32>>
    %140 = "pphlo.add"(%arg22, %139) : (tensor<16x!pphlo.pub<f32>>, tensor<16x!pphlo.pub<f32>>) -> tensor<16x!pphlo.pub<f32>>
    %141 = "pphlo.broadcast"(%122) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pub<f32>>) -> tensor<16x!pphlo.pub<f32>>
    %142 = "pphlo.multiply"(%140, %141) : (tensor<16x!pphlo.pub<f32>>, tensor<16x!pphlo.pub<f32>>) -> tensor<16x!pphlo.pub<f32>>
    %143 = "pphlo.multiply"(%136, %136) : (tensor<16x!pphlo.pub<f32>>, tensor<16x!pphlo.pub<f32>>) -> tensor<16x!pphlo.pub<f32>>
    %144 = "pphlo.subtract"(%143, %arg23) : (tensor<16x!pphlo.pub<f32>>, tensor<16x!pphlo.pub<f32>>) -> tensor<16x!pphlo.pub<f32>>
    %145 = "pphlo.broadcast"(%127) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pub<f32>>) -> tensor<16x!pphlo.pub<f32>>
    %146 = "pphlo.multiply"(%144, %145) : (tensor<16x!pphlo.pub<f32>>, tensor<16x!pphlo.pub<f32>>) -> tensor<16x!pphlo.pub<f32>>
    %147 = "pphlo.add"(%arg23, %146) : (tensor<16x!pphlo.pub<f32>>, tensor<16x!pphlo.pub<f32>>) -> tensor<16x!pphlo.pub<f32>>
    %148 = "pphlo.constant"() {value = dense<5.000000e-01> : tensor<16xf32>} : () -> tensor<16x!pphlo.pub<f32>>
    %149 = "pphlo.power"(%147, %148) : (tensor<16x!pphlo.pub<f32>>, tensor<16x!pphlo.pub<f32>>) -> tensor<16x!pphlo.pub<f32>>
    %150 = "pphlo.add"(%149, %10) : (tensor<16x!pphlo.pub<f32>>, tensor<16x!pphlo.pub<f32>>) -> tensor<16x!pphlo.pub<f32>>
    %151 = "pphlo.divide"(%142, %150) : (tensor<16x!pphlo.pub<f32>>, tensor<16x!pphlo.pub<f32>>) -> tensor<16x!pphlo.pub<f32>>
    %152 = "pphlo.subtract"(%arg5, %151) : (tensor<16x!pphlo.pub<f32>>, tensor<16x!pphlo.pub<f32>>) -> tensor<16x!pphlo.pub<f32>>
    %153 = "pphlo.transpose"(%30) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<15x16x!pphlo.pub<f32>>) -> tensor<16x15x!pphlo.pub<f32>>
    %154 = "pphlo.dot"(%153, %102) : (tensor<16x15x!pphlo.pub<f32>>, tensor<15x24x!pphlo.pub<f32>>) -> tensor<16x24x!pphlo.pub<f32>>
    %155 = "pphlo.subtract"(%154, %arg24) : (tensor<16x24x!pphlo.pub<f32>>, tensor<16x24x!pphlo.pub<f32>>) -> tensor<16x24x!pphlo.pub<f32>>
    %156 = "pphlo.broadcast"(%109) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pub<f32>>) -> tensor<16x24x!pphlo.pub<f32>>
    %157 = "pphlo.multiply"(%155, %156) : (tensor<16x24x!pphlo.pub<f32>>, tensor<16x24x!pphlo.pub<f32>>) -> tensor<16x24x!pphlo.pub<f32>>
    %158 = "pphlo.add"(%arg24, %157) : (tensor<16x24x!pphlo.pub<f32>>, tensor<16x24x!pphlo.pub<f32>>) -> tensor<16x24x!pphlo.pub<f32>>
    %159 = "pphlo.broadcast"(%122) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pub<f32>>) -> tensor<16x24x!pphlo.pub<f32>>
    %160 = "pphlo.multiply"(%158, %159) : (tensor<16x24x!pphlo.pub<f32>>, tensor<16x24x!pphlo.pub<f32>>) -> tensor<16x24x!pphlo.pub<f32>>
    %161 = "pphlo.multiply"(%154, %154) : (tensor<16x24x!pphlo.pub<f32>>, tensor<16x24x!pphlo.pub<f32>>) -> tensor<16x24x!pphlo.pub<f32>>
    %162 = "pphlo.subtract"(%161, %arg25) : (tensor<16x24x!pphlo.pub<f32>>, tensor<16x24x!pphlo.pub<f32>>) -> tensor<16x24x!pphlo.pub<f32>>
    %163 = "pphlo.broadcast"(%127) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pub<f32>>) -> tensor<16x24x!pphlo.pub<f32>>
    %164 = "pphlo.multiply"(%162, %163) : (tensor<16x24x!pphlo.pub<f32>>, tensor<16x24x!pphlo.pub<f32>>) -> tensor<16x24x!pphlo.pub<f32>>
    %165 = "pphlo.add"(%arg25, %164) : (tensor<16x24x!pphlo.pub<f32>>, tensor<16x24x!pphlo.pub<f32>>) -> tensor<16x24x!pphlo.pub<f32>>
    %166 = "pphlo.constant"() {value = dense<5.000000e-01> : tensor<16x24xf32>} : () -> tensor<16x24x!pphlo.pub<f32>>
    %167 = "pphlo.power"(%165, %166) : (tensor<16x24x!pphlo.pub<f32>>, tensor<16x24x!pphlo.pub<f32>>) -> tensor<16x24x!pphlo.pub<f32>>
    %168 = "pphlo.add"(%167, %9) : (tensor<16x24x!pphlo.pub<f32>>, tensor<16x24x!pphlo.pub<f32>>) -> tensor<16x24x!pphlo.pub<f32>>
    %169 = "pphlo.divide"(%160, %168) : (tensor<16x24x!pphlo.pub<f32>>, tensor<16x24x!pphlo.pub<f32>>) -> tensor<16x24x!pphlo.pub<f32>>
    %170 = "pphlo.subtract"(%arg6, %169) : (tensor<16x24x!pphlo.pub<f32>>, tensor<16x24x!pphlo.pub<f32>>) -> tensor<16x24x!pphlo.pub<f32>>
    %171 = "pphlo.reduce"(%102, %3) ({
    ^bb0(%arg40: tensor<!pphlo.pub<f32>>, %arg41: tensor<!pphlo.pub<f32>>):
      %292 = "pphlo.add"(%arg40, %arg41) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
      "pphlo.return"(%292) : (tensor<!pphlo.pub<f32>>) -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<15x24x!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<24x!pphlo.pub<f32>>
    %172 = "pphlo.subtract"(%171, %arg26) : (tensor<24x!pphlo.pub<f32>>, tensor<24x!pphlo.pub<f32>>) -> tensor<24x!pphlo.pub<f32>>
    %173 = "pphlo.broadcast"(%109) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pub<f32>>) -> tensor<24x!pphlo.pub<f32>>
    %174 = "pphlo.multiply"(%172, %173) : (tensor<24x!pphlo.pub<f32>>, tensor<24x!pphlo.pub<f32>>) -> tensor<24x!pphlo.pub<f32>>
    %175 = "pphlo.add"(%arg26, %174) : (tensor<24x!pphlo.pub<f32>>, tensor<24x!pphlo.pub<f32>>) -> tensor<24x!pphlo.pub<f32>>
    %176 = "pphlo.broadcast"(%122) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pub<f32>>) -> tensor<24x!pphlo.pub<f32>>
    %177 = "pphlo.multiply"(%175, %176) : (tensor<24x!pphlo.pub<f32>>, tensor<24x!pphlo.pub<f32>>) -> tensor<24x!pphlo.pub<f32>>
    %178 = "pphlo.multiply"(%171, %171) : (tensor<24x!pphlo.pub<f32>>, tensor<24x!pphlo.pub<f32>>) -> tensor<24x!pphlo.pub<f32>>
    %179 = "pphlo.subtract"(%178, %arg27) : (tensor<24x!pphlo.pub<f32>>, tensor<24x!pphlo.pub<f32>>) -> tensor<24x!pphlo.pub<f32>>
    %180 = "pphlo.broadcast"(%127) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pub<f32>>) -> tensor<24x!pphlo.pub<f32>>
    %181 = "pphlo.multiply"(%179, %180) : (tensor<24x!pphlo.pub<f32>>, tensor<24x!pphlo.pub<f32>>) -> tensor<24x!pphlo.pub<f32>>
    %182 = "pphlo.add"(%arg27, %181) : (tensor<24x!pphlo.pub<f32>>, tensor<24x!pphlo.pub<f32>>) -> tensor<24x!pphlo.pub<f32>>
    %183 = "pphlo.constant"() {value = dense<5.000000e-01> : tensor<24xf32>} : () -> tensor<24x!pphlo.pub<f32>>
    %184 = "pphlo.power"(%182, %183) : (tensor<24x!pphlo.pub<f32>>, tensor<24x!pphlo.pub<f32>>) -> tensor<24x!pphlo.pub<f32>>
    %185 = "pphlo.add"(%184, %5) : (tensor<24x!pphlo.pub<f32>>, tensor<24x!pphlo.pub<f32>>) -> tensor<24x!pphlo.pub<f32>>
    %186 = "pphlo.divide"(%177, %185) : (tensor<24x!pphlo.pub<f32>>, tensor<24x!pphlo.pub<f32>>) -> tensor<24x!pphlo.pub<f32>>
    %187 = "pphlo.subtract"(%arg7, %186) : (tensor<24x!pphlo.pub<f32>>, tensor<24x!pphlo.pub<f32>>) -> tensor<24x!pphlo.pub<f32>>
    %188 = "pphlo.transpose"(%36) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<15x24x!pphlo.pub<f32>>) -> tensor<24x15x!pphlo.pub<f32>>
    %189 = "pphlo.dot"(%188, %97) : (tensor<24x15x!pphlo.pub<f32>>, tensor<15x20x!pphlo.pub<f32>>) -> tensor<24x20x!pphlo.pub<f32>>
    %190 = "pphlo.subtract"(%189, %arg28) : (tensor<24x20x!pphlo.pub<f32>>, tensor<24x20x!pphlo.pub<f32>>) -> tensor<24x20x!pphlo.pub<f32>>
    %191 = "pphlo.broadcast"(%109) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pub<f32>>) -> tensor<24x20x!pphlo.pub<f32>>
    %192 = "pphlo.multiply"(%190, %191) : (tensor<24x20x!pphlo.pub<f32>>, tensor<24x20x!pphlo.pub<f32>>) -> tensor<24x20x!pphlo.pub<f32>>
    %193 = "pphlo.add"(%arg28, %192) : (tensor<24x20x!pphlo.pub<f32>>, tensor<24x20x!pphlo.pub<f32>>) -> tensor<24x20x!pphlo.pub<f32>>
    %194 = "pphlo.broadcast"(%122) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pub<f32>>) -> tensor<24x20x!pphlo.pub<f32>>
    %195 = "pphlo.multiply"(%193, %194) : (tensor<24x20x!pphlo.pub<f32>>, tensor<24x20x!pphlo.pub<f32>>) -> tensor<24x20x!pphlo.pub<f32>>
    %196 = "pphlo.multiply"(%189, %189) : (tensor<24x20x!pphlo.pub<f32>>, tensor<24x20x!pphlo.pub<f32>>) -> tensor<24x20x!pphlo.pub<f32>>
    %197 = "pphlo.subtract"(%196, %arg29) : (tensor<24x20x!pphlo.pub<f32>>, tensor<24x20x!pphlo.pub<f32>>) -> tensor<24x20x!pphlo.pub<f32>>
    %198 = "pphlo.broadcast"(%127) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pub<f32>>) -> tensor<24x20x!pphlo.pub<f32>>
    %199 = "pphlo.multiply"(%197, %198) : (tensor<24x20x!pphlo.pub<f32>>, tensor<24x20x!pphlo.pub<f32>>) -> tensor<24x20x!pphlo.pub<f32>>
    %200 = "pphlo.add"(%arg29, %199) : (tensor<24x20x!pphlo.pub<f32>>, tensor<24x20x!pphlo.pub<f32>>) -> tensor<24x20x!pphlo.pub<f32>>
    %201 = "pphlo.constant"() {value = dense<5.000000e-01> : tensor<24x20xf32>} : () -> tensor<24x20x!pphlo.pub<f32>>
    %202 = "pphlo.power"(%200, %201) : (tensor<24x20x!pphlo.pub<f32>>, tensor<24x20x!pphlo.pub<f32>>) -> tensor<24x20x!pphlo.pub<f32>>
    %203 = "pphlo.add"(%202, %8) : (tensor<24x20x!pphlo.pub<f32>>, tensor<24x20x!pphlo.pub<f32>>) -> tensor<24x20x!pphlo.pub<f32>>
    %204 = "pphlo.divide"(%195, %203) : (tensor<24x20x!pphlo.pub<f32>>, tensor<24x20x!pphlo.pub<f32>>) -> tensor<24x20x!pphlo.pub<f32>>
    %205 = "pphlo.subtract"(%arg8, %204) : (tensor<24x20x!pphlo.pub<f32>>, tensor<24x20x!pphlo.pub<f32>>) -> tensor<24x20x!pphlo.pub<f32>>
    %206 = "pphlo.reduce"(%97, %3) ({
    ^bb0(%arg40: tensor<!pphlo.pub<f32>>, %arg41: tensor<!pphlo.pub<f32>>):
      %292 = "pphlo.add"(%arg40, %arg41) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
      "pphlo.return"(%292) : (tensor<!pphlo.pub<f32>>) -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<15x20x!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<20x!pphlo.pub<f32>>
    %207 = "pphlo.subtract"(%206, %arg30) : (tensor<20x!pphlo.pub<f32>>, tensor<20x!pphlo.pub<f32>>) -> tensor<20x!pphlo.pub<f32>>
    %208 = "pphlo.broadcast"(%109) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pub<f32>>) -> tensor<20x!pphlo.pub<f32>>
    %209 = "pphlo.multiply"(%207, %208) : (tensor<20x!pphlo.pub<f32>>, tensor<20x!pphlo.pub<f32>>) -> tensor<20x!pphlo.pub<f32>>
    %210 = "pphlo.add"(%arg30, %209) : (tensor<20x!pphlo.pub<f32>>, tensor<20x!pphlo.pub<f32>>) -> tensor<20x!pphlo.pub<f32>>
    %211 = "pphlo.broadcast"(%122) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pub<f32>>) -> tensor<20x!pphlo.pub<f32>>
    %212 = "pphlo.multiply"(%210, %211) : (tensor<20x!pphlo.pub<f32>>, tensor<20x!pphlo.pub<f32>>) -> tensor<20x!pphlo.pub<f32>>
    %213 = "pphlo.multiply"(%206, %206) : (tensor<20x!pphlo.pub<f32>>, tensor<20x!pphlo.pub<f32>>) -> tensor<20x!pphlo.pub<f32>>
    %214 = "pphlo.subtract"(%213, %arg31) : (tensor<20x!pphlo.pub<f32>>, tensor<20x!pphlo.pub<f32>>) -> tensor<20x!pphlo.pub<f32>>
    %215 = "pphlo.broadcast"(%127) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pub<f32>>) -> tensor<20x!pphlo.pub<f32>>
    %216 = "pphlo.multiply"(%214, %215) : (tensor<20x!pphlo.pub<f32>>, tensor<20x!pphlo.pub<f32>>) -> tensor<20x!pphlo.pub<f32>>
    %217 = "pphlo.add"(%arg31, %216) : (tensor<20x!pphlo.pub<f32>>, tensor<20x!pphlo.pub<f32>>) -> tensor<20x!pphlo.pub<f32>>
    %218 = "pphlo.constant"() {value = dense<5.000000e-01> : tensor<20xf32>} : () -> tensor<20x!pphlo.pub<f32>>
    %219 = "pphlo.power"(%217, %218) : (tensor<20x!pphlo.pub<f32>>, tensor<20x!pphlo.pub<f32>>) -> tensor<20x!pphlo.pub<f32>>
    %220 = "pphlo.add"(%219, %7) : (tensor<20x!pphlo.pub<f32>>, tensor<20x!pphlo.pub<f32>>) -> tensor<20x!pphlo.pub<f32>>
    %221 = "pphlo.divide"(%212, %220) : (tensor<20x!pphlo.pub<f32>>, tensor<20x!pphlo.pub<f32>>) -> tensor<20x!pphlo.pub<f32>>
    %222 = "pphlo.subtract"(%arg9, %221) : (tensor<20x!pphlo.pub<f32>>, tensor<20x!pphlo.pub<f32>>) -> tensor<20x!pphlo.pub<f32>>
    %223 = "pphlo.transpose"(%40) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<15x20x!pphlo.pub<f32>>) -> tensor<20x15x!pphlo.pub<f32>>
    %224 = "pphlo.dot"(%223, %94) : (tensor<20x15x!pphlo.pub<f32>>, tensor<15x24x!pphlo.pub<f32>>) -> tensor<20x24x!pphlo.pub<f32>>
    %225 = "pphlo.subtract"(%224, %arg32) : (tensor<20x24x!pphlo.pub<f32>>, tensor<20x24x!pphlo.pub<f32>>) -> tensor<20x24x!pphlo.pub<f32>>
    %226 = "pphlo.broadcast"(%109) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pub<f32>>) -> tensor<20x24x!pphlo.pub<f32>>
    %227 = "pphlo.multiply"(%225, %226) : (tensor<20x24x!pphlo.pub<f32>>, tensor<20x24x!pphlo.pub<f32>>) -> tensor<20x24x!pphlo.pub<f32>>
    %228 = "pphlo.add"(%arg32, %227) : (tensor<20x24x!pphlo.pub<f32>>, tensor<20x24x!pphlo.pub<f32>>) -> tensor<20x24x!pphlo.pub<f32>>
    %229 = "pphlo.broadcast"(%122) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pub<f32>>) -> tensor<20x24x!pphlo.pub<f32>>
    %230 = "pphlo.multiply"(%228, %229) : (tensor<20x24x!pphlo.pub<f32>>, tensor<20x24x!pphlo.pub<f32>>) -> tensor<20x24x!pphlo.pub<f32>>
    %231 = "pphlo.multiply"(%224, %224) : (tensor<20x24x!pphlo.pub<f32>>, tensor<20x24x!pphlo.pub<f32>>) -> tensor<20x24x!pphlo.pub<f32>>
    %232 = "pphlo.subtract"(%231, %arg33) : (tensor<20x24x!pphlo.pub<f32>>, tensor<20x24x!pphlo.pub<f32>>) -> tensor<20x24x!pphlo.pub<f32>>
    %233 = "pphlo.broadcast"(%127) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pub<f32>>) -> tensor<20x24x!pphlo.pub<f32>>
    %234 = "pphlo.multiply"(%232, %233) : (tensor<20x24x!pphlo.pub<f32>>, tensor<20x24x!pphlo.pub<f32>>) -> tensor<20x24x!pphlo.pub<f32>>
    %235 = "pphlo.add"(%arg33, %234) : (tensor<20x24x!pphlo.pub<f32>>, tensor<20x24x!pphlo.pub<f32>>) -> tensor<20x24x!pphlo.pub<f32>>
    %236 = "pphlo.constant"() {value = dense<5.000000e-01> : tensor<20x24xf32>} : () -> tensor<20x24x!pphlo.pub<f32>>
    %237 = "pphlo.power"(%235, %236) : (tensor<20x24x!pphlo.pub<f32>>, tensor<20x24x!pphlo.pub<f32>>) -> tensor<20x24x!pphlo.pub<f32>>
    %238 = "pphlo.add"(%237, %6) : (tensor<20x24x!pphlo.pub<f32>>, tensor<20x24x!pphlo.pub<f32>>) -> tensor<20x24x!pphlo.pub<f32>>
    %239 = "pphlo.divide"(%230, %238) : (tensor<20x24x!pphlo.pub<f32>>, tensor<20x24x!pphlo.pub<f32>>) -> tensor<20x24x!pphlo.pub<f32>>
    %240 = "pphlo.subtract"(%arg10, %239) : (tensor<20x24x!pphlo.pub<f32>>, tensor<20x24x!pphlo.pub<f32>>) -> tensor<20x24x!pphlo.pub<f32>>
    %241 = "pphlo.reduce"(%94, %3) ({
    ^bb0(%arg40: tensor<!pphlo.pub<f32>>, %arg41: tensor<!pphlo.pub<f32>>):
      %292 = "pphlo.add"(%arg40, %arg41) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
      "pphlo.return"(%292) : (tensor<!pphlo.pub<f32>>) -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<15x24x!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<24x!pphlo.pub<f32>>
    %242 = "pphlo.subtract"(%241, %arg34) : (tensor<24x!pphlo.pub<f32>>, tensor<24x!pphlo.pub<f32>>) -> tensor<24x!pphlo.pub<f32>>
    %243 = "pphlo.multiply"(%242, %173) : (tensor<24x!pphlo.pub<f32>>, tensor<24x!pphlo.pub<f32>>) -> tensor<24x!pphlo.pub<f32>>
    %244 = "pphlo.add"(%arg34, %243) : (tensor<24x!pphlo.pub<f32>>, tensor<24x!pphlo.pub<f32>>) -> tensor<24x!pphlo.pub<f32>>
    %245 = "pphlo.multiply"(%244, %176) : (tensor<24x!pphlo.pub<f32>>, tensor<24x!pphlo.pub<f32>>) -> tensor<24x!pphlo.pub<f32>>
    %246 = "pphlo.multiply"(%241, %241) : (tensor<24x!pphlo.pub<f32>>, tensor<24x!pphlo.pub<f32>>) -> tensor<24x!pphlo.pub<f32>>
    %247 = "pphlo.subtract"(%246, %arg35) : (tensor<24x!pphlo.pub<f32>>, tensor<24x!pphlo.pub<f32>>) -> tensor<24x!pphlo.pub<f32>>
    %248 = "pphlo.multiply"(%247, %180) : (tensor<24x!pphlo.pub<f32>>, tensor<24x!pphlo.pub<f32>>) -> tensor<24x!pphlo.pub<f32>>
    %249 = "pphlo.add"(%arg35, %248) : (tensor<24x!pphlo.pub<f32>>, tensor<24x!pphlo.pub<f32>>) -> tensor<24x!pphlo.pub<f32>>
    %250 = "pphlo.power"(%249, %183) : (tensor<24x!pphlo.pub<f32>>, tensor<24x!pphlo.pub<f32>>) -> tensor<24x!pphlo.pub<f32>>
    %251 = "pphlo.add"(%250, %5) : (tensor<24x!pphlo.pub<f32>>, tensor<24x!pphlo.pub<f32>>) -> tensor<24x!pphlo.pub<f32>>
    %252 = "pphlo.divide"(%245, %251) : (tensor<24x!pphlo.pub<f32>>, tensor<24x!pphlo.pub<f32>>) -> tensor<24x!pphlo.pub<f32>>
    %253 = "pphlo.subtract"(%arg11, %252) : (tensor<24x!pphlo.pub<f32>>, tensor<24x!pphlo.pub<f32>>) -> tensor<24x!pphlo.pub<f32>>
    %254 = "pphlo.transpose"(%44) {permutation = dense<[1, 0]> : tensor<2xi64>, xla_shape = "f32[24,15]{0,1}"} : (tensor<15x24x!pphlo.pub<f32>>) -> tensor<24x15x!pphlo.pub<f32>>
    %255 = "pphlo.broadcast"(%91) {broadcast_dimensions = dense<1> : tensor<1xi64>, xla_shape = "f32[24,15]{0,1}"} : (tensor<15x!pphlo.pub<f32>>) -> tensor<24x15x!pphlo.pub<f32>>
    %256 = "pphlo.multiply"(%254, %255) {xla_shape = "f32[24,15]{0,1}"} : (tensor<24x15x!pphlo.pub<f32>>, tensor<24x15x!pphlo.pub<f32>>) -> tensor<24x15x!pphlo.pub<f32>>
    %257 = "pphlo.reduce"(%256, %3) ({
    ^bb0(%arg40: tensor<!pphlo.pub<f32>>, %arg41: tensor<!pphlo.pub<f32>>):
      %292 = "pphlo.add"(%arg40, %arg41) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
      "pphlo.return"(%292) : (tensor<!pphlo.pub<f32>>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<24x15x!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<24x!pphlo.pub<f32>>
    %258 = "pphlo.reshape"(%257) : (tensor<24x!pphlo.pub<f32>>) -> tensor<24x1x!pphlo.pub<f32>>
    %259 = "pphlo.subtract"(%258, %arg36) : (tensor<24x1x!pphlo.pub<f32>>, tensor<24x1x!pphlo.pub<f32>>) -> tensor<24x1x!pphlo.pub<f32>>
    %260 = "pphlo.broadcast"(%109) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pub<f32>>) -> tensor<24x1x!pphlo.pub<f32>>
    %261 = "pphlo.multiply"(%259, %260) : (tensor<24x1x!pphlo.pub<f32>>, tensor<24x1x!pphlo.pub<f32>>) -> tensor<24x1x!pphlo.pub<f32>>
    %262 = "pphlo.add"(%arg36, %261) : (tensor<24x1x!pphlo.pub<f32>>, tensor<24x1x!pphlo.pub<f32>>) -> tensor<24x1x!pphlo.pub<f32>>
    %263 = "pphlo.broadcast"(%122) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pub<f32>>) -> tensor<24x1x!pphlo.pub<f32>>
    %264 = "pphlo.multiply"(%262, %263) : (tensor<24x1x!pphlo.pub<f32>>, tensor<24x1x!pphlo.pub<f32>>) -> tensor<24x1x!pphlo.pub<f32>>
    %265 = "pphlo.multiply"(%258, %258) : (tensor<24x1x!pphlo.pub<f32>>, tensor<24x1x!pphlo.pub<f32>>) -> tensor<24x1x!pphlo.pub<f32>>
    %266 = "pphlo.subtract"(%265, %arg37) : (tensor<24x1x!pphlo.pub<f32>>, tensor<24x1x!pphlo.pub<f32>>) -> tensor<24x1x!pphlo.pub<f32>>
    %267 = "pphlo.broadcast"(%127) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pub<f32>>) -> tensor<24x1x!pphlo.pub<f32>>
    %268 = "pphlo.multiply"(%266, %267) : (tensor<24x1x!pphlo.pub<f32>>, tensor<24x1x!pphlo.pub<f32>>) -> tensor<24x1x!pphlo.pub<f32>>
    %269 = "pphlo.add"(%arg37, %268) : (tensor<24x1x!pphlo.pub<f32>>, tensor<24x1x!pphlo.pub<f32>>) -> tensor<24x1x!pphlo.pub<f32>>
    %270 = "pphlo.constant"() {value = dense<5.000000e-01> : tensor<24x1xf32>} : () -> tensor<24x1x!pphlo.pub<f32>>
    %271 = "pphlo.power"(%269, %270) : (tensor<24x1x!pphlo.pub<f32>>, tensor<24x1x!pphlo.pub<f32>>) -> tensor<24x1x!pphlo.pub<f32>>
    %272 = "pphlo.add"(%271, %4) : (tensor<24x1x!pphlo.pub<f32>>, tensor<24x1x!pphlo.pub<f32>>) -> tensor<24x1x!pphlo.pub<f32>>
    %273 = "pphlo.divide"(%264, %272) : (tensor<24x1x!pphlo.pub<f32>>, tensor<24x1x!pphlo.pub<f32>>) -> tensor<24x1x!pphlo.pub<f32>>
    %274 = "pphlo.subtract"(%arg12, %273) : (tensor<24x1x!pphlo.pub<f32>>, tensor<24x1x!pphlo.pub<f32>>) -> tensor<24x1x!pphlo.pub<f32>>
    %275 = "pphlo.reduce"(%90, %3) ({
    ^bb0(%arg40: tensor<!pphlo.pub<f32>>, %arg41: tensor<!pphlo.pub<f32>>):
      %292 = "pphlo.add"(%arg40, %arg41) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
      "pphlo.return"(%292) : (tensor<!pphlo.pub<f32>>) -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<15x1x!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<1x!pphlo.pub<f32>>
    %276 = "pphlo.subtract"(%275, %arg38) : (tensor<1x!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>) -> tensor<1x!pphlo.pub<f32>>
    %277 = "pphlo.reshape"(%109) : (tensor<!pphlo.pub<f32>>) -> tensor<1x!pphlo.pub<f32>>
    %278 = "pphlo.multiply"(%276, %277) : (tensor<1x!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>) -> tensor<1x!pphlo.pub<f32>>
    %279 = "pphlo.add"(%arg38, %278) : (tensor<1x!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>) -> tensor<1x!pphlo.pub<f32>>
    %280 = "pphlo.reshape"(%122) : (tensor<!pphlo.pub<f32>>) -> tensor<1x!pphlo.pub<f32>>
    %281 = "pphlo.multiply"(%279, %280) : (tensor<1x!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>) -> tensor<1x!pphlo.pub<f32>>
    %282 = "pphlo.multiply"(%275, %275) : (tensor<1x!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>) -> tensor<1x!pphlo.pub<f32>>
    %283 = "pphlo.subtract"(%282, %arg39) : (tensor<1x!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>) -> tensor<1x!pphlo.pub<f32>>
    %284 = "pphlo.reshape"(%127) : (tensor<!pphlo.pub<f32>>) -> tensor<1x!pphlo.pub<f32>>
    %285 = "pphlo.multiply"(%283, %284) : (tensor<1x!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>) -> tensor<1x!pphlo.pub<f32>>
    %286 = "pphlo.add"(%arg39, %285) : (tensor<1x!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>) -> tensor<1x!pphlo.pub<f32>>
    %287 = "pphlo.constant"() {value = dense<5.000000e-01> : tensor<1xf32>} : () -> tensor<1x!pphlo.pub<f32>>
    %288 = "pphlo.power"(%286, %287) : (tensor<1x!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>) -> tensor<1x!pphlo.pub<f32>>
    %289 = "pphlo.add"(%288, %1) : (tensor<1x!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>) -> tensor<1x!pphlo.pub<f32>>
    %290 = "pphlo.divide"(%281, %289) : (tensor<1x!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>) -> tensor<1x!pphlo.pub<f32>>
    %291 = "pphlo.subtract"(%arg13, %290) : (tensor<1x!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>) -> tensor<1x!pphlo.pub<f32>>
    return %71, %135, %152, %170, %187, %205, %222, %240, %253, %274, %291, %69, %21, %113, %112, %130, %140, %147, %158, %165, %175, %182, %193, %200, %210, %217, %228, %235, %244, %249, %262, %269, %279, %286 : tensor<!pphlo.pub<f32>>, tensor<29x16x!pphlo.pub<f32>>, tensor<16x!pphlo.pub<f32>>, tensor<16x24x!pphlo.pub<f32>>, tensor<24x!pphlo.pub<f32>>, tensor<24x20x!pphlo.pub<f32>>, tensor<20x!pphlo.pub<f32>>, tensor<20x24x!pphlo.pub<f32>>, tensor<24x!pphlo.pub<f32>>, tensor<24x1x!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<i64>>, tensor<29x16x!pphlo.pub<f32>>, tensor<29x16x!pphlo.pub<f32>>, tensor<16x!pphlo.pub<f32>>, tensor<16x!pphlo.pub<f32>>, tensor<16x24x!pphlo.pub<f32>>, tensor<16x24x!pphlo.pub<f32>>, tensor<24x!pphlo.pub<f32>>, tensor<24x!pphlo.pub<f32>>, tensor<24x20x!pphlo.pub<f32>>, tensor<24x20x!pphlo.pub<f32>>, tensor<20x!pphlo.pub<f32>>, tensor<20x!pphlo.pub<f32>>, tensor<20x24x!pphlo.pub<f32>>, tensor<20x24x!pphlo.pub<f32>>, tensor<24x!pphlo.pub<f32>>, tensor<24x!pphlo.pub<f32>>, tensor<24x1x!pphlo.pub<f32>>, tensor<24x1x!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>, tensor<1x!pphlo.pub<f32>>
  }
}
