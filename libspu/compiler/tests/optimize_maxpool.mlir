// RUN: mlir-pphlo-opt --optimize-maxpool --split-input-file %s | FileCheck %s

func.func @main(%arg0: tensor<129x24x24x16x!pphlo.sec<f32>>, %arg1: tensor<129x23x23x16x!pphlo.sec<f32>>) -> (tensor<129x23x23x16x!pphlo.sec<f32>>, tensor<129x24x24x16x!pphlo.sec<f32>>) {
    %0 = "pphlo.constant"() {value = dense<0xFF800000> : tensor<f32>} : () -> tensor<!pphlo.pub<f32>>
    %1 = "pphlo.constant"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<!pphlo.pub<f32>>
    %2 = "pphlo.convert"(%0) : (tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.sec<f32>>
    %3 = "pphlo.convert"(%1) : (tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.sec<f32>>
    //CHECK: "pphlo.argmax"(%arg0) {base_dilations = dense<1> : tensor<4xi64>, onehot_index = true, padding = dense<0> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 2, 2, 1]> : tensor<4xi64>, window_strides = dense<1> : tensor<4xi64>} : (tensor<129x24x24x16x!pphlo.sec<f32>>) -> (tensor<129x23x23x16x!pphlo.sec<f32>>, tensor<129x23x23x16x4x!pphlo.sec<i1>>)
    %4 = "pphlo.reduce_window"(%arg0, %2) ({
    ^bb0(%arg2: tensor<!pphlo.sec<f32>>, %arg3: tensor<!pphlo.sec<f32>>):
      %6 = "pphlo.maximum"(%arg2, %arg3) : (tensor<!pphlo.sec<f32>>, tensor<!pphlo.sec<f32>>) -> tensor<!pphlo.sec<f32>>
      "pphlo.return"(%6) : (tensor<!pphlo.sec<f32>>) -> ()
    }) {base_dilations = dense<1> : tensor<4xi64>, padding = dense<0> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 2, 2, 1]> : tensor<4xi64>, window_strides = dense<1> : tensor<4xi64>} : (tensor<129x24x24x16x!pphlo.sec<f32>>, tensor<!pphlo.sec<f32>>) -> tensor<129x23x23x16x!pphlo.sec<f32>>
    //CHECK-NOT: pphlo.select_and_scatter
    //CHECK : pphlo.maxpool_scatter
    %5 = "pphlo.select_and_scatter"(%arg0, %arg1, %3) ({
    ^bb0(%arg2: tensor<!pphlo.sec<f32>>, %arg3: tensor<!pphlo.sec<f32>>):
      %6 = "pphlo.greater_equal"(%arg2, %arg3) : (tensor<!pphlo.sec<f32>>, tensor<!pphlo.sec<f32>>) -> tensor<!pphlo.sec<i1>>
      "pphlo.return"(%6) : (tensor<!pphlo.sec<i1>>) -> ()
    }, {
    ^bb0(%arg2: tensor<!pphlo.sec<f32>>, %arg3: tensor<!pphlo.sec<f32>>):
      %6 = "pphlo.add"(%arg2, %arg3) : (tensor<!pphlo.sec<f32>>, tensor<!pphlo.sec<f32>>) -> tensor<!pphlo.sec<f32>>
      "pphlo.return"(%6) : (tensor<!pphlo.sec<f32>>) -> ()
    }) {padding = dense<0> : tensor<4x2xi64>, window_dimensions = dense<[1, 2, 2, 1]> : tensor<4xi64>, window_strides = dense<1> : tensor<4xi64>} : (tensor<129x24x24x16x!pphlo.sec<f32>>, tensor<129x23x23x16x!pphlo.sec<f32>>, tensor<!pphlo.sec<f32>>) -> tensor<129x24x24x16x!pphlo.sec<f32>>

    return %4, %5 : tensor<129x23x23x16x!pphlo.sec<f32>>, tensor<129x24x24x16x!pphlo.sec<f32>>
}

// -----

func.func @main(%arg0: tensor<128x2x2x256x!pphlo.sec<f32>>, %arg1: tensor<128x1x1x256x!pphlo.sec<f32>>) -> (tensor<128x2x2x256x!pphlo.sec<f32>>, tensor<128x2x2x256x!pphlo.sec<f32>>) {
    %0 = "pphlo.constant"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<128x2x2x256x!pphlo.pub<f32>>
    %1 = "pphlo.constant"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<!pphlo.pub<f32>>
    %2 = "pphlo.convert"(%1) : (tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.sec<f32>>
    %3 = "pphlo.maximum"(%arg0, %0) : (tensor<128x2x2x256x!pphlo.sec<f32>>, tensor<128x2x2x256x!pphlo.pub<f32>>) -> tensor<128x2x2x256x!pphlo.sec<f32>>
    // CHECK: pphlo.select_and_scatter
    %4 = "pphlo.select_and_scatter"(%arg0, %arg1, %2) ({
    ^bb0(%arg2: tensor<!pphlo.sec<f32>>, %arg3: tensor<!pphlo.sec<f32>>):
      %5 = "pphlo.greater_equal"(%arg2, %arg3) : (tensor<!pphlo.sec<f32>>, tensor<!pphlo.sec<f32>>) -> tensor<!pphlo.sec<i1>>
      "pphlo.return"(%5) : (tensor<!pphlo.sec<i1>>) -> ()
    }, {
    ^bb0(%arg2: tensor<!pphlo.sec<f32>>, %arg3: tensor<!pphlo.sec<f32>>):
      %5 = "pphlo.add"(%arg2, %arg3) : (tensor<!pphlo.sec<f32>>, tensor<!pphlo.sec<f32>>) -> tensor<!pphlo.sec<f32>>
      "pphlo.return"(%5) : (tensor<!pphlo.sec<f32>>) -> ()
    }) {padding = dense<0> : tensor<4x2xi64>, window_dimensions = dense<[1, 2, 2, 1]> : tensor<4xi64>, window_strides = dense<1> : tensor<4xi64>} : (tensor<128x2x2x256x!pphlo.sec<f32>>, tensor<128x1x1x256x!pphlo.sec<f32>>, tensor<!pphlo.sec<f32>>) -> tensor<128x2x2x256x!pphlo.sec<f32>>

    return %3, %4 : tensor<128x2x2x256x!pphlo.sec<f32>>, tensor<128x2x2x256x!pphlo.sec<f32>>
}
