// RUN: pphlo-opt --hlo-legalize-to-pphlo=input_vis_list=VIS_SECRET --lower-conversion-cast --split-input-file %s | FileCheck %s

func.func @main(%arg0: tensor<2x2xi32>) -> (tensor<2x2xf32>) {
    // CHECK: pphlo.convert %arg0 : (tensor<2x2x!pphlo.secret<i32>>) -> tensor<2x2x!pphlo.secret<f32>>
    %0 = "stablehlo.convert"(%arg0) : (tensor<2x2xi32>) -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
}

// -----

func.func @main(%arg0: tensor<2x2xf32>) -> (tensor<2x2xf32>) {
    // CHECK: pphlo.sqrt %arg0 : tensor<2x2x!pphlo.secret<f32>>
    %0 = "stablehlo.sqrt"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
}

// -----

func.func @main(%arg0: tensor<2x2xf32>) -> (tensor<2x2xf32>) {
    // CHECK: pphlo.negate %arg0 : tensor<2x2x!pphlo.secret<f32>>
    %0 = "stablehlo.negate"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
}

// -----

func.func @main(%arg0: tensor<2x2xf32>) -> (tensor<2x2xf32>) {
    // CHECK: pphlo.exponential %arg0 : tensor<2x2x!pphlo.secret<f32>>
    %0 = "stablehlo.exponential"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
}

// -----

func.func @main(%arg0: tensor<2x2xf32>) -> (tensor<2x2xf32>) {
    // CHECK: pphlo.log_plus_one %arg0 : tensor<2x2x!pphlo.secret<f32>>
    %0 = "stablehlo.log_plus_one"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
}

// -----

func.func @main(%arg0: tensor<2x2xf32>) -> (tensor<2x2xf32>) {
    // CHECK: pphlo.floor %arg0 : tensor<2x2x!pphlo.secret<f32>>
    %0 = "stablehlo.floor"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
}

// -----

func.func @main(%arg0: tensor<2x2xf32>) -> (tensor<2x2xf32>) {
    // CHECK: pphlo.ceil %arg0 : tensor<2x2x!pphlo.secret<f32>>
    %0 = "stablehlo.ceil"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
}

// -----

func.func @main(%arg0: tensor<2x2xf32>) -> (tensor<2x2xf32>) {
    // CHECK: pphlo.abs %arg0 : tensor<2x2x!pphlo.secret<f32>>
    %0 = "stablehlo.abs"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
}

// -----

func.func @main(%arg0: tensor<2x2xf32>) -> (tensor<2x2xf32>) {
    // CHECK: pphlo.logistic %arg0 : tensor<2x2x!pphlo.secret<f32>>
    %0 = "stablehlo.logistic"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
}

// -----

func.func @main(%arg0: tensor<2x2xi32>) -> (tensor<2x2xi32>) {
    // CHECK: pphlo.not %arg0 : tensor<2x2x!pphlo.secret<i32>>
    %0 = "stablehlo.not"(%arg0) : (tensor<2x2xi32>) -> tensor<2x2xi32>
    return %0 : tensor<2x2xi32>
}

// -----

func.func @main(%arg0: tensor<2x2xi32>) -> (tensor<2x2xf32>) {
    // CHECK: %0 = pphlo.bitcast_convert %arg0 : (tensor<2x2x!pphlo.secret<i32>>) -> tensor<2x2x!pphlo.secret<f32>>
    %0 = "stablehlo.bitcast_convert"(%arg0) : (tensor<2x2xi32>) -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
}
