// RUN: spu-translate --interpret -split-input-file %s

func.func @main() {
    %0 = pphlo.constant dense<[[[[ 1.0, 2.0, 3.0, 4.0],
                                 [ 5.0, 6.0, 7.0, 8.0],
                                 [ 9.0,10.0,11.0,12.0],
                                 [13.0,14.0,15.0,16.0]]]]> : tensor<1x1x4x4xf32>
    %1 = pphlo.constant dense<[[[[5.0,6.0],
                                 [7.0,8.0]]]]> : tensor<1x1x2x2xf32>
    %2 = pphlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = pphlo.pad %0, %2, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 0, 0] : (tensor<1x1x4x4xf32>, tensor<f32>) -> tensor<1x1x5x5xf32>
    %4 = pphlo.convolution(%3, %1)
            dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
            window = {stride = [1, 1]} : (tensor<1x1x5x5xf32>, tensor<1x1x2x2xf32>) -> tensor<1x1x4x4xf32>
    %expected = pphlo.constant dense<[[[[100.0, 126.0, 152.0, 76.0 ],
                                        [204.0, 230.0, 256.0, 124.0],
                                        [308.0, 334.0, 360.0, 172.0],
                                        [149.0, 160.0, 171.0, 80.0 ]]]]> : tensor<1x1x4x4xf32>
    pphlo.custom_call @expect_almost_eq (%4, %expected) : (tensor<1x1x4x4xf32>, tensor<1x1x4x4xf32>) -> ()
    return
  }

// -----

func.func @main() {
  %0 = pphlo.constant dense<[[[[ 1.0,  2.0,  3.0,  4.0]], [[ 5.0,  6.0,  7.0,  8.0]], [[ 9.0, 10.0, 11.0, 12.0]]],
                             [[[13.0, 14.0, 15.0, 16.0]], [[17.0, 18.0, 19.0, 20.0]], [[21.0, 22.0, 23.0, 24.0]]]]> : tensor<2x3x1x4xf32>
  %1 = pphlo.constant dense<[[[[1.0, 7.0, 13.0], [4.0, 10.0, 16.0]],
                              [[2.0, 8.0, 14.0], [5.0, 11.0, 17.0]],
                              [[3.0, 9.0, 15.0], [6.0, 12.0, 18.0]]]]> : tensor<1x3x2x3xf32>
  %2 = pphlo.convolution(%0, %1)
        dim_numbers = [f, 0, b, 1]x[o, 1, i, 0]->[f, 0, b, 1],
        window = {stride = [1, 1]} : (tensor<2x3x1x4xf32>, tensor<1x3x2x3xf32>) -> tensor<1x1x1x2xf32>
  %expected = pphlo.constant dense<[[[[2514.0, 2685.0]]]]> : tensor<1x1x1x2xf32>
  pphlo.custom_call @expect_almost_eq (%2, %expected) : (tensor<1x1x1x2xf32>, tensor<1x1x1x2xf32>) -> ()
  return
}

// -----

func.func @main() {
  %0 = pphlo.constant dense<[[[[ 1.0,  2.0,  3.0,  4.0],
                               [ 5.0,  6.0,  7.0,  8.0],
                               [ 9.0, 10.0, 11.0, 12.0],
                               [13.0, 14.0, 15.0, 16.0]]]]>: tensor<1x1x4x4xf32>
  %1 = pphlo.constant dense<[[[[5.0, 6.0],
                               [7.0, 8.0]]]]>: tensor<1x1x2x2xf32>
  %2 = pphlo.constant dense<0.000000e+00> : tensor<f32>
  %3 = pphlo.pad %0, %2, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<1x1x4x4xf32>, tensor<f32>) -> tensor<1x1x8x8xf32>
  %4 = pphlo.convolution(%3, %1)
              dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
              window = {stride = [1, 1]} : (tensor<1x1x8x8xf32>, tensor<1x1x2x2xf32>) -> tensor<1x1x7x7xf32>
  %expected = pphlo.constant dense<[[[[ 5.0,  12.0, 10.0,  18.0,  15.0,  24.0,  20.0],
                                      [35.0,  48.0, 42.0,  56.0,  49.0,  64.0,  56.0],
                                      [25.0,  36.0, 30.0,  42.0,  35.0,  48.0,  40.0],
                                      [63.0,  80.0, 70.0,  88.0,  77.0,  96.0,  84.0],
                                      [45.0,  60.0, 50.0,  66.0,  55.0,  72.0,  60.0],
                                      [91.0, 112.0, 98.0, 120.0, 105.0, 128.0, 112.0],
                                      [65.0,  84.0, 70.0,  90.0,  75.0,  96.0,  80.0]]]]> : tensor<1x1x7x7xf32>
  pphlo.custom_call @expect_almost_eq (%4, %expected) : (tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>) -> ()
  return
}


// -----

func.func @main() {
  %0 = pphlo.constant dense<[[[[ 1.0,  2.0,  3.0,  4.0],
                                [ 5.0,  6.0,  7.0,  8.0],
                                [ 9.0, 10.0, 11.0, 12.0],
                                [13.0, 14.0, 15.0, 16.0]]]]>: tensor<1x1x4x4xf32>
  %1 = pphlo.constant dense<[[[[5.0, 6.0],
                                [7.0, 8.0]]]]>: tensor<1x1x2x2xf32>
  %2 = pphlo.constant dense<0.000000e+00> : tensor<f32>
  %3 = pphlo.pad %0, %2, low = [0, 0, 1, 1], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<1x1x4x4xf32>, tensor<f32>) -> tensor<1x1x9x9xf32>
  %4 = pphlo.convolution(%3, %1)
        dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
        window = {stride = [1, 1]} : (tensor<1x1x9x9xf32>, tensor<1x1x2x2xf32>) -> tensor<1x1x8x8xf32>
  %expected = pphlo.constant dense<[[[[  8.0,  7.0,  16.0, 14.0,  24.0,  21.0,  32.0,  28.0],
                                      [  6.0,  5.0,  12.0, 10.0,  18.0,  15.0,  24.0,  20.0],
                                      [ 40.0, 35.0,  48.0, 42.0,  56.0,  49.0,  64.0,  56.0],
                                      [ 30.0, 25.0,  36.0, 30.0,  42.0,  35.0,  48.0,  40.0],
                                      [ 72.0, 63.0,  80.0, 70.0,  88.0,  77.0,  96.0,  84.0],
                                      [ 54.0, 45.0,  60.0, 50.0,  66.0,  55.0,  72.0,  60.0],
                                      [104.0, 91.0, 112.0, 98.0, 120.0, 105.0, 128.0, 112.0],
                                      [ 78.0, 65.0,  84.0, 70.0,  90.0,  75.0,  96.0,  80.0]]]]> : tensor<1x1x8x8xf32>
  pphlo.custom_call @expect_almost_eq (%4, %expected) : (tensor<1x1x8x8xf32>, tensor<1x1x8x8xf32>) -> ()
  return
}

// -----

func.func @main() {
  %0 = pphlo.constant dense<[[[[ 0.0,  1.0,  2.0,  3.0,  4.0,  5.0],
                               [ 6.0,  7.0,  8.0,  9.0, 10.0, 11.0],
                               [12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
                               [18.0, 19.0, 20.0, 21.0, 22.0, 23.0]]]]> : tensor<1x1x4x6xf32>
  %1 = pphlo.constant dense<[[[[1.0, 10.0, 100.0],
                               [2.0, 20.0, 200.0]]]]> : tensor<1x1x2x3xf32>
  %2 = pphlo.constant dense<0.000000e+00> : tensor<f32>
  %3 = pphlo.pad %1, %2, low = [0, 0, 0, 0], high = [0, 0, 0, 0], interior = [0, 0, 1, 1] : (tensor<1x1x2x3xf32>, tensor<f32>) -> tensor<1x1x3x5xf32>
  %4 = pphlo.convolution(%0, %3)
        dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
        window = {stride = [1, 1]} : (tensor<1x1x4x6xf32>, tensor<1x1x3x5xf32>) -> tensor<1x1x2x2xf32>
  %expected = pphlo.constant dense<[[[[3924.0, 4257.0], [5922.0, 6255.0]]]]> : tensor<1x1x2x2xf32>
  pphlo.custom_call @expect_almost_eq (%4, %expected) : (tensor<1x1x2x2xf32>, tensor<1x1x2x2xf32>) -> ()
  return
}
