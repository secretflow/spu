// RUN: spu-translate --protocol_kind=1 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=2 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=3 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=4 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=5 --interpret -split-input-file %s

func.func @while() {
  // int i = 0;
  // int sum = 0;
  // while (i < 10) {
  //   sum += 1;
  //   i += 1;
  // }
  %init_i = pphlo.constant dense<0> : tensor<i64>
  %init_sum = pphlo.constant dense<0> : tensor<i64>
  %one = pphlo.constant dense<1> : tensor<i64>
  %ten = pphlo.constant dense<10> : tensor<i64>
  %results0, %results1 = pphlo.while(%arg0 = %init_i, %arg1 = %init_sum) : tensor<i64>, tensor<i64>
  cond {
    %cond = pphlo.less %arg0, %ten : (tensor<i64>, tensor<i64>) -> tensor<i1>
    pphlo.return %cond : tensor<i1>
  } do {
    %new_sum = pphlo.add %arg1, %one : tensor<i64>
    %new_i = pphlo.add %arg0, %one : tensor<i64>
    pphlo.return %new_i, %new_sum : tensor<i64>, tensor<i64>
  }
  %expected = pphlo.constant dense<10> : tensor<i64>
  pphlo.custom_call @expect_eq (%results0, %expected) : (tensor<i64>,tensor<i64>)->()
  pphlo.custom_call @expect_eq (%results1, %expected) : (tensor<i64>,tensor<i64>)->()
  func.return
}
