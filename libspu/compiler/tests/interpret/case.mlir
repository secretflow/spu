// RUN: spu-translate --protocol_kind=1 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=2 --interpret -split-input-file %s
// RUN: spu-translate --protocol_kind=3 --interpret -split-input-file %s

func.func @case_negative_index_default() {
  %index = pphlo.constant dense<-1> : tensor<i32>
  %result_branch0 = pphlo.constant dense<0> : tensor<2xi64>
  %result_branch1 = pphlo.constant dense<1> : tensor<2xi64>
  %result0, %result1 = "pphlo.case"(%index) ({
    pphlo.return %result_branch0, %result_branch0 : tensor<2xi64>, tensor<2xi64>
  }, {
    pphlo.return %result_branch1, %result_branch1 : tensor<2xi64>, tensor<2xi64>
  }) : (tensor<i32>) -> (tensor<2xi64>, tensor<2xi64>)
  %expected = pphlo.constant dense<[1, 1]> : tensor<2xi64>
  pphlo.custom_call @expect_eq(%result0, %expected) : (tensor<2xi64>,tensor<2xi64>)->()
  pphlo.custom_call @expect_eq(%result1, %expected) : (tensor<2xi64>,tensor<2xi64>)->()
  func.return
}

// -----

func.func @case_in_bound_index() {
  %index = pphlo.constant dense<0> : tensor<i32>
  %result_branch0 = pphlo.constant dense<0> : tensor<2xi64>
  %result_branch1 = pphlo.constant dense<1> : tensor<2xi64>
  %result0, %result1 = "pphlo.case"(%index) ({
    pphlo.return %result_branch0, %result_branch0 : tensor<2xi64>, tensor<2xi64>
  }, {
    pphlo.return %result_branch1, %result_branch1 : tensor<2xi64>, tensor<2xi64>
  }) : (tensor<i32>) -> (tensor<2xi64>, tensor<2xi64>)
  %expected = pphlo.constant dense<[0, 0]> : tensor<2xi64>
  pphlo.custom_call @expect_eq(%result0, %expected) : (tensor<2xi64>,tensor<2xi64>)->()
  pphlo.custom_call @expect_eq(%result1, %expected) : (tensor<2xi64>,tensor<2xi64>)->()
  func.return
}

// -----

func.func @case_out_of_bound_index_default() {
  %index = pphlo.constant dense<2> : tensor<i32>
  %result_branch0 = pphlo.constant dense<0> : tensor<2xi64>
  %result_branch1 = pphlo.constant dense<1> : tensor<2xi64>
  %result0, %result1 = "pphlo.case"(%index) ({
    pphlo.return %result_branch0, %result_branch0 : tensor<2xi64>, tensor<2xi64>
  }, {
    pphlo.return %result_branch1, %result_branch1 : tensor<2xi64>, tensor<2xi64>
  }) : (tensor<i32>) -> (tensor<2xi64>, tensor<2xi64>)
  %expected = pphlo.constant dense<[1, 1]> : tensor<2xi64>
  pphlo.custom_call @expect_eq(%result0, %expected) : (tensor<2xi64>,tensor<2xi64>)->()
  pphlo.custom_call @expect_eq(%result1, %expected) : (tensor<2xi64>,tensor<2xi64>)->()
  func.return
}

// -----

func.func @case_out_of_bound_index_default() {
  %0 = pphlo.constant dense<2> : tensor<i32>
  %index = pphlo.convert %0 : (tensor<i32>) -> tensor<!pphlo.secret<i32>>
  %1 = pphlo.constant dense<0> : tensor<2xi64>
  %2 = pphlo.constant dense<1> : tensor<2xi64>
  %result_branch0 = pphlo.convert %1 : (tensor<2xi64>) -> tensor<2x!pphlo.secret<i64>>
  %result_branch1 = pphlo.convert %2 : (tensor<2xi64>) -> tensor<2x!pphlo.secret<i64>>
  %result0, %result1 = "pphlo.case"(%index) ({
    pphlo.return %result_branch0, %result_branch0 : tensor<2x!pphlo.secret<i64>>, tensor<2x!pphlo.secret<i64>>
  }, {
    pphlo.return %result_branch1, %result_branch1 : tensor<2x!pphlo.secret<i64>>, tensor<2x!pphlo.secret<i64>>
  }) : (tensor<!pphlo.secret<i32>>) -> (tensor<2x!pphlo.secret<i64>>, tensor<2x!pphlo.secret<i64>>)
  %expected = pphlo.constant dense<[1, 1]> : tensor<2xi64>
  pphlo.custom_call @expect_eq(%result0, %expected) : (tensor<2x!pphlo.secret<i64>>,tensor<2xi64>)->()
  pphlo.custom_call @expect_eq(%result1, %expected) : (tensor<2x!pphlo.secret<i64>>,tensor<2xi64>)->()
  func.return
}
