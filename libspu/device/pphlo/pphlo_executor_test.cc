// Copyright 2021 Ant Group Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <array>
#include <cstddef>
#include <exception>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xtensor/xarray.hpp"

#include "libspu/device/pphlo/pphlo_executor_test_runner.h"

namespace spu::device::pphlo::test {

class ExecutorTest : public ::testing::TestWithParam<
                         std::tuple<size_t, FieldType, ProtocolKind>> {};

TEST_P(ExecutorTest, Basic) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));
  r.addInput(1);
  r.addInput(2);

  r.run(R"(
func.func @main(%arg0: tensor<i32>, %arg1: tensor<i32>) -> (tensor<i32>) {
  %0 = pphlo.add %arg0, %arg1 : tensor<i32>
  return %0 : tensor<i32>
})");

  r.verifyScalarOutput(3);
}

TEST_P(ExecutorTest, BoolSplatConstant) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  r.getConfig().set_enable_type_checker(false);

  r.run(R"(
func.func @main() -> (tensor<i32>) {
  %0 = pphlo.constant dense<true> : tensor<i1>
  %1 = pphlo.constant dense<1> : tensor<i32>
  %2 = pphlo.constant dense<0> : tensor<i32>
  %3 = pphlo.select %0, %1, %2 : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
  return %3 : tensor<i32>
})");

  int32_t expected = 1;
  r.verifyOutput(&expected);
}

TEST_P(ExecutorTest, EmptyConstant) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  r.run(R"(
func.func @main() -> tensor<0xf32> {
  %0 = pphlo.constant dense<> : tensor<0xf32>
  return %0 : tensor<0xf32>
})");

  r.verifyOutput<float>(nullptr);
}

TEST_P(ExecutorTest, BoolConstant) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  r.getConfig().set_enable_type_checker(false);

  r.run(R"(
func.func @main() -> (tensor<2xi32>) {
  %0 = pphlo.constant dense<[true,false]> : tensor<2xi1>
  %1 = pphlo.constant dense<1> : tensor<2xi32>
  %2 = pphlo.constant dense<0> : tensor<2xi32>
  %3 = pphlo.select %0, %1, %2 : (tensor<2xi1>, tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  return %3 : tensor<2xi32>
})");

  std::array<int32_t, 2> expected{1, 0};
  r.verifyOutput(expected.data());
}

TEST_P(ExecutorTest, ComplexConstant) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  r.getConfig().set_enable_type_checker(false);

  r.run(R"(
func.func @main() -> (tensor<2xcomplex<f32>>) {
  %0 = pphlo.constant dense<[(1.5, 2.5), (3.5, 4.5)]> : tensor<2xcomplex<f32>>
  return %0 : tensor<2xcomplex<f32>>
})");

  std::vector<std::complex<float>> expected = {{1.5, 2.5}, {3.5, 4.5}};
  r.verifyOutput(expected.data());
}

TEST_P(ExecutorTest, InvalidIR) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  ASSERT_THROW(r.run(R"(
func.func @main() -> tensor<i32> {
  %2 = pphlo.constant dense<[0x41DA6E5887800000, 0x41C94E3940000000, 0x41C4BD2007000000, 0x41DC95133AC00000, 0x41D1650CEC000000, 0x41C9DF42E7800000, 0x41D46C43B6800000, 0x41C467EE0E800000, 0x41DC705F14400000]> : tensor<9xf64>
  %3 = pphlo.floor %2 : tensor<9xf64>
  %9 = pphlo.concatenate %3 dim = 0 : (tensor<9xf64>) -> tensor<9xf64>
  %10 = pphlo.broadcast %9, dims = [13] : (tensor<9xf64>) -> tensor<9xf64>
  %51 = pphlo.constant dense<5> : tensor<i32>
  pphlo.return %51 : tensor<i32>
})"),
               std::exception);
}

TEST_P(ExecutorTest, WithConst) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));
  r.addInput(xt::xarray<int>({{1, 1}, {1, 1}}));

  r.run(R"(
func.func @main(%arg0: tensor<2x2xi32>) -> (tensor<2x2xi32>) {
    %0 = pphlo.constant dense<[[1,2],[3,4]]> : tensor<2x2xi32>
    %1 = pphlo.add %arg0, %0 : tensor<2x2xi32>
    return %1 : tensor<2x2xi32>
})");

  std::array<int, 4> expect = {2, 3, 4, 5};
  r.verifyOutput(expect.data());
}

TEST_P(ExecutorTest, RowConcat) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));
  r.addInput(xt::xarray<int>({{1, 2, 3}, {4, 5, 6}}));
  r.addInput(xt::xarray<int>({{7, 8, 9}, {10, 11, 12}}));

  r.run(R"(
func.func @main(%arg0: tensor<2x3xi32>, %arg1: tensor<2x3xi32>) -> (tensor<4x3xi32>) {
  %0 = pphlo.concatenate %arg0, %arg1 dim = 0 : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<4x3xi32>
  return %0 : tensor<4x3xi32>
})");

  std::array<int, 12> expect = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  r.verifyOutput(expect.data());
}

TEST_P(ExecutorTest, ColConcat) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));
  r.addInput(xt::xarray<int>({{1, 2, 3}, {4, 5, 6}}));
  r.addInput(xt::xarray<int>({{7, 8, 9}, {10, 11, 12}}));

  r.run(R"(
func.func @main(%arg0: tensor<2x3xi32>, %arg1: tensor<2x3xi32>) -> (tensor<2x6xi32>) {
  %0 = pphlo.concatenate %arg0, %arg1 dim = 1 : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x6xi32>
  return %0 : tensor<2x6xi32>
}
  )");

  std::array<int, 12> expect = {1, 2, 3, 7, 8, 9, 4, 5, 6, 10, 11, 12};
  r.verifyOutput(expect.data());
}

TEST_P(ExecutorTest, VariadicConcat) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));
  r.addInput(xt::xarray<int>({1, 2, 3}));

  r.run(R"(
func.func @main(%arg0: tensor<3xi32>) -> (tensor<9xi32>) {
  %0 = pphlo.concatenate %arg0, %arg0, %arg0 dim = 0 : (tensor<3xi32>, tensor<3xi32>, tensor<3xi32>) -> tensor<9xi32>
  return %0 : tensor<9xi32>
})");

  std::array<int, 12> expect = {1, 2, 3, 1, 2, 3, 1, 2, 3};
  r.verifyOutput(expect.data());
}

TEST_P(ExecutorTest, Slice) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));
  r.addInput(xt::xarray<int>({{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}}));

  r.run(R"(
func.func @main(%arg0: tensor<4x3xi32>) -> (tensor<2x2xi32>) {
  %0 = pphlo.slice %arg0 [2:1:4, 1:1:3] : (tensor<4x3xi32>) -> tensor<2x2xi32>
  return %0 : tensor<2x2xi32>
})");

  std::array<int, 4> expect = {7, 8, 10, 11};
  r.verifyOutput(expect.data());
}

TEST_P(ExecutorTest, SliceStride) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));
  r.addInput(xt::xarray<int>({                           //
                              {0, 1, 2, 3, 4, 5},        //
                              {6, 7, 8, 9, 10, 11},      //
                              {12, 13, 14, 15, 16, 17},  //
                              {18, 19, 20, 21, 22, 23}}));

  r.run(R"(
func.func @main(%arg0: tensor<4x6xi32>) -> (tensor<2x3xi32>) {
  %0 = pphlo.slice %arg0 [0:2:4, 0:2:6] : (tensor<4x6xi32>) -> tensor<2x3xi32>
  return %0 : tensor<2x3xi32>
})");

  std::array<int, 6> expect = {0,  2,  4,  //
                               12, 14, 16};
  r.verifyOutput(expect.data());
}

TEST_P(ExecutorTest, Reshape) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));
  r.addInput(xt::xarray<int>({{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}}));

  // Reshape to 2x6
  r.run(R"(
func.func @main(%arg0: tensor<4x3xi32>) -> (tensor<2x6xi32>) {
  %0 = pphlo.reshape %arg0 : (tensor<4x3xi32>) -> tensor<2x6xi32>
  return %0 : tensor<2x6xi32>
}
  )");

  std::array<int, 12> expected = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  r.verifyOutput(expected.data());
}

TEST_P(ExecutorTest, While) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));
  r.addInput(1);
  r.addInput(3);

  // while(x < y) { x = x + 1; }
  r.run(R"(
func.func @main(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
  %0, %1 = pphlo.while(%arg2 = %arg0, %arg3 = %arg1): tensor<i32>, tensor<i32>
  cond {
    %2 = pphlo.less %arg2, %arg3 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    pphlo.return %2 : tensor<i1>
  } do {
    %2 = pphlo.constant dense<1> : tensor<i32>
    %3 = pphlo.add %arg2, %2 : tensor<i32>
    pphlo.return %3, %arg3 : tensor<i32>, tensor<i32>
  }
  return %0 : tensor<i32>
})");

  r.verifyScalarOutput(3);
}

TEST_P(ExecutorTest, Reduce1D) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  const xt::xarray<int> in1({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  r.addInput(in1);

  r.run(R"(
func.func @main(%arg0: tensor<10xi32>) -> (tensor<i32>) {
  %0 = pphlo.constant dense<0> : tensor<i32>
  %1 = pphlo.reduce(%arg0 init: %0) applies pphlo.add across dimensions = [0] : (tensor<10xi32>, tensor<i32>) -> tensor<i32>
  return %1 :  tensor<i32>
})");

  xt::xarray<int> expect = xt::sum(in1);
  r.verifyOutput(expect.data());
}

TEST_P(ExecutorTest, Reduce2D1) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  const xt::xarray<int> in1({{1, 2, 3}, {4, 5, 6}});
  r.addInput(in1);

  r.run(R"(
func.func @main(%arg0: tensor<2x3xi32>) -> (tensor<2xi32>) {
  %0 = pphlo.constant dense<0> : tensor<i32>
  %1 = pphlo.reduce(%arg0 init: %0) applies pphlo.add across dimensions = [1] : (tensor<2x3xi32>, tensor<i32>) -> tensor<2xi32>
  return %1 :  tensor<2xi32>
})");

  xt::xarray<int> expect = {6, 15};
  r.verifyOutput(expect.data());
}

TEST_P(ExecutorTest, Reduce2D2) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  const xt::xarray<int> in1({{1, 2, 3}, {4, 5, 6}});
  r.addInput(in1);

  r.run(R"(
func.func @main(%arg0: tensor<2x3xi32>) -> (tensor<3xi32>) {
  %0 = pphlo.constant dense<0> : tensor<i32>
  %1 = pphlo.reduce(%arg0 init: %0) applies pphlo.add across dimensions = [0] : (tensor<2x3xi32>, tensor<i32>) -> tensor<3xi32>
  return %1 :  tensor<3xi32>
})");

  xt::xarray<int> expect = {5, 7, 9};
  r.verifyOutput(expect.data());
}

TEST_P(ExecutorTest, VReduce) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  const xt::xarray<int> in1({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  r.addInput(in1);

  r.run(R"(
func.func @main(%arg0: tensor<10xi32>) -> (tensor<i32>, tensor<i32>) {
  %0 = pphlo.constant dense<0> : tensor<i32>
  %1:2 = pphlo.reduce(%arg0 init: %0), (%arg0 init: %0) across dimensions = [0]: (tensor<10xi32>, tensor<10xi32>, tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>)
          reducer(%arg1: tensor<i32>, %arg3: tensor<i32>) (%arg2: tensor<i32>, %arg4: tensor<i32>) {
            %2 = pphlo.add %arg1, %arg3 : tensor<i32>
            %3 = pphlo.maximum %arg2, %arg4 : tensor<i32>
            pphlo.return %2, %3 : tensor<i32>, tensor<i32>
          }
  return %1#0, %1#1 : tensor<i32>, tensor<i32>
})",
        2);

  xt::xarray<int> expect0 = xt::sum(in1);
  r.verifyOutput(expect0.data(), 0);
  xt::xarray<int> expect1 = {10};
  r.verifyOutput(expect1.data(), 1);
}

TEST_P(ExecutorTest, MaxReduce) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  const xt::xarray<float> in1({{0, 0, 0, 0, 0, 0, 0, 0, 0, 0}});
  r.addInput(in1);

  r.run(R"(
func.func @main(%arg0: tensor<1x10xf32>) -> (tensor<1xf32>) {
  // Initial value is -inf
  %0 = pphlo.constant dense<0xFF800000> : tensor<f32>
  %1 = pphlo.reduce(%arg0 init: %0) applies pphlo.maximum across dimensions = [1] : (tensor<1x10xf32>, tensor<f32>) -> tensor<1xf32>
  return %1 :  tensor<1xf32>
})");

  xt::xarray<float> expect = {0};
  r.verifyOutput(expect.data());
}

TEST_P(ExecutorTest, ReduceMultiDims) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  const xt::xarray<float> in1({{{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}},
                               {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}});
  r.addInput(in1, VIS_SECRET);
  r.addInput(in1, VIS_SECRET);

  r.run(R"(
func.func @main(%arg0: tensor<2x3x4x!pphlo.secret<f32>>, %arg1: tensor<2x3x4x!pphlo.secret<f32>>) -> (tensor<!pphlo.secret<i1>>) {
  %0 = pphlo.constant dense<true> : tensor<i1>
  %1 = pphlo.equal %arg0, %arg1 : (tensor<2x3x4x!pphlo.secret<f32>>, tensor<2x3x4x!pphlo.secret<f32>>) -> tensor<2x3x4x!pphlo.secret<i1>>
  %2 = pphlo.convert %0 : (tensor<i1>) -> tensor<!pphlo.secret<i1>>
  %3 = pphlo.reduce(%1 init: %2) applies pphlo.and across dimensions = [0,1,2] : (tensor<2x3x4x!pphlo.secret<i1>>, tensor<!pphlo.secret<i1>>) -> tensor<!pphlo.secret<i1>>
  return %3 :  tensor<!pphlo.secret<i1>>
})");

  xt::xarray<bool> expect = {true};
  r.verifyOutput(expect.data());
}

TEST_P(ExecutorTest, ReduceWindow) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  const xt::xarray<int> in1 = {{-7, 6, 1, -14, -7, 5},
                               {-13, -14, -11, 13, -13, -7},
                               {8, -11, 12, -2, 14, 4},
                               {0, 13, 3, -13, -7, -3}};
  r.addInput(in1);

  r.run(R"(
func.func @main(%arg0: tensor<4x6xi32>) -> (tensor<2x2xi32>) {
  %0 = pphlo.constant dense<0> : tensor<i32>
  %1 = "pphlo.reduce_window"(%arg0, %0) ( {
    ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):  // no predecessors
      %2 = pphlo.add %arg1, %arg2 : tensor<i32>
      pphlo.return %2 : tensor<i32>
    }) {
      base_dilations = array<i64: 1, 1>,
      window_dilations = array<i64: 1, 1>,
      window_dimensions = array<i64: 2,3>,
      window_strides = array<i64: 2,3>
    } : (tensor<4x6xi32>, tensor<i32>) -> tensor<2x2xi32>

  return %1 :  tensor<2x2xi32>
})");

  xt::xarray<int> expect = {{-38, -23}, {25, -7}};
  r.verifyOutput(expect.data());
}

TEST_P(ExecutorTest, ReduceWindowStableHloTest) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  const xt::xarray<int> in1 = {{1, 2}, {3, 4}, {5, 6}};
  r.addInput(in1);

  r.run(r.compileMHlo(R"(
func.func @main(%arg0: tensor<3x2xi32>) -> (tensor<2x2xi32>) {
  %0 = stablehlo.constant dense<0> : tensor<i32>
  %1 = "stablehlo.reduce_window"(%arg0, %0) ( {
    ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):  // no predecessors
      %2 = stablehlo.maximum %arg1, %arg2 : tensor<i32>
      stablehlo.return %2 : tensor<i32>
    }) {
      base_dilations = array<i64: 2, 1>,
      padding = dense<[[2, 1], [0, 0]]> : tensor<2x2xi64>,
      window_dilations = array<i64: 3, 1>,
      window_dimensions = array<i64: 2, 1>,
      window_strides = array<i64: 4, 1>
    } : (tensor<3x2xi32>, tensor<i32>) -> tensor<2x2xi32>
  return %1 : tensor<2x2xi32>
})",
                      {Visibility::VIS_PUBLIC}));

  xt::xarray<int> expect = {{0, 0}, {3, 4}};
  r.verifyOutput(expect.data());
}

TEST_P(ExecutorTest, ReduceWindowStableHloTest2) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  const xt::xarray<int> in1 = {{1, 2}, {3, 4}, {5, 6}};
  r.addInput(in1);

  r.run(r.compileMHlo(R"(
func.func @main(%arg0: tensor<3x2xi32>) -> (tensor<1x2xi32>) {
  %0 = stablehlo.constant dense<0> : tensor<i32>
  %1 = "stablehlo.reduce_window"(%arg0, %0) ( {
    ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):  // no predecessors
      %2 = stablehlo.maximum %arg1, %arg2 : tensor<i32>
      stablehlo.return %2 : tensor<i32>
    }) {
      base_dilations = array<i64: 2, 1>,
      padding = dense<[[2, 1], [0, 0]]> : tensor<2x2xi64>,
      window_dilations = array<i64: 3, 1>,
      window_dimensions = array<i64: 3, 1>,
      window_strides = array<i64: 4, 1>
    } : (tensor<3x2xi32>, tensor<i32>) -> tensor<1x2xi32>
  return %1 : tensor<1x2xi32>
})",
                      {Visibility::VIS_PUBLIC}));

  xt::xarray<int> expect = {{5, 6}};
  r.verifyOutput(expect.data());
}

TEST_P(ExecutorTest, ReduceWindowDefaultStrides) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  const xt::xarray<int> in1 = {{-7, 6, 1, -14, -7, 5},
                               {-13, -14, -11, 13, -13, -7},
                               {8, -11, 12, -2, 14, 4},
                               {0, 13, 3, -13, -7, -3}};
  r.addInput(in1);

  r.run(R"(
func.func @main(%arg0: tensor<4x6xi32>) -> (tensor<3x4xi32>) {
  %0 = pphlo.constant dense<0> : tensor<i32>
  %1 = "pphlo.reduce_window"(%arg0, %0) ( {
    ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):  // no predecessors
      %2 = pphlo.maximum %arg1, %arg2 : tensor<i32>
      pphlo.return %2 : tensor<i32>
    }) {
      base_dilations = array<i64: 1, 1>,
      window_dilations = array<i64: 1, 1>,
      window_dimensions = array<i64: 2,3>,
      window_strides = array<i64: 1, 1>
    } : (tensor<4x6xi32>, tensor<i32>) -> tensor<3x4xi32>

  return %1 :  tensor<3x4xi32>
})");

  xt::xarray<int> expect = {
      {6, 13, 13, 13}, {12, 13, 14, 14}, {13, 13, 14, 14}};
  r.verifyOutput(expect.data());
}

TEST_P(ExecutorTest, ReduceWindowIotaWindowDilation) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  const xt::xarray<int> in1 = {
      {0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}, {12, 13, 14, 15}};
  r.addInput(in1);

  r.run(R"(
func.func @main(%arg0: tensor<4x4xi32>) -> (tensor<2x2xi32>) {
  %0 = pphlo.constant dense<0> : tensor<i32>
  %1 = "pphlo.reduce_window"(%arg0, %0) ( {
    ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):  // no predecessors
      %2 = pphlo.maximum %arg1, %arg2 : tensor<i32>
      pphlo.return %2 : tensor<i32>
    }) {
      base_dilations = array<i64: 1, 1>,
      window_dilations = array<i64: 2, 2>,
      window_dimensions = array<i64: 2, 2>,
      window_strides = array<i64: 1, 1>
    } : (tensor<4x4xi32>, tensor<i32>) -> tensor<2x2xi32>

  return %1 :  tensor<2x2xi32>
})");

  xt::xarray<int> expect = {{10, 11}, {14, 15}};
  r.verifyOutput(expect.data());
}

TEST_P(ExecutorTest, ReduceWindowIotaStrideWindowDilation) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  const xt::xarray<int> in1 = {
      {0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}, {12, 13, 14, 15}};
  r.addInput(in1);

  r.run(R"(
func.func @main(%arg0: tensor<4x4xi32>) -> (tensor<1x1xi32>) {
  %0 = pphlo.constant dense<0> : tensor<i32>
  %1 = "pphlo.reduce_window"(%arg0, %0) ( {
    ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):  // no predecessors
      %2 = pphlo.maximum %arg1, %arg2 : tensor<i32>
      pphlo.return %2 : tensor<i32>
    }) {
      base_dilations = array<i64: 1, 1>,
      window_dilations = array<i64: 2, 2>,
      window_dimensions = array<i64: 2, 2>,
      window_strides = array<i64: 2, 2>
    } : (tensor<4x4xi32>, tensor<i32>) -> tensor<1x1xi32>

  return %1 : tensor<1x1xi32>
})");

  xt::xarray<int> expect = {10};
  r.verifyOutput(expect.data());
}

TEST_P(ExecutorTest, ReduceWindowMaxIotaBaseDilation) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  const xt::xarray<int> in1 = {
      {0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}, {12, 13, 14, 15}};
  r.addInput(in1);

  r.run(r.compileMHlo(R"(
func.func @main(%arg0: tensor<4x4xi32>) -> (tensor<6x6xi32>) {
  %0 = stablehlo.constant dense<0> : tensor<i32>
  %1 = "stablehlo.reduce_window"(%arg0, %0) ( {
    ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):  // no predecessors
      %2 = stablehlo.maximum %arg1, %arg2 : tensor<i32>
      stablehlo.return %2 : tensor<i32>
    }) {
      base_dilations = array<i64: 2, 2>,
      padding = dense<0> : tensor<2x2xi64>,
      window_dilations = array<i64: 1, 1>,
      window_dimensions = array<i64: 2, 2>,
      window_strides = array<i64: 1, 1>
    } : (tensor<4x4xi32>, tensor<i32>) -> tensor<6x6xi32>
  return %1 : tensor<6x6xi32>
})",
                      {Visibility::VIS_PUBLIC}));

  xt::xarray<int> expect = {{0, 1, 1, 2, 2, 3},    {4, 5, 5, 6, 6, 7},
                            {4, 5, 5, 6, 6, 7},    {8, 9, 9, 10, 10, 11},
                            {8, 9, 9, 10, 10, 11}, {12, 13, 13, 14, 14, 15}};
  r.verifyOutput(expect.data());
}

TEST_P(ExecutorTest, ReduceWindowMaxIotaStrideBaseDilation) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  const xt::xarray<int> in1 = {
      {0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}, {12, 13, 14, 15}};
  r.addInput(in1);

  auto compiled = r.compileMHlo(R"(
func.func @main(%arg0: tensor<4x4xi32>) -> (tensor<3x3xi32>) {
  %0 = stablehlo.constant dense<0> : tensor<i32>
  %1 = "stablehlo.reduce_window"(%arg0, %0) ( {
    ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):  // no predecessors
      %2 = stablehlo.maximum %arg1, %arg2 : tensor<i32>
      stablehlo.return %2 : tensor<i32>
    }) {
      base_dilations = array<i64: 2, 2>,
      padding = dense<0> : tensor<2x2xi64>,
      window_dilations = array<i64: 1, 1>,
      window_dimensions = array<i64: 2, 2>,
      window_strides = array<i64: 2, 2>
    } : (tensor<4x4xi32>, tensor<i32>) -> tensor<3x3xi32>

  return %1 :  tensor<3x3xi32>
})",
                                {Visibility::VIS_PUBLIC});

  r.run(compiled);

  xt::xarray<int> expect = {{0, 1, 2}, {4, 5, 6}, {8, 9, 10}};
  r.verifyOutput(expect.data());
}

TEST_P(ExecutorTest, ReduceWindowMaxIotaStrideBothDilation) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  const xt::xarray<int> in1 = {
      {0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}, {12, 13, 14, 15}};
  r.addInput(in1);

  auto compiled = r.compileMHlo(R"(
func.func @main(%arg0: tensor<4x4xi32>) -> (tensor<3x3xi32>) {
  %0 = stablehlo.constant dense<0> : tensor<i32>
  %1 = "stablehlo.reduce_window"(%arg0, %0) ( {
    ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):  // no predecessors
      %2 = stablehlo.maximum %arg1, %arg2 : tensor<i32>
      stablehlo.return %2 : tensor<i32>
    }) {
      base_dilations = array<i64: 2, 2>,
      padding = dense<0> : tensor<2x2xi64>,
      window_dilations = array<i64: 2, 2>,
      window_dimensions = array<i64: 2, 2>,
      window_strides = array<i64: 2, 2>
    } : (tensor<4x4xi32>, tensor<i32>) -> tensor<3x3xi32>

  return %1 :  tensor<3x3xi32>
})",
                                {Visibility::VIS_PUBLIC});

  r.run(compiled);

  xt::xarray<int> expect = {{5, 6, 7}, {9, 10, 11}, {13, 14, 15}};
  r.verifyOutput(expect.data());
}

TEST_P(ExecutorTest, ReduceWindowMaxIotaPaddingStrideBaseDilation) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  const xt::xarray<int> in1 = {
      {0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}, {12, 13, 14, 15}};
  r.addInput(in1);

  auto compiled = r.compileMHlo(R"(
func.func @main(%arg0: tensor<4x4xi32>) -> (tensor<3x3xi32>) {
  %0 = stablehlo.constant dense<0> : tensor<i32>
  %1 = "stablehlo.reduce_window"(%arg0, %0) ( {
    ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):  // no predecessors
      %2 = stablehlo.maximum %arg1, %arg2 : tensor<i32>
      stablehlo.return %2 : tensor<i32>
    }) {
      base_dilations = array<i64: 2, 2>,
      padding = dense<1> : tensor<2x2xi64>,
      window_dilations = array<i64: 1, 1>,
      window_dimensions = array<i64: 3, 3>,
      window_strides = array<i64: 3, 3>
    } : (tensor<4x4xi32>, tensor<i32>) -> tensor<3x3xi32>

  return %1 : tensor<3x3xi32>
})",
                                {Visibility::VIS_PUBLIC});

  r.run(compiled);

  xt::xarray<int> expect = {{0, 2, 3}, {8, 10, 11}, {12, 14, 15}};
  r.verifyOutput(expect.data());
}

TEST_P(ExecutorTest, If) {
  const auto *prog = R"(
 func.func @main(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = pphlo.constant dense<1.000000e+01> : tensor<f32>
  %1 = pphlo.less %arg0, %0 : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %2 = "pphlo.if"(%1) ( {
    %3 = pphlo.multiply %arg0, %arg0 : tensor<f32>
    pphlo.return %3 : tensor<f32>
  },  {
    %3 = pphlo.add %arg0, %arg0 : tensor<f32>
    pphlo.return %3 : tensor<f32>
  }) : (tensor<i1>) -> tensor<f32>
  return %2 : tensor<f32>
}
)";
  {
    // True case
    Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
             std::get<2>(GetParam()));

    r.addInput(2.5F);

    r.run(prog);

    r.verifyScalarOutput(2.5F * 2.5F);
  }

  {
    // False case
    Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
             std::get<2>(GetParam()));

    r.addInput(12.5F);

    r.run(prog);

    r.verifyScalarOutput(12.5F + 12.5F);
  }
}

TEST_P(ExecutorTest, SecretControlFlow) {
  const auto *prog = R"(
func.func @main(%arg0: tensor<f32>) -> tensor<!pphlo.secret<f32>> {
  %0 = pphlo.constant dense<1.000000e+01> : tensor<f32>
  %1 = pphlo.convert %arg0 : (tensor<f32>) -> tensor<!pphlo.secret<f32>>
  %2 = pphlo.less %1, %0 : (tensor<!pphlo.secret<f32>>, tensor<f32>) -> tensor<!pphlo.secret<i1>>
  %3 = "pphlo.if"(%2) ( {
    %4 = pphlo.multiply %arg0, %arg0 : tensor<f32>
    pphlo.return %4 : tensor<f32>
  },  {
    %4 = pphlo.add %arg0, %arg0 : tensor<f32>
    pphlo.return %4 : tensor<f32>
  }) : (tensor<!pphlo.secret<i1>>) -> tensor<!pphlo.secret<f32>>
  return %3 : tensor<!pphlo.secret<f32>>
}
)";

  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  r.addInput(2.5F);

  r.run(prog);

  r.verifyScalarOutput(2.5F * 2.5F);
}

TEST_P(ExecutorTest, Iota1D) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  r.run(R"(
func.func @main() -> (tensor<4xi32>) {
    %0 = pphlo.iota dim = 0 : tensor<4xi32>
    return %0 : tensor<4xi32>
})");

  std::array<int, 4> expect = {0, 1, 2, 3};
  r.verifyOutput(expect.data());
}

TEST_P(ExecutorTest, Iota2D) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  r.run(R"(
func.func @main() -> (tensor<4x2xi32>) {
    %0 = pphlo.iota dim = 1 : tensor<4x2xi32>
    return %0 : tensor<4x2xi32>
})");

  std::array<int, 8> expect = {0, 1, 0, 1, 0, 1, 0, 1};
  r.verifyOutput(expect.data());
}

TEST_P(ExecutorTest, IotaComplex) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  r.run(R"(
func.func @main() -> (tensor<4xcomplex<f32>>) {
    %0 = pphlo.iota dim = 0 : tensor<4xcomplex<f32>>
    return %0 : tensor<4xcomplex<f32>>
})");

  std::vector<std::complex<float>> expect = {{0, 0}, {1, 0}, {2, 0}, {3, 0}};
  r.verifyOutput(expect.data());
}

TEST_P(ExecutorTest, NonComplexReal) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  r.addInput(std::vector<float>{1, 2, 3, 4});

  r.run(R"(
func.func @main(%arg0:tensor<4xf32>) -> (tensor<4xf32>) {
    %0 = pphlo.real %arg0 : tensor<4xf32>
    return %0 : tensor<4xf32>
})");

  std::vector<float> expect = {1, 2, 3, 4};
  r.verifyOutput(expect.data());
}

TEST_P(ExecutorTest, NonComplexImag) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  r.addInput(std::vector<float>{1, 2, 3, 4});

  r.run(R"(
func.func @main(%arg0:tensor<4xf32>) -> (tensor<4xf32>) {
    %0 = pphlo.imag %arg0 : tensor<4xf32>
    return %0 : tensor<4xf32>
})");

  std::vector<float> expect = {0, 0, 0, 0};
  r.verifyOutput(expect.data());
}

TEST_P(ExecutorTest, SimpleBitcast) {
  GTEST_SKIP();

  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  // the fixed-point bitcast behaves differently then floating point ones.
  float in = 2.0F;
  r.addInput(in);

  r.run(R"(
func.func @main(%arg0: tensor<f32>) -> (tensor<i32>) {
    %0 = "pphlo.bitcast_convert"(%arg0) {elsize = 32 : i64} : (tensor<f32>) -> tensor<i32>
    return %0 : tensor<i32>
})");

  r.verifyOutput(reinterpret_cast<int32_t *>(&in));
}

TEST_P(ExecutorTest, Simple4x4Conv2DWith2x2Kernel) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  xt::xarray<float> lhs = {{{
      {1, 2, 3, 4},
      {5, 6, 7, 8},
      {9, 10, 11, 12},
      {13, 14, 15, 16},
  }}};
  r.addInput(lhs);

  xt::xarray<float> rhs = {{{
      {5, 6},
      {7, 8},
  }}};
  r.addInput(rhs);

  auto ir = r.compileMHlo(R"(
func.func @main(%arg0: tensor<1x1x4x4xf32>, %arg1: tensor<1x1x2x2xf32>) -> (tensor<1x1x4x4xf32>) {
    %0 = stablehlo.convolution(%arg0, %arg1)
            dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
            window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
            {
              batch_group_count = 1 : i64,
              feature_group_count = 1 : i64
            } : (tensor<1x1x4x4xf32>, tensor<1x1x2x2xf32>) -> tensor<1x1x4x4xf32>
    return %0 : tensor<1x1x4x4xf32>
})",
                          {VIS_PUBLIC, VIS_PUBLIC});

  r.run(ir);

  xt::xarray<float> expected = {{{
      {100, 126, 152, 76},
      {204, 230, 256, 124},
      {308, 334, 360, 172},
      {149, 160, 171, 80},
  }}};
  r.verifyOutput(expected.data());
}

TEST_P(ExecutorTest, Conv2DGeneralDimensions) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  xt::xarray<float> lhs = {
      {{{1, 2, 3, 4}}, {{5, 6, 7, 8}}, {{9, 10, 11, 12}}},
      {{{13, 14, 15, 16}}, {{17, 18, 19, 20}}, {{21, 22, 23, 24}}}};
  r.addInput(lhs);

  xt::xarray<float> rhs = {{{{1, 7, 13}, {4, 10, 16}},
                            {{2, 8, 14}, {5, 11, 17}},
                            {{3, 9, 15}, {6, 12, 18}}}};

  r.addInput(rhs);

  auto ir = r.compileMHlo(R"(
func.func @main(%arg0: tensor<2x3x1x4xf32>, %arg1:tensor<1x3x2x3xf32>) -> (tensor<1x1x1x2xf32>) {
    %0 = stablehlo.convolution(%arg0, %arg1)
          dim_numbers = [f, 0, b, 1]x[o, 1, i,0]->[f, 0, b, 1],
          window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
          {
            batch_group_count = 1 : i64,
            feature_group_count = 1 : i64
          } : (tensor<2x3x1x4xf32>,tensor<1x3x2x3xf32>) -> tensor<1x1x1x2xf32>
    return %0 : tensor<1x1x1x2xf32>
})",
                          {VIS_PUBLIC, VIS_PUBLIC});

  r.run(ir);

  xt::xarray<float> expected = {{{{2514, 2685}}}};
  r.verifyOutput(expected.data());
}

TEST_P(ExecutorTest, DilatedBaseConv2DWithHighPadding) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  xt::xarray<float> lhs = {{{
      {1, 2, 3, 4},  //
      {5, 6, 7, 8},
      {9, 10, 11, 12},
      {13, 14, 15, 16},
  }}};
  r.addInput(lhs);

  xt::xarray<float> rhs = {{{
      {5, 6},  //
      {7, 8},
  }}};

  r.addInput(rhs);

  auto ir = r.compileMHlo(R"(
func.func @main(%arg0: tensor<1x1x4x4xf32>, %arg1: tensor<1x1x2x2xf32>) -> (tensor<1x1x7x7xf32>) {
    %0 = stablehlo.convolution(%arg0, %arg1)
          dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
          window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]}
          {
            batch_group_count = 1 : i64,
            feature_group_count = 1 : i64
          } : (tensor<1x1x4x4xf32>, tensor<1x1x2x2xf32>) -> tensor<1x1x7x7xf32>
    return %0 : tensor<1x1x7x7xf32>
})",
                          {VIS_PUBLIC, VIS_PUBLIC});

  r.run(ir);

  xt::xarray<float> expected = {{{5, 12, 10, 18, 15, 24, 20},
                                 {35, 48, 42, 56, 49, 64, 56},
                                 {25, 36, 30, 42, 35, 48, 40},
                                 {63, 80, 70, 88, 77, 96, 84},
                                 {45, 60, 50, 66, 55, 72, 60},
                                 {91, 112, 98, 120, 105, 128, 112},
                                 {65, 84, 70, 90, 75, 96, 80}}};
  r.verifyOutput(expected.data());
}

TEST_P(ExecutorTest, DilatedBaseConv2DWithLowAndHighPadding) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  xt::xarray<float> lhs = {{{
      {1, 2, 3, 4},  //
      {5, 6, 7, 8},
      {9, 10, 11, 12},
      {13, 14, 15, 16},
  }}};
  r.addInput(lhs);

  xt::xarray<float> rhs = {{{
      {5, 6},  //
      {7, 8},
  }}};

  r.addInput(rhs);

  auto ir = r.compileMHlo(R"(
func.func @main(%arg0: tensor<1x1x4x4xf32>, %arg1: tensor<1x1x2x2xf32>) -> (tensor<1x1x8x8xf32>) {
    %0 = stablehlo.convolution(%arg0, %arg1)
          dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
          window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]}
          {
            batch_group_count = 1 : i64,
            feature_group_count = 1 : i64
          } : (tensor<1x1x4x4xf32>, tensor<1x1x2x2xf32>) -> tensor<1x1x8x8xf32>
    return %0 : tensor<1x1x8x8xf32>
})",
                          {VIS_PUBLIC, VIS_PUBLIC});

  r.run(ir);

  xt::xarray<float> expected = {{
      {8, 7, 16, 14, 24, 21, 32, 28},
      {6, 5, 12, 10, 18, 15, 24, 20},
      {40, 35, 48, 42, 56, 49, 64, 56},
      {30, 25, 36, 30, 42, 35, 48, 40},
      {72, 63, 80, 70, 88, 77, 96, 84},
      {54, 45, 60, 50, 66, 55, 72, 60},
      {104, 91, 112, 98, 120, 105, 128, 112},
      {78, 65, 84, 70, 90, 75, 96, 80},
  }};
  r.verifyOutput(expected.data());
}

TEST_P(ExecutorTest, FlatRhsDilation) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  xt::xarray<float> lhs = {{{
      {0, 1, 2, 3, 4, 5},  //
      {6, 7, 8, 9, 10, 11},
      {12, 13, 14, 15, 16, 17},
      {18, 19, 20, 21, 22, 23},

  }}};
  r.addInput(lhs);

  xt::xarray<float> rhs = {{{{1, 10, 100},  //
                             {2, 20, 200}}}};

  r.addInput(rhs);

  auto ir = r.compileMHlo(R"(
func.func @main(%arg0: tensor<1x1x4x6xf32>, %arg1: tensor<1x1x2x3xf32>) -> (tensor<1x1x2x2xf32>) {
    %0 = stablehlo.convolution(%arg0, %arg1)
          dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
          window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]}
          {
            batch_group_count = 1 : i64,
            feature_group_count = 1 : i64
          } : (tensor<1x1x4x6xf32>, tensor<1x1x2x3xf32>) -> tensor<1x1x2x2xf32>
    return %0 : tensor<1x1x2x2xf32>
})",
                          {VIS_PUBLIC, VIS_PUBLIC});

  r.run(ir);

  xt::xarray<float> expected = {{3924, 4257}, {5922, 6255}};
  r.verifyOutput(expected.data());
}

TEST_P(ExecutorTest, ShiftLeft) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  xt::xarray<int> lhs = {1, 1};
  r.addInput(lhs);

  xt::xarray<int> rhs = {1, 2};
  r.addInput(rhs);

  r.run(R"(
func.func @main(%arg0: tensor<2xi32>, %arg1: tensor<2xi32>) -> (tensor<2xi32>) {
    %0 = pphlo.shift_left %arg0, %arg1 : tensor<2xi32>
    return %0 : tensor<2xi32>
})");

  xt::xarray<int> expected = {1 << 1, 1 << 2};
  r.verifyOutput(expected.data());
}

TEST_P(ExecutorTest, RightShiftLogical) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  xt::xarray<int> lhs = {1 << 4, 1 << 4};
  r.addInput(lhs);

  xt::xarray<int> rhs = {1, 2};
  r.addInput(rhs);

  r.run(R"(
func.func @main(%arg0: tensor<2xi32>, %arg1: tensor<2xi32>) -> (tensor<2xi32>) {
    %0 = pphlo.shift_right_logical %arg0, %arg1 : tensor<2xi32>
    return %0 : tensor<2xi32>
})");

  xt::xarray<int> expected = {1 << 3, 1 << 2};
  r.verifyOutput(expected.data());
}

TEST_P(ExecutorTest, Maximum) {
  if (std::get<1>(GetParam()) == FM32) {
    return;  // Ring type is not large enough to hold value
  }

  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  r.addInput(10);

  r.run(R"(
func.func @main(%arg0: tensor<i32>) -> (tensor<i32>) {
  %0 = pphlo.constant dense<-2147483648> : tensor<i32>
  %1 = pphlo.maximum %0, %arg0 : tensor<i32>
  return %1 :  tensor<i32>
})");

  int expected = 10;
  r.verifyOutput(&expected);
}

TEST_P(ExecutorTest, Minimum) {
  if (std::get<1>(GetParam()) == FM32) {
    return;  // Ring type is not large enough to hold value
  }

  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  r.addInput(10);

  r.run(R"(
func.func @main(%arg0: tensor<i32>) -> (tensor<i32>) {
  %0 = pphlo.constant dense<2147483647> : tensor<i32>
  %1 = pphlo.minimum %0, %arg0 : tensor<i32>
  return %1 : tensor<i32>
})");

  int expected = 10;
  r.verifyOutput(&expected);
}

TEST_P(ExecutorTest, DynamicSlice1D) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  std::vector<int> op = {0, 1, 2, 3, 4};
  r.addInput(op);

  r.addInput(2);

  r.run(R"(
func.func @main(%arg0: tensor<5xi32>, %arg1: tensor<i32>) -> tensor<2xi32> {
  %0 = pphlo.dynamic_slice %arg0, %arg1 sizes =[2] : (tensor<5xi32>, tensor<i32>) -> tensor<2xi32>
  return %0 : tensor<2xi32>
})");

  std::vector<int> expected = {2, 3};
  r.verifyOutput(expected.data());
}

TEST_P(ExecutorTest, DynamicSlice2D) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  xt::xarray<float> op = {{0.0, 1.0, 2.0},  //
                          {3.0, 4.0, 5.0},
                          {6.0, 7.0, 8.0},
                          {9.0, 10.0, 11.0}};
  r.addInput(op);

  r.addInput(2);
  r.addInput(1);

  r.run(R"(
func.func @main(%arg0: tensor<4x3xf32>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<2x2xf32> {
  %0 = pphlo.dynamic_slice %arg0, %arg1, %arg2 sizes = [2, 2] : (tensor<4x3xf32>, tensor<i32>, tensor<i32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
})");

  xt::xarray<float> expected = {{7.0, 8.0}, {10.0, 11.0}};
  r.verifyOutput(expected.data());
}

TEST_P(ExecutorTest, DynamicSlice3D) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  xt::xarray<int32_t> op = {{{0, 1, 9},  //
                             {3, 4, 9},
                             {6, 7, 9},
                             {9, 10, 9}}};
  r.addInput(op);

  r.addInput(0);
  r.addInput(1, VIS_SECRET);
  r.addInput(0);

  r.run(R"(
func.func @main(%arg0: tensor<1x4x3xi32>, %arg1: tensor<i32>, %arg2: tensor<!pphlo.secret<i32>>, %arg3: tensor<i32>) -> tensor<1x1x2x!pphlo.secret<i32>> {
  %0 = pphlo.dynamic_slice %arg0, %arg1, %arg2, %arg3 sizes = [1, 1, 2]
        : (tensor<1x4x3xi32>, tensor<i32>, tensor<!pphlo.secret<i32>>, tensor<i32>) -> tensor<1x1x2x!pphlo.secret<i32>>
  return %0: tensor<1x1x2x!pphlo.secret<i32>>
})");

  xt::xarray<int32_t> expected = {{{3, 4}}};
  r.verifyOutput(expected.data());
}

TEST_P(ExecutorTest, DynamicSlice3D_1) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  xt::xarray<int32_t> op = {{{0, 1, 9},  //
                             {3, 4, 9},
                             {6, 7, 9},
                             {9, 10, 9}}};
  r.addInput(op);

  r.addInput(0);
  r.addInput(1, VIS_SECRET);
  r.addInput(2);

  r.run(R"(
func.func @main(%arg0: tensor<1x4x3xi32>, %arg1: tensor<i32>, %arg2: tensor<!pphlo.secret<i32>>, %arg3: tensor<i32>) -> tensor<1x1x2x!pphlo.secret<i32>> {
  %0 = pphlo.dynamic_slice %arg0, %arg1, %arg2, %arg3 sizes = [1, 1, 2]
        : (tensor<1x4x3xi32>, tensor<i32>, tensor<!pphlo.secret<i32>>, tensor<i32>) -> tensor<1x1x2x!pphlo.secret<i32>>
  return %0: tensor<1x1x2x!pphlo.secret<i32>>
})");

  xt::xarray<int32_t> expected = {{{4, 9}}};
  r.verifyOutput(expected.data());
}

TEST_P(ExecutorTest, DynamicUpdateSlice1D) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  std::vector<int> op = {0, 1, 2, 3, 4};
  r.addInput(op);

  std::vector<int> u = {5, 6};
  r.addInput(u);

  r.addInput(2);

  r.run(R"(
func.func @main(%arg0: tensor<5xi32>, %arg1: tensor<2xi32>, %arg2: tensor<i32>) -> tensor<5xi32> {
  %0 = pphlo.dynamic_update_slice %arg0, %arg1, %arg2 : (tensor<5xi32>, tensor<2xi32>, tensor<i32>) -> tensor<5xi32>
  return %0 : tensor<5xi32>
})");

  std::vector<int> expected = {0, 1, 5, 6, 4};
  r.verifyOutput(expected.data());
}

TEST_P(ExecutorTest, DynamicUpdateSlice2D) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  xt::xarray<float> op = {{0.0, 1.0, 2.0},  //
                          {3.0, 4.0, 5.0},
                          {6.0, 7.0, 8.0},
                          {9.0, 10.0, 11.0}};
  r.addInput(op);

  xt::xarray<float> u = {{12.0, 13.0},  //
                         {14.0, 15.0},
                         {16.0, 17.0}};
  r.addInput(u);

  r.addInput(1);
  r.addInput(1);

  r.run(R"(
func.func @main(%arg0: tensor<4x3xf32>, %arg1: tensor<3x2xf32>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> tensor<4x3xf32> {
  %0 = pphlo.dynamic_update_slice %arg0, %arg1, %arg2, %arg3 : (tensor<4x3xf32>, tensor<3x2xf32>, tensor<i32>, tensor<i32>) -> tensor<4x3xf32>
  return %0 : tensor<4x3xf32>
})");

  xt::xarray<float> expected = {{0.0, 1.0, 2.0},  //
                                {3.0, 12.0, 13.0},
                                {6.0, 14.0, 15.0},
                                {9.0, 16.0, 17.0}};
  r.verifyOutput(expected.data());
}

TEST_P(ExecutorTest, Sort1D) {
  xt::xarray<float> op = {2.0, 1.0, 3.0, -10.0};
  xt::xarray<float> expected = {-10.0, 1.0, 2.0, 3.0};

  {
    Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
             std::get<2>(GetParam()));

    r.addInput(op);

    r.run(R"(
func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %0 = "pphlo.sort"(%arg0) ( {
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):  // no predecessors
      %1 = pphlo.less %arg1, %arg2 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      pphlo.return %1 : tensor<i1>
    }) {dimension = 0 : i64, is_stable = true} : (tensor<4xf32>) -> (tensor<4xf32>)
    return %0 : tensor<4xf32>
})");
    r.verifyOutput(expected.data());
  }

  {
    Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
             std::get<2>(GetParam()));

    r.addInput(op, VIS_SECRET);

    r.run(R"(
func.func @main(%arg0: tensor<4x!pphlo.secret<f32>>) -> tensor<4x!pphlo.secret<f32>> {
    %0 = "pphlo.sort"(%arg0) ( {
    ^bb0(%arg1: tensor<!pphlo.secret<f32>>, %arg2: tensor<!pphlo.secret<f32>>):  // no predecessors
      %1 = pphlo.less %arg1, %arg2 : (tensor<!pphlo.secret<f32>>, tensor<!pphlo.secret<f32>>) -> tensor<!pphlo.secret<i1>>
      pphlo.return %1 : tensor<!pphlo.secret<i1>>
    }) {dimension = 0 : i64, is_stable = true} : (tensor<4x!pphlo.secret<f32>>) -> (tensor<4x!pphlo.secret<f32>>)
    return %0 : tensor<4x!pphlo.secret<f32>>
})");
    r.verifyOutput(expected.data());
  }
}

TEST_P(ExecutorTest, Sort2DRow) {
  xt::xarray<float> op = {{2.0, 1.0, 3.0, -10.0, 11.0},  //
                          {4.0, 3.0, 2.0, 1.0, 6.0}};

  xt::xarray<float> expected = {{-10.0, 1.0, 2.0, 3.0, 11.0},  //
                                {1.0, 2.0, 3.0, 4.0, 6.0}};

  {
    Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
             std::get<2>(GetParam()));
    // Row sort
    r.addInput(op);
    r.run(R"(
func.func @main(%arg0: tensor<2x5xf32>) -> tensor<2x5xf32> {
    %0 = "pphlo.sort"(%arg0) ( {
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):  // no predecessors
      %1 = pphlo.less %arg1, %arg2 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      pphlo.return %1 : tensor<i1>
    }) {dimension = 1 : i64, is_stable = true} : (tensor<2x5xf32>) -> (tensor<2x5xf32>)
    return %0 : tensor<2x5xf32>
})");
    r.verifyOutput(expected.data());
  }

  {
    Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
             std::get<2>(GetParam()));
    // Row sort
    r.addInput(op, VIS_SECRET);
    r.run(R"(
func.func @main(%arg0: tensor<2x5x!pphlo.secret<f32>>) -> tensor<2x5x!pphlo.secret<f32>> {
    %0 = "pphlo.sort"(%arg0) ( {
    ^bb0(%arg1: tensor<!pphlo.secret<f32>>, %arg2: tensor<!pphlo.secret<f32>>):  // no predecessors
      %1 = pphlo.less %arg1, %arg2 : (tensor<!pphlo.secret<f32>>, tensor<!pphlo.secret<f32>>) -> tensor<!pphlo.secret<i1>>
      pphlo.return %1 : tensor<!pphlo.secret<i1>>
    }) {dimension = 1 : i64, is_stable = true} : (tensor<2x5x!pphlo.secret<f32>>) -> (tensor<2x5x!pphlo.secret<f32>>)
    return %0 : tensor<2x5x!pphlo.secret<f32>>
})");
    r.verifyOutput(expected.data());
  }
}

TEST_P(ExecutorTest, Sort2DCol) {
  xt::xarray<float> op = {{2.0, 1.0, 3.0, -10.0},  //
                          {4.0, 3.0, 2.0, 1.0}};
  xt::xarray<float> expected = {{2.0, 1.0, 2.0, -10.0},  //
                                {4.0, 3.0, 3.0, 1.0}};

  {
    Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
             std::get<2>(GetParam()));
    r.addInput(op);

    // Column sort
    r.run(R"(
func.func @main(%arg0: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %0 = "pphlo.sort"(%arg0) ( {
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):  // no predecessors
      %1 = pphlo.less %arg1, %arg2 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      pphlo.return %1 : tensor<i1>
    }) {dimension = 0 : i64, is_stable = true} : (tensor<2x4xf32>) -> (tensor<2x4xf32>)
    return %0 : tensor<2x4xf32>
})");

    r.verifyOutput(expected.data());
  }

  {
    Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
             std::get<2>(GetParam()));
    r.addInput(op, VIS_SECRET);

    // Column sort
    r.run(R"(
func.func @main(%arg0: tensor<2x4x!pphlo.secret<f32>>) -> tensor<2x4x!pphlo.secret<f32>> {
    %0 = "pphlo.sort"(%arg0) ( {
    ^bb0(%arg1: tensor<!pphlo.secret<f32>>, %arg2: tensor<!pphlo.secret<f32>>):  // no predecessors
      %1 = pphlo.less %arg1, %arg2 : (tensor<!pphlo.secret<f32>>, tensor<!pphlo.secret<f32>>) -> tensor<!pphlo.secret<i1>>
      pphlo.return %1 : tensor<!pphlo.secret<i1>>
    }) {dimension = 0 : i64, is_stable = true} : (tensor<2x4x!pphlo.secret<f32>>) -> (tensor<2x4x!pphlo.secret<f32>>)
    return %0 : tensor<2x4x!pphlo.secret<f32>>
})");

    r.verifyOutput(expected.data());
  }
}

TEST_P(ExecutorTest, SortMultiOperands) {
  xt::xarray<int> x = {3, 1};
  xt::xarray<int> y = {42, 50};
  xt::xarray<float> z = {-3.0, 1.5};

  xt::xarray<int> expected_x = {1, 3};
  xt::xarray<int> expected_y = {50, 42};
  xt::xarray<float> expected_z = {1.5, -3.0};

  {
    Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
             std::get<2>(GetParam()));

    r.addInput(x);
    r.addInput(y);
    r.addInput(z);

    // Row sort
    r.run(R"(
func.func @main(%arg0: tensor<2xi32>, %arg1: tensor<2xi32>, %arg2: tensor<2xf32>) -> (tensor<2xi32>, tensor<2xi32>, tensor<2xf32>) {
    %0:3 = "pphlo.sort"(%arg0, %arg1, %arg2) ( {
    ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>, %arg5: tensor<i32>, %arg6: tensor<i32>, %arg7: tensor<f32>, %arg8: tensor<f32>):  // no predecessors
      %1 = pphlo.less %arg3, %arg4 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      pphlo.return %1 : tensor<i1>
    }) {dimension = 0 : i64, is_stable = true} : (tensor<2xi32>, tensor<2xi32>, tensor<2xf32>) -> (tensor<2xi32>, tensor<2xi32>, tensor<2xf32>)
    return %0#0, %0#1, %0#2 : tensor<2xi32>, tensor<2xi32>, tensor<2xf32>
})",
          3);

    r.verifyOutput(expected_x.data(), 0);
    r.verifyOutput(expected_y.data(), 1);
    r.verifyOutput(expected_z.data(), 2);
  }

  {
    Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
             std::get<2>(GetParam()));
    r.addInput(x, VIS_SECRET);
    r.addInput(y, VIS_SECRET);
    r.addInput(z, VIS_SECRET);

    // Row sort
    r.run(R"(
func.func @main(%arg0: tensor<2x!pphlo.secret<i32>>, %arg1: tensor<2x!pphlo.secret<i32>>, %arg2: tensor<2x!pphlo.secret<f32>>) -> (tensor<2x!pphlo.secret<i32>>, tensor<2x!pphlo.secret<i32>>, tensor<2x!pphlo.secret<f32>>) {
    %0:3 = "pphlo.sort"(%arg0, %arg1, %arg2) ( {
    ^bb0(%arg3: tensor<!pphlo.secret<i32>>, %arg4: tensor<!pphlo.secret<i32>>, %arg5: tensor<!pphlo.secret<i32>>, %arg6: tensor<!pphlo.secret<i32>>, %arg7: tensor<!pphlo.secret<f32>>, %arg8: tensor<!pphlo.secret<f32>>):  // no predecessors
      %1 = pphlo.less %arg3, %arg4 : (tensor<!pphlo.secret<i32>>, tensor<!pphlo.secret<i32>>) -> tensor<!pphlo.secret<i1>>
      pphlo.return %1 : tensor<!pphlo.secret<i1>>
    }) {dimension = 0 : i64, is_stable = true} : (tensor<2x!pphlo.secret<i32>>, tensor<2x!pphlo.secret<i32>>, tensor<2x!pphlo.secret<f32>>) -> (tensor<2x!pphlo.secret<i32>>, tensor<2x!pphlo.secret<i32>>, tensor<2x!pphlo.secret<f32>>)
    return %0#0, %0#1, %0#2 : tensor<2x!pphlo.secret<i32>>, tensor<2x!pphlo.secret<i32>>, tensor<2x!pphlo.secret<f32>>
})",
          3);

    r.verifyOutput(expected_x.data(), 0);
    r.verifyOutput(expected_y.data(), 1);
    r.verifyOutput(expected_z.data(), 2);
  }
}

TEST_P(ExecutorTest, SimpleSortMultiOperands) {
  xt::xarray<int> x = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
  xt::xarray<int> y = {1, 2, 3, 6, 7, 6, 5, 2, 1, 2};

  xt::xarray<int> expected_x = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  xt::xarray<int> expected_y = {2, 1, 2, 5, 6, 7, 6, 3, 2, 1};

  // ascending direction
  {
    Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
             std::get<2>(GetParam()));

    r.addInput(x);
    r.addInput(y);

    // Row sort
    r.run(R"(
func.func @main(%arg0: tensor<10xi32>, %arg1: tensor<10xi32>) -> (tensor<10xi32>, tensor<10xi32>) {
    %0:2 = pphlo.simple_sort %arg0, %arg1 ASC, dim = 0, num_keys = 1 : (tensor<10xi32>, tensor<10xi32>) -> (tensor<10xi32>, tensor<10xi32>)
    return %0#0, %0#1 : tensor<10xi32>, tensor<10xi32>
})",
          2);

    r.verifyOutput(expected_x.data(), 0);
    r.verifyOutput(expected_y.data(), 1);
  }

  {
    Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
             std::get<2>(GetParam()));
    r.addInput(x, VIS_SECRET);
    r.addInput(y, VIS_SECRET);

    // Row sort
    r.run(R"(
func.func @main(%arg0: tensor<10x!pphlo.secret<i32>>, %arg1: tensor<10x!pphlo.secret<i32>>) -> (tensor<10x!pphlo.secret<i32>>, tensor<10x!pphlo.secret<i32>>) {
    %0:2 = pphlo.simple_sort %arg0, %arg1 ASC, dim = 0, num_keys = 1 : (tensor<10x!pphlo.secret<i32>>, tensor<10x!pphlo.secret<i32>>) -> (tensor<10x!pphlo.secret<i32>>, tensor<10x!pphlo.secret<i32>>)
    return %0#0, %0#1 : tensor<10x!pphlo.secret<i32>>, tensor<10x!pphlo.secret<i32>>
})",
          2);

    r.verifyOutput(expected_x.data(), 0);
    r.verifyOutput(expected_y.data(), 1);
  }

  // Descending direction
  expected_x = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
  expected_y = {1, 2, 3, 6, 7, 6, 5, 2, 1, 2};

  {
    Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
             std::get<2>(GetParam()));

    r.addInput(x);
    r.addInput(y);

    // Row sort
    r.run(R"(
    func.func @main(%arg0: tensor<10xi32>, %arg1: tensor<10xi32>) -> (tensor<10xi32>, tensor<10xi32>) {
        %0:2 = pphlo.simple_sort %arg0, %arg1 DES, dim = 0, num_keys = 1 : (tensor<10xi32>, tensor<10xi32>) -> (tensor<10xi32>, tensor<10xi32>)
        return %0#0, %0#1 : tensor<10xi32>, tensor<10xi32>
    })",
          2);

    r.verifyOutput(expected_x.data(), 0);
    r.verifyOutput(expected_y.data(), 1);
  }

  {
    Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
             std::get<2>(GetParam()));
    r.addInput(x, VIS_SECRET);
    r.addInput(y, VIS_SECRET);

    // Row sort
    r.run(R"(
    func.func @main(%arg0: tensor<10x!pphlo.secret<i32>>, %arg1:tensor<10x!pphlo.secret<i32>>) -> (tensor<10x!pphlo.secret<i32>>, tensor<10x!pphlo.secret<i32>>) {
        %0:2 = pphlo.simple_sort %arg0, %arg1 DES, dim = 0, num_keys = 1 : (tensor<10x!pphlo.secret<i32>>, tensor<10x!pphlo.secret<i32>>) -> (tensor<10x!pphlo.secret<i32>>, tensor<10x!pphlo.secret<i32>>)
        return %0#0, %0#1 : tensor<10x!pphlo.secret<i32>>, tensor<10x!pphlo.secret<i32>>
    })",
          2);

    r.verifyOutput(expected_x.data(), 0);
    r.verifyOutput(expected_y.data(), 1);
  }
}

TEST_P(ExecutorTest, SimpleSortMultiKeys) {
  xt::xarray<int> x = {10, 10, 8, 8, 6, 6, 4, 4, 2, 2};
  xt::xarray<float> y = {-1.0, -2.0, -3.0, -6.0, -7.0,
                         -6.0, -5.0, -2.0, -1.0, -0.0};
  xt::xarray<int> z = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  xt::xarray<int> expected_x = {2, 2, 4, 4, 6, 6, 8, 8, 10, 10};
  xt::xarray<float> expected_y = {-1.0, -0.0, -5.0, -2.0, -7.0,
                                  -6.0, -6.0, -3.0, -2.0, -1.0};
  xt::xarray<int> expected_z = {9, 10, 7, 8, 5, 6, 4, 3, 2, 1};

  // ascending direction
  {
    Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
             std::get<2>(GetParam()));

    r.addInput(x);
    r.addInput(y);
    r.addInput(z);

    // Row sort
    r.run(R"(
func.func @main(%arg0: tensor<10xi32>, %arg1: tensor<10xf32>, %arg2: tensor<10xi32>) -> (tensor<10xi32>, tensor<10xf32>, tensor<10xi32>) {
    %0:3 = pphlo.simple_sort %arg0, %arg1, %arg2 ASC, dim = 0, num_keys = 2 : (tensor<10xi32>, tensor<10xf32>, tensor<10xi32>) -> (tensor<10xi32>, tensor<10xf32>, tensor<10xi32>)
    return %0#0, %0#1, %0#2 : tensor<10xi32>, tensor<10xf32>, tensor<10xi32>
})",
          3);

    r.verifyOutput(expected_x.data(), 0);
    r.verifyOutput(expected_y.data(), 1);
    r.verifyOutput(expected_z.data(), 2);
  }

  {
    Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
             std::get<2>(GetParam()));
    r.addInput(x, VIS_SECRET);
    r.addInput(y, VIS_SECRET);
    r.addInput(z, VIS_SECRET);

    // Row sort
    r.run(R"(
func.func @main(%arg0: tensor<10x!pphlo.secret<i32>>, %arg1: tensor<10x!pphlo.secret<f32>>, %arg2: tensor<10x!pphlo.secret<i32>>) -> (tensor<10x!pphlo.secret<i32>>, tensor<10x!pphlo.secret<f32>>, tensor<10x!pphlo.secret<i32>>) {
    %0:3 = pphlo.simple_sort %arg0, %arg1, %arg2 ASC, dim = 0, num_keys = 2 : (tensor<10x!pphlo.secret<i32>>, tensor<10x!pphlo.secret<f32>>, tensor<10x!pphlo.secret<i32>>) -> (tensor<10x!pphlo.secret<i32>>, tensor<10x!pphlo.secret<f32>>, tensor<10x!pphlo.secret<i32>>)
    return %0#0, %0#1, %0#2 : tensor<10x!pphlo.secret<i32>>, tensor<10x!pphlo.secret<f32>>, tensor<10x!pphlo.secret<i32>>
})",
          3);

    r.verifyOutput(expected_x.data(), 0);
    r.verifyOutput(expected_y.data(), 1);
    r.verifyOutput(expected_z.data(), 2);
  }

  // mixed-visibility
  {
    Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
             std::get<2>(GetParam()));
    r.addInput(x);
    r.addInput(y, VIS_SECRET);
    r.addInput(z, VIS_SECRET);

    // Row sort
    r.run(R"(
func.func @main(%arg0: tensor<10xi32>, %arg1: tensor<10x!pphlo.secret<f32>>, %arg2: tensor<10x!pphlo.secret<i32>>) -> (tensor<10x!pphlo.secret<i32>>, tensor<10x!pphlo.secret<f32>>, tensor<10x!pphlo.secret<i32>>) {
    %0:3 = pphlo.simple_sort %arg0, %arg1, %arg2 ASC, dim = 0, num_keys = 2 : (tensor<10xi32>, tensor<10x!pphlo.secret<f32>>, tensor<10x!pphlo.secret<i32>>) -> (tensor<10x!pphlo.secret<i32>>, tensor<10x!pphlo.secret<f32>>, tensor<10x!pphlo.secret<i32>>)
    return %0#0, %0#1, %0#2 : tensor<10x!pphlo.secret<i32>>, tensor<10x!pphlo.secret<f32>>, tensor<10x!pphlo.secret<i32>>
})",
          3);

    r.verifyOutput(expected_x.data(), 0);
    r.verifyOutput(expected_y.data(), 1);
    r.verifyOutput(expected_z.data(), 2);
  }

  expected_x = {10, 10, 8, 8, 6, 6, 4, 4, 2, 2};
  expected_y = {-1.0, -2.0, -3.0, -6.0, -6.0, -7.0, -2.0, -5.0, -0.0, -1.0};
  expected_z = {1, 2, 3, 4, 6, 5, 8, 7, 10, 9};

  // descending direction
  {
    Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
             std::get<2>(GetParam()));

    r.addInput(x);
    r.addInput(y);
    r.addInput(z);

    // Row sort
    r.run(R"(
func.func @main(%arg0: tensor<10xi32>, %arg1: tensor<10xf32>, %arg2: tensor<10xi32>) -> (tensor<10xi32>, tensor<10xf32>, tensor<10xi32>) {
    %0:3 = pphlo.simple_sort %arg0, %arg1, %arg2 DES, dim = 0, num_keys = 2 : (tensor<10xi32>, tensor<10xf32>, tensor<10xi32>) -> (tensor<10xi32>, tensor<10xf32>, tensor<10xi32>)
    return %0#0, %0#1, %0#2 : tensor<10xi32>, tensor<10xf32>, tensor<10xi32>
})",
          3);

    r.verifyOutput(expected_x.data(), 0);
    r.verifyOutput(expected_y.data(), 1);
    r.verifyOutput(expected_z.data(), 2);
  }

  {
    Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
             std::get<2>(GetParam()));
    r.addInput(x, VIS_SECRET);
    r.addInput(y, VIS_SECRET);
    r.addInput(z, VIS_SECRET);

    // Row sort
    r.run(R"(
func.func @main(%arg0: tensor<10x!pphlo.secret<i32>>, %arg1: tensor<10x!pphlo.secret<f32>>, %arg2: tensor<10x!pphlo.secret<i32>>) -> (tensor<10x!pphlo.secret<i32>>, tensor<10x!pphlo.secret<f32>>, tensor<10x!pphlo.secret<i32>>) {
    %0:3 = pphlo.simple_sort %arg0, %arg1, %arg2 DES, dim = 0, num_keys = 2 : (tensor<10x!pphlo.secret<i32>>, tensor<10x!pphlo.secret<f32>>, tensor<10x!pphlo.secret<i32>>) -> (tensor<10x!pphlo.secret<i32>>, tensor<10x!pphlo.secret<f32>>, tensor<10x!pphlo.secret<i32>>)
    return %0#0, %0#1, %0#2 : tensor<10x!pphlo.secret<i32>>, tensor<10x!pphlo.secret<f32>>, tensor<10x!pphlo.secret<i32>>
})",
          3);

    r.verifyOutput(expected_x.data(), 0);
    r.verifyOutput(expected_y.data(), 1);
    r.verifyOutput(expected_z.data(), 2);
  }

  // mixed-visibility
  {
    Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
             std::get<2>(GetParam()));
    r.addInput(x);
    r.addInput(y, VIS_SECRET);
    r.addInput(z, VIS_SECRET);

    // Row sort
    r.run(R"(
func.func @main(%arg0: tensor<10xi32>, %arg1: tensor<10x!pphlo.secret<f32>>, %arg2: tensor<10x!pphlo.secret<i32>>) -> (tensor<10x!pphlo.secret<i32>>, tensor<10x!pphlo.secret<f32>>, tensor<10x!pphlo.secret<i32>>) {
    %0:3 = pphlo.simple_sort %arg0, %arg1, %arg2 DES, dim = 0, num_keys = 2 : (tensor<10xi32>, tensor<10x!pphlo.secret<f32>>, tensor<10x!pphlo.secret<i32>>) -> (tensor<10x!pphlo.secret<i32>>, tensor<10x!pphlo.secret<f32>>, tensor<10x!pphlo.secret<i32>>)
    return %0#0, %0#1, %0#2 : tensor<10x!pphlo.secret<i32>>, tensor<10x!pphlo.secret<f32>>, tensor<10x!pphlo.secret<i32>>
})",
          3);

    r.verifyOutput(expected_x.data(), 0);
    r.verifyOutput(expected_y.data(), 1);
    r.verifyOutput(expected_z.data(), 2);
  }
}

TEST_P(ExecutorTest, SimpleSortComplicatedMultiKeys) {
  xt::xarray<int> key0 = {10,  10,  10,  10,  10,  10,  10,  10,
                          -10, -10, -10, -10, -10, -10, -10, -10};
  xt::xarray<float> key1 = {-3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -1.0,
                            1.0,  3.0,  3.0,  3.0,  3.0,  3.0,  3.0,  3.0};
  xt::xarray<int> key2 = {-10, -10, -10, -10, -10, 8,  9,  6,
                          7,   5,   4,   10,  10,  10, 10, 10};
  xt::xarray<float> key3 = {4.0, 4.0, 4.0, 4.0, -4.0, -3.0, -2.0, -1.0,
                            0.0, 1.0, 2.0, 3.0, 4.0,  4.0,  4.0,  4.0};
  xt::xarray<int> key4 = {-10, -10, -10, 1, 2, 3,  4,  5,
                          6,   7,   8,   9, 9, 10, 10, 10};
  xt::xarray<int> key5 = {10, 10, -1, -2, -3, -4, -5, -6,
                          6,  5,  4,  3,  2,  1,  10, 10};
  xt::xarray<int> key6 = {10, 9, -1, -2, -3, -4, -5, -6,
                          6,  5, 4,  3,  2,  1,  9,  10};
  xt::xarray<int> val = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

  // ascending
  xt::xarray<int> expected_key0_asc = {-10, -10, -10, -10, -10, -10, -10, -10,
                                       10,  10,  10,  10,  10,  10,  10,  10};
  xt::xarray<float> expected_key1_asc = {1.0,  3.0,  3.0,  3.0,  3.0,  3.0,
                                         3.0,  3.0,  -3.0, -3.0, -3.0, -3.0,
                                         -3.0, -3.0, -3.0, -1.0};
  xt::xarray<int> expected_key2_asc = {7,   4,   5,   10,  10,  10, 10, 10,
                                       -10, -10, -10, -10, -10, 8,  9,  6};
  xt::xarray<float> expected_key3_asc = {0.0, 2.0,  1.0,  3.0, 4.0, 4.0,
                                         4.0, 4.0,  -4.0, 4.0, 4.0, 4.0,
                                         4.0, -3.0, -2.0, -1.0};
  xt::xarray<int> expected_key4_asc = {6, 8,   7,   9,   9, 10, 10, 10,
                                       2, -10, -10, -10, 1, 3,  4,  5};
  xt::xarray<int> expected_key5_asc = {6,  4,  5,  3,  2,  1,  10, 10,
                                       -3, -1, 10, 10, -2, -4, -5, -6};
  xt::xarray<int> expected_key6_asc = {6,  4,  5, 3,  2,  1,  9,  10,
                                       -3, -1, 9, 10, -2, -4, -5, -6};
  xt::xarray<int> expected_val_asc = {9, 11, 10, 12, 13, 14, 15, 16,
                                      5, 3,  2,  1,  4,  6,  7,  8};

  // descending
  xt::xarray<int> expected_key0_des = {10,  10,  10,  10,  10,  10,  10,  10,
                                       -10, -10, -10, -10, -10, -10, -10, -10};
  xt::xarray<float> expected_key1_des = {-1.0, -3.0, -3.0, -3.0, -3.0, -3.0,
                                         -3.0, -3.0, 3.0,  3.0,  3.0,  3.0,
                                         3.0,  3.0,  3.0,  1.0};
  xt::xarray<int> expected_key2_des = {6,  9,  8,  -10, -10, -10, -10, -10,
                                       10, 10, 10, 10,  10,  5,   4,   7};
  xt::xarray<float> expected_key3_des = {-1.0, -2.0, -3.0, 4.0, 4.0, 4.0,
                                         4.0,  -4.0, 4.0,  4.0, 4.0, 4.0,
                                         3.0,  1.0,  2.0,  0.0};
  xt::xarray<int> expected_key4_des = {5,  4,  3,  1, -10, -10, -10, 2,
                                       10, 10, 10, 9, 9,   7,   8,   6};
  xt::xarray<int> expected_key5_des = {-6, -5, -4, -2, 10, 10, -1, -3,
                                       10, 10, 1,  2,  3,  5,  4,  6};
  xt::xarray<int> expected_key6_des = {-6, -5, -4, -2, 10, 9, -1, -3,
                                       10, 9,  1,  2,  3,  5, 4,  6};
  xt::xarray<int> expected_val_des = {8,  7,  6,  4,  1,  2,  3,  5,
                                      16, 15, 14, 13, 12, 10, 11, 9};

  auto VERIFY_RESULTS = [&](Runner &r, bool is_ascending) {
    if (is_ascending) {
      r.verifyOutput(expected_key0_asc.data(), 0);
      r.verifyOutput(expected_key1_asc.data(), 1);
      r.verifyOutput(expected_key2_asc.data(), 2);
      r.verifyOutput(expected_key3_asc.data(), 3);
      r.verifyOutput(expected_key4_asc.data(), 4);
      r.verifyOutput(expected_key5_asc.data(), 5);
      r.verifyOutput(expected_key6_asc.data(), 6);
      r.verifyOutput(expected_val_asc.data(), 7);
    } else {
      r.verifyOutput(expected_key0_des.data(), 0);
      r.verifyOutput(expected_key1_des.data(), 1);
      r.verifyOutput(expected_key2_des.data(), 2);
      r.verifyOutput(expected_key3_des.data(), 3);
      r.verifyOutput(expected_key4_des.data(), 4);
      r.verifyOutput(expected_key5_des.data(), 5);
      r.verifyOutput(expected_key6_des.data(), 6);
      r.verifyOutput(expected_val_des.data(), 7);
    }
  };

  // ascending direction
  {
    Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
             std::get<2>(GetParam()));

    r.addInput(key0);
    r.addInput(key1);
    r.addInput(key2);
    r.addInput(key3);
    r.addInput(key4);
    r.addInput(key5);
    r.addInput(key6);
    r.addInput(val);

    // all public
    r.run(R"(
func.func @main(%arg0: tensor<16xi32>, %arg1: tensor<16xf32>, %arg2: tensor<16xi32>, %arg3: tensor<16xf32>, %arg4: tensor<16xi32>, %arg5: tensor<16xi32>, %arg6: tensor<16xi32>, %arg7: tensor<16xi32>) -> (tensor<16xi32>, tensor<16xf32>, tensor<16xi32>, tensor<16xf32>, tensor<16xi32>, tensor<16xi32>,tensor<16xi32>, tensor<16xi32>) {
    %0:8 = pphlo.simple_sort %arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7 ASC, dim = 0, num_keys = 7 : (tensor<16xi32>, tensor<16xf32>, tensor<16xi32>, tensor<16xf32>, tensor<16xi32>, tensor<16xi32>,tensor<16xi32>, tensor<16xi32>) -> (tensor<16xi32>, tensor<16xf32>, tensor<16xi32>, tensor<16xf32>, tensor<16xi32>, tensor<16xi32>,tensor<16xi32>, tensor<16xi32>)
    return %0#0, %0#1, %0#2, %0#3, %0#4, %0#5, %0#6, %0#7 : tensor<16xi32>, tensor<16xf32>, tensor<16xi32>, tensor<16xf32>, tensor<16xi32>, tensor<16xi32>,tensor<16xi32>, tensor<16xi32>
})",
          8);

    VERIFY_RESULTS(r, true);
  }

  // descending direction
  {
    Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
             std::get<2>(GetParam()));

    r.addInput(key0);
    r.addInput(key1);
    r.addInput(key2);
    r.addInput(key3);
    r.addInput(key4);
    r.addInput(key5);
    r.addInput(key6);
    r.addInput(val);

    // all public
    r.run(R"(
func.func @main(%arg0: tensor<16xi32>, %arg1: tensor<16xf32>, %arg2: tensor<16xi32>, %arg3: tensor<16xf32>, %arg4: tensor<16xi32>, %arg5: tensor<16xi32>, %arg6: tensor<16xi32>, %arg7: tensor<16xi32>) -> (tensor<16xi32>, tensor<16xf32>, tensor<16xi32>, tensor<16xf32>, tensor<16xi32>, tensor<16xi32>,tensor<16xi32>, tensor<16xi32>) {
    %0:8 = pphlo.simple_sort %arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7 DES, dim = 0, num_keys = 7 : (tensor<16xi32>, tensor<16xf32>, tensor<16xi32>, tensor<16xf32>, tensor<16xi32>, tensor<16xi32>,tensor<16xi32>, tensor<16xi32>) -> (tensor<16xi32>, tensor<16xf32>, tensor<16xi32>, tensor<16xf32>, tensor<16xi32>, tensor<16xi32>,tensor<16xi32>, tensor<16xi32>)
    return %0#0, %0#1, %0#2, %0#3, %0#4, %0#5, %0#6, %0#7 : tensor<16xi32>, tensor<16xf32>, tensor<16xi32>, tensor<16xf32>, tensor<16xi32>, tensor<16xi32>,tensor<16xi32>, tensor<16xi32>
})",
          8);

    VERIFY_RESULTS(r, false);
  }

  // ascending direction
  {
    Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
             std::get<2>(GetParam()));

    r.addInput(key0, VIS_SECRET, 0);
    r.addInput(key1, VIS_SECRET, 0);
    r.addInput(key2, VIS_SECRET, 0);
    r.addInput(key3, VIS_SECRET, 0);
    r.addInput(key4, VIS_SECRET, 1);
    r.addInput(key5, VIS_SECRET, 1);
    r.addInput(key6, VIS_SECRET, 1);
    r.addInput(val, VIS_SECRET);

    // all private
    r.run(R"(
func.func @main(%arg0: tensor<16x!pphlo.secret<i32>>, %arg1: tensor<16x!pphlo.secret<f32>>, %arg2: tensor<16x!pphlo.secret<i32>>, %arg3: tensor<16x!pphlo.secret<f32>>, %arg4: tensor<16x!pphlo.secret<i32>>, %arg5: tensor<16x!pphlo.secret<i32>>, %arg6: tensor<16x!pphlo.secret<i32>>, %arg7: tensor<16x!pphlo.secret<i32>>) -> (tensor<16x!pphlo.secret<i32>>, tensor<16x!pphlo.secret<f32>>, tensor<16x!pphlo.secret<i32>>, tensor<16x!pphlo.secret<f32>>, tensor<16x!pphlo.secret<i32>>, tensor<16x!pphlo.secret<i32>>,tensor<16x!pphlo.secret<i32>>, tensor<16x!pphlo.secret<i32>>) {
    %0:8 = pphlo.simple_sort %arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7 ASC, dim = 0, num_keys = 7 : (tensor<16x!pphlo.secret<i32>>, tensor<16x!pphlo.secret<f32>>, tensor<16x!pphlo.secret<i32>>, tensor<16x!pphlo.secret<f32>>, tensor<16x!pphlo.secret<i32>>, tensor<16x!pphlo.secret<i32>>,tensor<16x!pphlo.secret<i32>>, tensor<16x!pphlo.secret<i32>>) -> (tensor<16x!pphlo.secret<i32>>, tensor<16x!pphlo.secret<f32>>, tensor<16x!pphlo.secret<i32>>, tensor<16x!pphlo.secret<f32>>, tensor<16x!pphlo.secret<i32>>, tensor<16x!pphlo.secret<i32>>,tensor<16x!pphlo.secret<i32>>, tensor<16x!pphlo.secret<i32>>)
    return %0#0, %0#1, %0#2, %0#3, %0#4, %0#5, %0#6, %0#7 : tensor<16x!pphlo.secret<i32>>, tensor<16x!pphlo.secret<f32>>, tensor<16x!pphlo.secret<i32>>, tensor<16x!pphlo.secret<f32>>, tensor<16x!pphlo.secret<i32>>, tensor<16x!pphlo.secret<i32>>,tensor<16x!pphlo.secret<i32>>, tensor<16x!pphlo.secret<i32>>
})",
          8);

    VERIFY_RESULTS(r, true);
  }

  // descending direction
  {
    Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
             std::get<2>(GetParam()));

    r.addInput(key0, VIS_SECRET, 0);
    r.addInput(key1, VIS_SECRET, 0);
    r.addInput(key2, VIS_SECRET, 0);
    r.addInput(key3, VIS_SECRET, 0);
    r.addInput(key4, VIS_SECRET, 1);
    r.addInput(key5, VIS_SECRET, 1);
    r.addInput(key6, VIS_SECRET, 1);
    r.addInput(val, VIS_SECRET);

    // all private
    r.run(R"(
func.func @main(%arg0: tensor<16x!pphlo.secret<i32>>, %arg1: tensor<16x!pphlo.secret<f32>>, %arg2: tensor<16x!pphlo.secret<i32>>, %arg3: tensor<16x!pphlo.secret<f32>>, %arg4: tensor<16x!pphlo.secret<i32>>, %arg5: tensor<16x!pphlo.secret<i32>>, %arg6: tensor<16x!pphlo.secret<i32>>, %arg7: tensor<16x!pphlo.secret<i32>>) -> (tensor<16x!pphlo.secret<i32>>, tensor<16x!pphlo.secret<f32>>, tensor<16x!pphlo.secret<i32>>, tensor<16x!pphlo.secret<f32>>, tensor<16x!pphlo.secret<i32>>, tensor<16x!pphlo.secret<i32>>,tensor<16x!pphlo.secret<i32>>, tensor<16x!pphlo.secret<i32>>) {
    %0:8 = pphlo.simple_sort %arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7 DES, dim = 0, num_keys = 7 : (tensor<16x!pphlo.secret<i32>>, tensor<16x!pphlo.secret<f32>>, tensor<16x!pphlo.secret<i32>>, tensor<16x!pphlo.secret<f32>>, tensor<16x!pphlo.secret<i32>>, tensor<16x!pphlo.secret<i32>>,tensor<16x!pphlo.secret<i32>>, tensor<16x!pphlo.secret<i32>>) -> (tensor<16x!pphlo.secret<i32>>, tensor<16x!pphlo.secret<f32>>, tensor<16x!pphlo.secret<i32>>, tensor<16x!pphlo.secret<f32>>, tensor<16x!pphlo.secret<i32>>, tensor<16x!pphlo.secret<i32>>,tensor<16x!pphlo.secret<i32>>, tensor<16x!pphlo.secret<i32>>)
    return %0#0, %0#1, %0#2, %0#3, %0#4, %0#5, %0#6, %0#7 : tensor<16x!pphlo.secret<i32>>, tensor<16x!pphlo.secret<f32>>, tensor<16x!pphlo.secret<i32>>, tensor<16x!pphlo.secret<f32>>, tensor<16x!pphlo.secret<i32>>, tensor<16x!pphlo.secret<i32>>,tensor<16x!pphlo.secret<i32>>, tensor<16x!pphlo.secret<i32>>
})",
          8);

    VERIFY_RESULTS(r, false);
  }

  // ascending direction
  {
    Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
             std::get<2>(GetParam()));

    r.addInput(key0, VIS_SECRET, 0);
    r.addInput(key1, VIS_PUBLIC, 0);
    r.addInput(key2, VIS_SECRET, 0);
    r.addInput(key3, VIS_SECRET, 0);
    r.addInput(key4, VIS_SECRET, 1);
    r.addInput(key5, VIS_PUBLIC, 1);
    r.addInput(key6, VIS_SECRET, 1);
    r.addInput(val, VIS_SECRET);

    // mixed visibility
    r.run(R"(
func.func @main(%arg0: tensor<16x!pphlo.secret<i32>>, %arg1: tensor<16xf32>, %arg2: tensor<16x!pphlo.secret<i32>>, %arg3: tensor<16x!pphlo.secret<f32>>, %arg4: tensor<16x!pphlo.secret<i32>>, %arg5: tensor<16xi32>, %arg6: tensor<16x!pphlo.secret<i32>>, %arg7: tensor<16x!pphlo.secret<i32>>) -> (tensor<16x!pphlo.secret<i32>>, tensor<16x!pphlo.secret<f32>>, tensor<16x!pphlo.secret<i32>>, tensor<16x!pphlo.secret<f32>>, tensor<16x!pphlo.secret<i32>>, tensor<16x!pphlo.secret<i32>>,tensor<16x!pphlo.secret<i32>>, tensor<16x!pphlo.secret<i32>>) {
    %0:8 = pphlo.simple_sort %arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7 ASC, dim = 0, num_keys = 7 : (tensor<16x!pphlo.secret<i32>>, tensor<16xf32>, tensor<16x!pphlo.secret<i32>>, tensor<16x!pphlo.secret<f32>>, tensor<16x!pphlo.secret<i32>>, tensor<16xi32>,tensor<16x!pphlo.secret<i32>>, tensor<16x!pphlo.secret<i32>>) -> (tensor<16x!pphlo.secret<i32>>, tensor<16x!pphlo.secret<f32>>, tensor<16x!pphlo.secret<i32>>, tensor<16x!pphlo.secret<f32>>, tensor<16x!pphlo.secret<i32>>, tensor<16x!pphlo.secret<i32>>,tensor<16x!pphlo.secret<i32>>, tensor<16x!pphlo.secret<i32>>)
    return %0#0, %0#1, %0#2, %0#3, %0#4, %0#5, %0#6, %0#7 : tensor<16x!pphlo.secret<i32>>, tensor<16x!pphlo.secret<f32>>, tensor<16x!pphlo.secret<i32>>, tensor<16x!pphlo.secret<f32>>, tensor<16x!pphlo.secret<i32>>, tensor<16x!pphlo.secret<i32>>,tensor<16x!pphlo.secret<i32>>, tensor<16x!pphlo.secret<i32>>
})",
          8);

    VERIFY_RESULTS(r, true);
  }

  // descending direction
  {
    Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
             std::get<2>(GetParam()));

    r.addInput(key0, VIS_SECRET, 0);
    r.addInput(key1, VIS_PUBLIC, 0);
    r.addInput(key2, VIS_SECRET, 0);
    r.addInput(key3, VIS_SECRET, 0);
    r.addInput(key4, VIS_SECRET, 1);
    r.addInput(key5, VIS_PUBLIC, 1);
    r.addInput(key6, VIS_SECRET, 1);
    r.addInput(val, VIS_SECRET);

    // mixed visibility
    r.run(R"(
func.func @main(%arg0: tensor<16x!pphlo.secret<i32>>, %arg1: tensor<16xf32>, %arg2: tensor<16x!pphlo.secret<i32>>, %arg3: tensor<16x!pphlo.secret<f32>>, %arg4: tensor<16x!pphlo.secret<i32>>, %arg5: tensor<16xi32>, %arg6: tensor<16x!pphlo.secret<i32>>, %arg7: tensor<16x!pphlo.secret<i32>>) -> (tensor<16x!pphlo.secret<i32>>, tensor<16x!pphlo.secret<f32>>, tensor<16x!pphlo.secret<i32>>, tensor<16x!pphlo.secret<f32>>, tensor<16x!pphlo.secret<i32>>, tensor<16x!pphlo.secret<i32>>,tensor<16x!pphlo.secret<i32>>, tensor<16x!pphlo.secret<i32>>) {
    %0:8 = pphlo.simple_sort %arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7 DES, dim = 0, num_keys = 7 : (tensor<16x!pphlo.secret<i32>>, tensor<16xf32>, tensor<16x!pphlo.secret<i32>>, tensor<16x!pphlo.secret<f32>>, tensor<16x!pphlo.secret<i32>>, tensor<16xi32>,tensor<16x!pphlo.secret<i32>>, tensor<16x!pphlo.secret<i32>>) -> (tensor<16x!pphlo.secret<i32>>, tensor<16x!pphlo.secret<f32>>, tensor<16x!pphlo.secret<i32>>, tensor<16x!pphlo.secret<f32>>, tensor<16x!pphlo.secret<i32>>, tensor<16x!pphlo.secret<i32>>,tensor<16x!pphlo.secret<i32>>, tensor<16x!pphlo.secret<i32>>)
    return %0#0, %0#1, %0#2, %0#3, %0#4, %0#5, %0#6, %0#7 : tensor<16x!pphlo.secret<i32>>, tensor<16x!pphlo.secret<f32>>, tensor<16x!pphlo.secret<i32>>, tensor<16x!pphlo.secret<f32>>, tensor<16x!pphlo.secret<i32>>, tensor<16x!pphlo.secret<i32>>,tensor<16x!pphlo.secret<i32>>, tensor<16x!pphlo.secret<i32>>
})",
          8);

    VERIFY_RESULTS(r, false);
  }

  // ascending direction
  {
    Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
             std::get<2>(GetParam()));

    r.addInput(key0, VIS_SECRET, 0);
    r.addInput(key1, VIS_PUBLIC, 0);
    r.addInput(key2, VIS_SECRET);
    r.addInput(key3, VIS_SECRET, 0);
    r.addInput(key4, VIS_SECRET, 1);
    r.addInput(key5, VIS_PUBLIC, 1);
    r.addInput(key6, VIS_SECRET);
    r.addInput(val, VIS_SECRET);

    // mixed visibility
    r.run(R"(
func.func @main(%arg0: tensor<16x!pphlo.secret<i32>>, %arg1: tensor<16xf32>, %arg2: tensor<16x!pphlo.secret<i32>>, %arg3: tensor<16x!pphlo.secret<f32>>, %arg4: tensor<16x!pphlo.secret<i32>>, %arg5: tensor<16xi32>, %arg6: tensor<16x!pphlo.secret<i32>>, %arg7: tensor<16x!pphlo.secret<i32>>) -> (tensor<16x!pphlo.secret<i32>>, tensor<16x!pphlo.secret<f32>>, tensor<16x!pphlo.secret<i32>>, tensor<16x!pphlo.secret<f32>>, tensor<16x!pphlo.secret<i32>>, tensor<16x!pphlo.secret<i32>>,tensor<16x!pphlo.secret<i32>>, tensor<16x!pphlo.secret<i32>>) {
    %0:8 = pphlo.simple_sort %arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7 ASC, dim = 0, num_keys = 7 : (tensor<16x!pphlo.secret<i32>>, tensor<16xf32>, tensor<16x!pphlo.secret<i32>>, tensor<16x!pphlo.secret<f32>>, tensor<16x!pphlo.secret<i32>>, tensor<16xi32>,tensor<16x!pphlo.secret<i32>>, tensor<16x!pphlo.secret<i32>>) -> (tensor<16x!pphlo.secret<i32>>, tensor<16x!pphlo.secret<f32>>, tensor<16x!pphlo.secret<i32>>, tensor<16x!pphlo.secret<f32>>, tensor<16x!pphlo.secret<i32>>, tensor<16x!pphlo.secret<i32>>,tensor<16x!pphlo.secret<i32>>, tensor<16x!pphlo.secret<i32>>)
    return %0#0, %0#1, %0#2, %0#3, %0#4, %0#5, %0#6, %0#7 : tensor<16x!pphlo.secret<i32>>, tensor<16x!pphlo.secret<f32>>, tensor<16x!pphlo.secret<i32>>, tensor<16x!pphlo.secret<f32>>, tensor<16x!pphlo.secret<i32>>, tensor<16x!pphlo.secret<i32>>,tensor<16x!pphlo.secret<i32>>, tensor<16x!pphlo.secret<i32>>
})",
          8);

    VERIFY_RESULTS(r, true);
  }

  // descending direction
  {
    Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
             std::get<2>(GetParam()));

    r.addInput(key0, VIS_SECRET, 0);
    r.addInput(key1, VIS_PUBLIC, 0);
    r.addInput(key2, VIS_SECRET);
    r.addInput(key3, VIS_SECRET, 0);
    r.addInput(key4, VIS_SECRET, 1);
    r.addInput(key5, VIS_PUBLIC, 1);
    r.addInput(key6, VIS_SECRET);
    r.addInput(val, VIS_SECRET);

    // mixed visibility
    r.run(R"(
func.func @main(%arg0: tensor<16x!pphlo.secret<i32>>, %arg1: tensor<16xf32>, %arg2: tensor<16x!pphlo.secret<i32>>, %arg3: tensor<16x!pphlo.secret<f32>>, %arg4: tensor<16x!pphlo.secret<i32>>, %arg5: tensor<16xi32>, %arg6: tensor<16x!pphlo.secret<i32>>, %arg7: tensor<16x!pphlo.secret<i32>>) -> (tensor<16x!pphlo.secret<i32>>, tensor<16x!pphlo.secret<f32>>, tensor<16x!pphlo.secret<i32>>, tensor<16x!pphlo.secret<f32>>, tensor<16x!pphlo.secret<i32>>, tensor<16x!pphlo.secret<i32>>,tensor<16x!pphlo.secret<i32>>, tensor<16x!pphlo.secret<i32>>) {
    %0:8 = pphlo.simple_sort %arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7 DES, dim = 0, num_keys = 7 : (tensor<16x!pphlo.secret<i32>>, tensor<16xf32>, tensor<16x!pphlo.secret<i32>>, tensor<16x!pphlo.secret<f32>>, tensor<16x!pphlo.secret<i32>>, tensor<16xi32>,tensor<16x!pphlo.secret<i32>>, tensor<16x!pphlo.secret<i32>>) -> (tensor<16x!pphlo.secret<i32>>, tensor<16x!pphlo.secret<f32>>, tensor<16x!pphlo.secret<i32>>, tensor<16x!pphlo.secret<f32>>, tensor<16x!pphlo.secret<i32>>, tensor<16x!pphlo.secret<i32>>,tensor<16x!pphlo.secret<i32>>, tensor<16x!pphlo.secret<i32>>)
    return %0#0, %0#1, %0#2, %0#3, %0#4, %0#5, %0#6, %0#7 : tensor<16x!pphlo.secret<i32>>, tensor<16x!pphlo.secret<f32>>, tensor<16x!pphlo.secret<i32>>, tensor<16x!pphlo.secret<f32>>, tensor<16x!pphlo.secret<i32>>, tensor<16x!pphlo.secret<i32>>,tensor<16x!pphlo.secret<i32>>, tensor<16x!pphlo.secret<i32>>
})",
          8);

    VERIFY_RESULTS(r, false);
  }
}

TEST_P(ExecutorTest, SortComplicatedComparator) {
  xt::xarray<int> x = {3, 1, 4, 2};
  xt::xarray<int> y = {42, 50, 49, 47};
  xt::xarray<int> expected_x = {3, 2, 1, 4};
  xt::xarray<int> expected_y = {42, 47, 50, 49};

  {
    Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
             std::get<2>(GetParam()));

    r.addInput(x);
    r.addInput(y);

    // Row sort
    r.run(R"(
func.func @main(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> (tensor<4xi32>, tensor<4xi32>) {
    %0:2 = "pphlo.sort"(%arg0, %arg1) ( {
    ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>, %arg4: tensor<i32>, %arg5: tensor<i32>):  // no predecessors
      %1 = pphlo.add %arg2, %arg4 : tensor<i32>
      %2 = pphlo.add %arg3, %arg5 : tensor<i32>
      %3 = pphlo.less %1, %2 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      pphlo.return %3 : tensor<i1>
    }) {dimension = 0 : i64, is_stable = true} : (tensor<4xi32>, tensor<4xi32>) -> (tensor<4xi32>, tensor<4xi32>)
    return %0#0, %0#1 : tensor<4xi32>, tensor<4xi32>
})",
          2);

    r.verifyOutput(expected_x.data(), 0);
    r.verifyOutput(expected_y.data(), 1);
  }

  {
    Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
             std::get<2>(GetParam()));

    r.addInput(x, VIS_SECRET);
    r.addInput(y, VIS_SECRET);

    // Row sort
    r.run(R"(
func.func @main(%arg0: tensor<4x!pphlo.secret<i32>>, %arg1: tensor<4x!pphlo.secret<i32>>) -> (tensor<4x!pphlo.secret<i32>>, tensor<4x!pphlo.secret<i32>>) {
    %0:2 = "pphlo.sort"(%arg0, %arg1) ( {
    ^bb0(%arg2: tensor<!pphlo.secret<i32>>, %arg3: tensor<!pphlo.secret<i32>>, %arg4: tensor<!pphlo.secret<i32>>, %arg5: tensor<!pphlo.secret<i32>>):  // no predecessors
      %1 = pphlo.add %arg2, %arg4 : tensor<!pphlo.secret<i32>>
      %2 = pphlo.add %arg3, %arg5 : tensor<!pphlo.secret<i32>>
      %3 = pphlo.less %1, %2 : (tensor<!pphlo.secret<i32>>, tensor<!pphlo.secret<i32>>) -> tensor<!pphlo.secret<i1>>
      pphlo.return %3 : tensor<!pphlo.secret<i1>>
    }) {dimension = 0 : i64, is_stable = true} : (tensor<4x!pphlo.secret<i32>>, tensor<4x!pphlo.secret<i32>>) -> (tensor<4x!pphlo.secret<i32>>, tensor<4x!pphlo.secret<i32>>)
    return %0#0, %0#1 : tensor<4x!pphlo.secret<i32>>, tensor<4x!pphlo.secret<i32>>
})",
          2);

    r.verifyOutput(expected_x.data(), 0);
    r.verifyOutput(expected_y.data(), 1);
  }
}

TEST_P(ExecutorTest, RemainderFxp) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));
  r.addInput(std::vector<float>{2.5, 18.5, 5.3});
  r.addInput(std::vector<float>{5.0, 4.2, 2.0});

  r.run(R"(
func.func @main(%arg0: tensor<3xf32>, %arg1: tensor<3xf32>) -> (tensor<3xf32>) {
  %0 = pphlo.remainder %arg0, %arg1 : tensor<3xf32>
  return %0 : tensor<3xf32>
})");

  std::vector<float> expected{2.5, 1.7, 1.3};
  r.verifyOutput(expected.data());
}

TEST_P(ExecutorTest, RemainderInt) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));
  r.addInput(std::vector<int>{5, -5, 17, -17, 5, -5, 17, -17});
  r.addInput(std::vector<int>{2, 2, 3, 3, -2, -2, -3, -3});

  r.run(R"(
func.func @main(%arg0: tensor<8xi32>, %arg1: tensor<8xi32>) -> (tensor<8xi32>) {
  %0 = pphlo.remainder %arg0, %arg1 : tensor<8xi32>
  return %0 : tensor<8xi32>
})");

  std::vector<int> expected{1, -1, 2, -2, 1, -1, 2, -2};
  r.verifyOutput(expected.data());
}

TEST_P(ExecutorTest, ShiftLeftS32) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  r.addInput(std::vector<int32_t>{static_cast<int32_t>(0x12345678),
                                  static_cast<int32_t>(0xF0001000), 1, 3, 77, 1,
                                  -3, 77});
  r.addInput(std::vector<int32_t>{4, 8, 2, 7, 15, 32, 100, -1});

  r.run(R"(
func.func @main(%arg0: tensor<8xi32>, %arg1: tensor<8xi32>) -> (tensor<8xi32>) {
  %0 = pphlo.shift_left %arg0, %arg1 : tensor<8xi32>
  return %0 : tensor<8xi32>
})");

  std::vector<int32_t> expected{static_cast<int32_t>(0x23456780),
                                0x00100000,
                                0x4,
                                0x180,
                                2523136,
                                0,
                                0,
                                0};
  r.verifyOutput(expected.data());
}

TEST_P(ExecutorTest, ShiftLeftU32) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  r.addInput(
      std::vector<uint32_t>{0x12345678, 0xF0001000, 1, 3, 77, 1, ~3U, 77});
  r.addInput(std::vector<uint32_t>{4, 8, 2, 7, 15, 32, 100, ~0U});

  r.run(R"(
func.func @main(%arg0: tensor<8xui32>, %arg1: tensor<8xui32>) -> (tensor<8xui32>) {
  %0 = pphlo.shift_left %arg0, %arg1 : tensor<8xui32>
  return %0 : tensor<8xui32>
})");

  std::vector<uint32_t> expected{0x23456780, 0x00100000, 0x4, 0x180,
                                 2523136,    0,          0,   0};
  r.verifyOutput(expected.data());
}

TEST_P(ExecutorTest, ShiftRightLogicalS32) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  r.addInput(std::vector<int32_t>{static_cast<int32_t>(0x92345678),
                                  static_cast<int32_t>(0x10001000), 1, 3, 77, 1,
                                  -3, 77});
  r.addInput(std::vector<int32_t>{4, 8, 2, 7, 5, 32, /*100*/ 0, -1});

  r.run(R"(
func.func @main(%arg0: tensor<8xi32>, %arg1: tensor<8xi32>) -> (tensor<8xi32>) {
  %0 = pphlo.shift_right_logical %arg0, %arg1 : tensor<8xi32>
  return %0 : tensor<8xi32>
})");

  std::vector<int32_t> expected{
      static_cast<int>(0xF9234567), 0x00100010, 0, 0, 2, 0, -3, 0};
  r.verifyOutput(expected.data());
}

TEST_P(ExecutorTest, ShiftRightLogicalU32) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  r.addInput(
      std::vector<uint32_t>{0x92345678, 0x10001000, 1, 3, 77, 1, ~3U, 77});
  r.addInput(std::vector<uint32_t>{4, 8, 2, 7, 5, 32, /*100*/ 0, ~0U});

  r.run(R"(
func.func @main(%arg0: tensor<8xui32>, %arg1: tensor<8xui32>) -> (tensor<8xui32>) {
  %0 = pphlo.shift_right_logical %arg0, %arg1 : tensor<8xui32>
  return %0 : tensor<8xui32>
})");

  std::vector<uint32_t> expected{0x09234567, 0x00100010, 0, 0, 2, 0, ~3U, 0};
  r.verifyOutput(expected.data());
}

TEST_P(ExecutorTest, ShiftRightArithmeticS32) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  r.addInput(std::vector<int32_t>{static_cast<int32_t>(0x92345678),
                                  static_cast<int32_t>(0x10001000), 1, 3, 77, 1,
                                  -3, 77});
  r.addInput(std::vector<int32_t>{4, 8, 2, 7, 2, 32, /*100*/ 0, -1});

  r.run(R"(
func.func @main(%arg0: tensor<8xi32>, %arg1: tensor<8xi32>) -> (tensor<8xi32>) {
  %0 = pphlo.shift_right_arithmetic %arg0, %arg1 : tensor<8xi32>
  return %0 : tensor<8xi32>
})");

  std::vector<int32_t> expected{static_cast<int32_t>(0xF9234567),
                                static_cast<int32_t>(0x00100010),
                                0,
                                0,
                                19,
                                0,
                                -3,
                                0};
  r.verifyOutput(expected.data());
}

TEST_P(ExecutorTest, ShiftRightArithmeticU32) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  r.addInput(
      std::vector<uint32_t>{0x92345678, 0x10001000, 1, 3, 77, 1, ~3U, 77});
  r.addInput(std::vector<uint32_t>{4, 8, 2, 7, 2, 32, /*100*/ 0, ~0U});

  r.run(R"(
func.func @main(%arg0: tensor<8xui32>, %arg1: tensor<8xui32>) -> (tensor<8xui32>) {
  %0 = pphlo.shift_right_arithmetic %arg0, %arg1 : tensor<8xui32>
  return %0 : tensor<8xui32>
})");

  std::vector<uint32_t> expected{0x09234567, 0x00100010, 0, 0, 19, 0, ~3U, 0};
  r.verifyOutput(expected.data());
}

TEST_P(ExecutorTest, ARShift_Secret) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  r.addInput(
      std::vector<uint32_t>{0x92345678, 0x10001000, 1, 3, 77, 1, ~3U, 77},
      VIS_SECRET);
  r.addInput(std::vector<uint32_t>{4, 8, 2, 7, 2, 32, /*100*/ 0, ~0U},
             VIS_SECRET);

  r.run(R"(
func.func @main(%arg0: tensor<8x!pphlo.secret<ui32>>, %arg1: tensor<8x!pphlo.secret<ui32>>) -> (tensor<8x!pphlo.secret<ui32>>) {
  %0 = pphlo.shift_right_arithmetic %arg0, %arg1 : tensor<8x!pphlo.secret<ui32>>
  return %0 : tensor<8x!pphlo.secret<ui32>>
})");

  std::vector<uint32_t> expected{0x09234567, 0x00100010, 0, 0, 19, 0, ~3U, 0};
  r.verifyOutput(expected.data());
}

TEST_P(ExecutorTest, DotGeneral) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  std::vector<int32_t> x(3L * 4, 0);
  std::iota(x.begin(), x.end(), 0);
  r.addInput(x);

  std::vector<int32_t> y(3L * 4 * 5, 0);
  std::iota(y.begin(), y.end(), 0);
  r.addInput(y);

  r.run(R"(
func.func @main(%arg0: tensor<12xi32>, %arg1: tensor<60xi32>) -> (tensor<3x5xi32>) {
  %0 = pphlo.reshape %arg0 : (tensor<12xi32>) -> tensor<3x1x4xi32>
  %1 = pphlo.reshape %arg1 : (tensor<60xi32>) -> tensor<3x4x5xi32>
  %2 = pphlo.dot_general %0, %1, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<3x1x4xi32>, tensor<3x4x5xi32>) -> tensor<3x5xi32>
  return %2 : tensor<3x5xi32>
})");

  std::vector<int32_t> expected{70,  76,  82,   88,   94,   630,  652, 674,
                                696, 718, 1830, 1868, 1906, 1944, 1982};

  r.verifyOutput(expected.data());
}

TEST_P(ExecutorTest, SelectAndScatter1) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  r.addInput(xt::xarray<int32_t>{
      {7, 2, 5, 3, 10, 2},  //
      {3, 8, 9, 3, 4, 2},   //
      {1, 5, 7, 5, 6, 1},   //
      {0, 6, 2, 7, 2, 8}    //
  });
  r.addInput(xt::xarray<int32_t>{
      {2, 6},  //
      {3, 1}   //
  });
  r.addInput(static_cast<int32_t>(0));

  r.run(R"(
func.func @main(%arg0: tensor<4x6xi32>, %arg1: tensor<2x2xi32>, %arg2: tensor<i32>) -> (tensor<4x6xi32>) {
  %0 = "pphlo.select_and_scatter"(%arg0, %arg1, %arg2) ({
    ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):
      %1 = pphlo.less %arg3, %arg4 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %2 = pphlo.not %1 : tensor<i1>
      pphlo.return %2 : tensor<i1>
    }, {
    ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):
      %1 = pphlo.add %arg3, %arg4 : tensor<i32>
      pphlo.return %1 : tensor<i32>
    }) {
      window_dimensions = array<i64: 2,3>,
      window_strides = array<i64: 2,3>
    } : (tensor<4x6xi32>, tensor<2x2xi32>, tensor<i32>) -> tensor<4x6xi32>
    return %0 : tensor<4x6xi32>
})");

  xt::xarray<int32_t> expected = {{0, 0, 0, 0, 6, 0},  //
                                  {0, 0, 2, 0, 0, 0},  //
                                  {0, 0, 3, 0, 0, 0},  //
                                  {0, 0, 0, 0, 0, 1}};
  r.verifyOutput(expected.data());
}

TEST_P(ExecutorTest, SelectAndScatter2) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  r.addInput(xt::xarray<int32_t>{
      {7, 2, 5, 3, 8},  //
      {3, 8, 9, 3, 4},  //
      {1, 5, 7, 5, 6},  //
      {0, 6, 2, 10, 2}  //
  });
  r.addInput(xt::xarray<int32_t>{
      {2, 6},  //
      {3, 1}   //
  });
  r.addInput(static_cast<int32_t>(0));

  r.run(R"(
func.func @main(%arg0: tensor<4x5xi32>, %arg1: tensor<2x2xi32>, %arg2: tensor<i32>) -> (tensor<4x5xi32>) {
  %0 = "pphlo.select_and_scatter"(%arg0, %arg1, %arg2) ({
    ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):
      %1 = pphlo.less %arg3, %arg4 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %2 = pphlo.not %1 : tensor<i1>
      pphlo.return %2 : tensor<i1>
    }, {
    ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):
      %1 = pphlo.add %arg3, %arg4 : tensor<i32>
      pphlo.return %1 : tensor<i32>
    }) {
      window_dimensions = array<i64:2,3>,
      window_strides = array<i64:2,2>
    } : (tensor<4x5xi32>, tensor<2x2xi32>, tensor<i32>) -> tensor<4x5xi32>
    return %0 : tensor<4x5xi32>
})");

  xt::xarray<int32_t> expected = {{0, 0, 0, 0, 0},  //
                                  {0, 0, 8, 0, 0},  //
                                  {0, 0, 3, 0, 0},  //
                                  {0, 0, 0, 1, 0}};
  r.verifyOutput(expected.data());
}

TEST_P(ExecutorTest, MaxPoolScatter1) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  r.addInput(xt::xarray<bool>{
      {{0, 0, 0, 0, 0, 1}, {0, 1, 0, 0, 0, 0}},  //
      {{0, 0, 1, 0, 0, 0}, {0, 0, 0, 0, 0, 1}},  //
  });

  r.addInput(xt::xarray<int32_t>{
      {2, 6},  //
      {3, 1}   //
  });

  r.run(R"(
func.func @main(%arg0: tensor<2x2x6xi1>, %arg1: tensor<2x2xi32>) -> (tensor<4x6xi32>) {
    %0 = pphlo.maxpool_scatter %arg0, %arg1 {window_dimensions = array<i64:2,3>, window_strides = array<i64:2,3>} : (tensor<2x2x6xi1>, tensor<2x2xi32>) -> tensor<4x6xi32>
    return %0 : tensor<4x6xi32>
})");

  xt::xarray<int32_t> expected = {{0, 0, 0, 0, 6, 0},  //
                                  {0, 0, 2, 0, 0, 0},  //
                                  {0, 0, 3, 0, 0, 0},  //
                                  {0, 0, 0, 0, 0, 1}};
  r.verifyOutput(expected.data());
}

TEST_P(ExecutorTest, MaxPoolScatter2) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  r.addInput(xt::xarray<bool>{
      {{0, 0, 0, 0, 0, 1}, {0, 0, 0, 1, 0, 0}},  //
      {{0, 0, 1, 0, 0, 0}, {0, 0, 0, 0, 1, 0}},  //
  });

  r.addInput(xt::xarray<int32_t>{
      {2, 6},  //
      {3, 1}   //
  });

  r.run(R"(
func.func @main(%arg0: tensor<2x2x6xi1>, %arg1: tensor<2x2xi32>) -> (tensor<4x5xi32>) {
    %0 = pphlo.maxpool_scatter %arg0, %arg1 {window_dimensions = array<i64:2,3>, window_strides = array<i64:2,2>} : (tensor<2x2x6xi1>, tensor<2x2xi32>) -> tensor<4x5xi32>
    return %0 : tensor<4x5xi32>
})");

  xt::xarray<int32_t> expected = {{0, 0, 0, 0, 0},  //
                                  {0, 0, 8, 0, 0},  //
                                  {0, 0, 3, 0, 0},  //
                                  {0, 0, 0, 1, 0}};
  r.verifyOutput(expected.data());
}

TEST_P(ExecutorTest, MaxPoolReduce1) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  r.addInput(xt::xarray<int32_t>{
      {7, 2, 5, 3, 10, 2},  //
      {3, 8, 9, 3, 4, 2},   //
      {1, 5, 7, 5, 6, 1},   //
      {0, 6, 2, 7, 2, 8}    //
  });

  r.run(R"(
func.func @main(%arg0: tensor<4x6xi32>) -> (tensor<2x2xi32>, tensor<2x2x6xi1>) {
    %4:2 = pphlo.argmax %arg0 {
      window_dilations = array<i64: 1, 1>,
      window_dimensions = array<i64:2, 3>,
      window_strides = array<i64:2, 3>
    } : (tensor<4x6xi32>) -> (tensor<2x2xi32>, tensor<2x2x6xi1>)
    return %4#0, %4#1: tensor<2x2xi32>, tensor<2x2x6xi1>
})",
        2);

  xt::xarray<int32_t> reduce_ret = {{9, 10},  //
                                    {7, 8}};
  r.verifyOutput(reduce_ret.data(), 0);

  xt::xarray<uint8_t> mask = {
      {{0, 0, 0, 0, 0, 1}, {0, 1, 0, 0, 0, 0}},  //
      {{0, 0, 1, 0, 0, 0}, {0, 0, 0, 0, 0, 1}},  //
  };

  r.verifyOutput(mask.data(), 1);
}

TEST_P(ExecutorTest, MaxPoolReduce2) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  r.addInput(xt::xarray<int32_t>{
      {7, 2, 5, 3, 8},  //
      {3, 8, 9, 3, 4},  //
      {1, 5, 7, 5, 6},  //
      {0, 6, 2, 10, 2}  //
  });

  r.run(R"(
func.func @main(%arg0: tensor<4x5xi32>) -> (tensor<2x2xi32>, tensor<2x2x6xi1>) {
    %4:2 = pphlo.argmax %arg0 {
      window_dilations = array<i64: 1,1>,
      window_dimensions = array<i64:2, 3>,
      window_strides = array<i64:2, 2>
    } : (tensor<4x5xi32>) -> (tensor<2x2xi32>, tensor<2x2x6xi1>)
    return %4#0, %4#1: tensor<2x2xi32>, tensor<2x2x6xi1>
})",
        2);

  xt::xarray<int32_t> reduce_ret = {{9, 9},  //
                                    {7, 10}};
  r.verifyOutput(reduce_ret.data(), 0);

  xt::xarray<int8_t> mask = {
      {{0, 0, 0, 0, 0, 1}, {0, 0, 0, 1, 0, 0}},  //
      {{0, 0, 1, 0, 0, 0}, {0, 0, 0, 0, 1, 0}},  //
  };

  r.verifyOutput(mask.data(), 1);
}

TEST_P(ExecutorTest, OptimizedMaxPool1) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  r.addInput(xt::xarray<int32_t>{
      {7, 2, 5, 3, 10, 2},  //
      {3, 8, 9, 3, 4, 2},   //
      {1, 5, 7, 5, 6, 1},   //
      {0, 6, 2, 7, 2, 8}    //
  });
  r.addInput(xt::xarray<int32_t>{
      {2, 6},  //
      {3, 1}   //
  });

  auto ir = r.compileMHlo(R"(
func.func @main(%arg0: tensor<4x6xi32>, %arg1: tensor<2x2xi32>) -> (tensor<2x2xi32>, tensor<4x6xi32>) {
  %0 = stablehlo.constant dense<0> : tensor<i32>
  %1 = "stablehlo.reduce_window"(%arg0, %0) ({
    ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>):
      %3 = stablehlo.maximum %arg2, %arg3 : tensor<i32>
      stablehlo.return %3 : tensor<i32>
    }) {
      base_dilations = array<i64: 1, 1>,
      padding = dense<0> : tensor<2x2xi64>,
      window_dilations = array<i64: 1, 1>,
      window_dimensions = array<i64: 2, 3>,
      window_strides = array<i64: 2, 3>
    } : (tensor<4x6xi32>, tensor<i32>) -> tensor<2x2xi32>
  %2 = "stablehlo.select_and_scatter"(%arg0, %arg1, %0) ({
    ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):
      %3 = stablehlo.compare GE, %arg3, %arg4 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %3 : tensor<i1>
    }, {
    ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):
      %3 = stablehlo.add %arg3, %arg4 : tensor<i32>
      stablehlo.return %3 : tensor<i32>
    }) {
      padding = dense<0> : tensor<2x2xi64>,
      window_dimensions = array<i64: 2,3>,
      window_strides = array<i64: 2,3>
    } : (tensor<4x6xi32>, tensor<2x2xi32>, tensor<i32>) -> tensor<4x6xi32>
    return %1, %2 : tensor<2x2xi32>, tensor<4x6xi32>
})",
                          {VIS_PUBLIC, VIS_PUBLIC});

  EXPECT_THAT(ir, testing::HasSubstr("pphlo.maxpool_scatter"));

  r.run(ir, 2);

  xt::xarray<int32_t> reduce_ret = {{9, 10},  //
                                    {7, 8}};
  r.verifyOutput(reduce_ret.data(), 0);

  xt::xarray<int32_t> mask = {
      {{0, 0, 0, 0, 6, 0}, {0, 0, 2, 0, 0, 0}},  //
      {{0, 0, 3, 0, 0, 0}, {0, 0, 0, 0, 0, 1}},  //
  };

  r.verifyOutput(mask.data(), 1);
}

TEST_P(ExecutorTest, MaxPoolReduce3) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  {
    xt::xarray<int32_t> in = xt::reshape_view(
        xt::xarray<int32_t>{
            {7, 2, 5, 3},  //
            {3, 8, 9, 3},  //
            {1, 5, 7, 5},  //
            {0, 6, 2, 7}   //
        },
        {1, 4, 4, 1});

    r.addInput(in);
  }

  {
    xt::xarray<int32_t> in = xt::reshape_view(
        xt::xarray<int32_t>{
            {10, 11, 12},  //
            {13, 14, 15},  //
            {16, 17, 18},  //
        },
        {1, 3, 3, 1});

    r.addInput(in);
  }

  r.run(R"(
func.func @main(%arg0: tensor<1x4x4x1xi32>, %arg1: tensor<1x3x3x1xi32>) -> (tensor<1x3x3x1xi32>, tensor<1x3x3x1x4xi1>, tensor<1x4x4x1xi32>) {
    %0:2 = pphlo.argmax %arg0 {
      window_dilations = array<i64:1,1,1,1>,
      window_dimensions = array<i64:1, 2, 2, 1>,
      window_strides = array<i64:1,1,1,1>
    } : (tensor<1x4x4x1xi32>) -> (tensor<1x3x3x1xi32>, tensor<1x3x3x1x4xi1>)
    %1 = pphlo.maxpool_scatter %0#1, %arg1 {
      window_dimensions = array<i64:1, 2, 2, 1>,
      window_strides = array<i64:1,1,1,1>
    } : (tensor<1x3x3x1x4xi1>, tensor<1x3x3x1xi32>) -> tensor<1x4x4x1xi32>
    return %0#0, %0#1, %1: tensor<1x3x3x1xi32>, tensor<1x3x3x1x4xi1>, tensor<1x4x4x1xi32>
})",
        3);

  xt::xarray<int32_t> reduce_ret = {{8, 9, 9},  //
                                    {8, 9, 9},
                                    {6, 7, 7}};
  r.verifyOutput(reduce_ret.data(), 0);

  xt::xarray<uint8_t> mask = {{{0, 0, 0, 1}, {0, 0, 0, 1}, {0, 0, 1, 0}},  //
                              {{0, 1, 0, 0}, {0, 1, 0, 0}, {1, 0, 0, 0}},  //
                              {{0, 0, 0, 1}, {0, 1, 0, 0}, {0, 0, 0, 1}}};

  r.verifyOutput(mask.data(), 1);
}

TEST_P(ExecutorTest, IntNot) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  r.run(R"(
func.func @main() -> (tensor<i32>, tensor<i1>) {
    %0 = pphlo.constant dense<5> : tensor<i32>
    %1 = pphlo.not %0 : tensor<i32>
    %2 = pphlo.constant dense<0> : tensor<i1>
    %3 = pphlo.not %2 : tensor<i1>
    return %1, %3: tensor<i32>, tensor<i1>
})",
        2);

  {
    int32_t expected = ~5;
    r.verifyOutput(&expected, 0);
  }
  {
    bool expected = true;
    r.verifyOutput(&expected, 1);
  }
}

TEST_P(ExecutorTest, Case) {
  const auto *prog = R"(
 func.func @main(%arg0: tensor<i32>) -> (tensor<i32>,tensor<i32>) {
  %0:2 = "pphlo.case"(%arg0) ({
    %1 = pphlo.constant dense<1> : tensor<i32>
    %2 = pphlo.constant dense<11> : tensor<i32>
    pphlo.return %1, %2 : tensor<i32>, tensor<i32>
  }, {
    %1 = pphlo.constant dense<2> : tensor<i32>
    %2 = pphlo.constant dense<12> : tensor<i32>
    pphlo.return %1, %2 : tensor<i32>, tensor<i32>
  }, {
    %1 = pphlo.constant dense<3> : tensor<i32>
    %2 = pphlo.constant dense<13> : tensor<i32>
    pphlo.return %1, %2 : tensor<i32>, tensor<i32>
  }) : (tensor<i32>) -> (tensor<i32>, tensor<i32>)
  return %0#0, %0#1: tensor<i32>, tensor<i32>
})";

  {
    // case 0
    Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
             std::get<2>(GetParam()));

    r.addInput(static_cast<int32_t>(0));

    r.run(prog, 2);

    r.verifyScalarOutput(static_cast<int32_t>(1), 0);
    r.verifyScalarOutput(static_cast<int32_t>(11), 1);
  }

  {
    // case 1
    Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
             std::get<2>(GetParam()));

    r.addInput(static_cast<int32_t>(1));

    r.run(prog, 2);

    r.verifyScalarOutput(static_cast<int32_t>(2), 0);
    r.verifyScalarOutput(static_cast<int32_t>(12), 1);
  }

  {
    // case 2
    Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
             std::get<2>(GetParam()));

    r.addInput(static_cast<int32_t>(2));

    r.run(prog, 2);

    r.verifyScalarOutput(static_cast<int32_t>(3), 0);
    r.verifyScalarOutput(static_cast<int32_t>(13), 1);
  }
}

TEST_P(ExecutorTest, CasePrivate) {
  const auto *prog = R"(
 func.func @main(%arg0: tensor<!pphlo.secret<i32>>) -> (tensor<!pphlo.secret<i32>>, tensor<!pphlo.secret<i32>>) {
  %0:2 = "pphlo.case"(%arg0) ({
    %1 = pphlo.constant dense<1> : tensor<i32>
    %2 = pphlo.convert %1 : (tensor<i32>) -> tensor<!pphlo.secret<i32>>
    %3 = pphlo.constant dense<11> : tensor<i32>
    %4 = pphlo.convert %3 : (tensor<i32>) -> tensor<!pphlo.secret<i32>>
    pphlo.return %2, %4 : tensor<!pphlo.secret<i32>>, tensor<!pphlo.secret<i32>>
  }, {
    %1 = pphlo.constant dense<2> : tensor<i32>
    %2 = pphlo.convert %1 : (tensor<i32>) -> tensor<!pphlo.secret<i32>>
    %3 = pphlo.constant dense<12> : tensor<i32>
    %4 = pphlo.convert %3 : (tensor<i32>) -> tensor<!pphlo.secret<i32>>
    pphlo.return %2, %4 : tensor<!pphlo.secret<i32>>, tensor<!pphlo.secret<i32>>
  }, {
    %1 = pphlo.constant dense<3> : tensor<i32>
    %2 = pphlo.convert %1 : (tensor<i32>) -> tensor<!pphlo.secret<i32>>
    %3 = pphlo.constant dense<13> : tensor<i32>
    %4 = pphlo.convert %3 : (tensor<i32>) -> tensor<!pphlo.secret<i32>>
    pphlo.return %2, %4 : tensor<!pphlo.secret<i32>>, tensor<!pphlo.secret<i32>>
  }) : (tensor<!pphlo.secret<i32>>) -> (tensor<!pphlo.secret<i32>>, tensor<!pphlo.secret<i32>>)
  return %0#0, %0#1: tensor<!pphlo.secret<i32>>, tensor<!pphlo.secret<i32>>
})";

  {
    // case 0
    Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
             std::get<2>(GetParam()));

    r.addInput(static_cast<int32_t>(0), VIS_SECRET);

    r.run(prog, 2);

    r.verifyScalarOutput(static_cast<int32_t>(1), 0);
    r.verifyScalarOutput(static_cast<int32_t>(11), 1);
  }

  {
    // case 1
    Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
             std::get<2>(GetParam()));

    r.addInput(static_cast<int32_t>(1), VIS_SECRET);

    r.run(prog, 2);

    r.verifyScalarOutput(static_cast<int32_t>(2), 0);
    r.verifyScalarOutput(static_cast<int32_t>(12), 1);
  }

  {
    // case 2
    Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
             std::get<2>(GetParam()));

    r.addInput(static_cast<int32_t>(2), VIS_SECRET);

    r.run(prog, 2);

    r.verifyScalarOutput(static_cast<int32_t>(3), 0);
    r.verifyScalarOutput(static_cast<int32_t>(13), 1);
  }
}

TEST_P(ExecutorTest, MixedPayload) {
  xt::xarray<int32_t> op = {10, 9,  8,  7,  6,  5,  4,  3,  2,  1,
                            99, 97, 98, 96, 91, 11, 12, 13, 14, 15};
  xt::xarray<int32_t> expected_ret0 = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                                       11, 12, 13, 14, 15, 91, 96, 97, 98, 99};
  xt::xarray<int32_t> expected_ret1 = {9,  8,  7,  6,  5,  4,  3,  2,  1,  0,
                                       15, 16, 17, 18, 19, 14, 13, 11, 12, 10};

  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  r.addInput(op, VIS_SECRET);

  r.run(r.compileMHlo(
            R"(
func.func @main(%arg0: tensor<20xi32>) -> (tensor<20xi32>, tensor<20xi32>) {
    %0 = stablehlo.iota dim = 0: tensor<20xi32>
    %1:2 = "stablehlo.sort"(%arg0, %0) ({
    ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<i32>, %arg4: tensor<i32>):
      %2 = stablehlo.compare LT, %arg1, %arg2 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %2 : tensor<i1>
    }) {dimension = 0 : i64, is_stable = true} : (tensor<20xi32>, tensor<20xi32>) -> (tensor<20xi32>, tensor<20xi32>)
    return %1#0, %1#1: tensor<20xi32>, tensor<20xi32>
})",
            {VIS_SECRET}),
        2);
  r.verifyOutput(expected_ret0.data(), 0);
  r.verifyOutput(expected_ret1.data(), 1);
}

TEST_P(ExecutorTest, MixedPayloadDescending) {
  xt::xarray<int32_t> op = {10, 9,  8,  7,  6,  5,  4,  3,  2,  1,
                            99, 97, 98, 96, 91, 11, 12, 13, 14, 15};
  xt::xarray<int32_t> expected_ret0 = {99, 98, 97, 96, 91, 15, 14, 13, 12, 11,
                                       10, 9,  8,  7,  6,  5,  4,  3,  2,  1};
  xt::xarray<int32_t> expected_ret1 = {10, 12, 11, 13, 14, 19, 18, 17, 16, 15,
                                       0,  1,  2,  3,  4,  5,  6,  7,  8,  9};

  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  r.addInput(op, VIS_SECRET);

  r.run(r.compileMHlo(
            R"(
func.func @main(%arg0: tensor<20xi32>) -> (tensor<20xi32>, tensor<20xi32>) {
    %0 = stablehlo.iota dim = 0 : tensor<20xi32>
    %1:2 = "stablehlo.sort"(%arg0, %0) ({
    ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<i32>, %arg4: tensor<i32>):
      %2 = stablehlo.compare GT, %arg1, %arg2 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %2 : tensor<i1>
    }) {dimension = 0 : i64, is_stable = true} : (tensor<20xi32>, tensor<20xi32>) -> (tensor<20xi32>, tensor<20xi32>)
    return %1#0, %1#1: tensor<20xi32>, tensor<20xi32>
})",
            {VIS_SECRET}),
        2);
  r.verifyOutput(expected_ret0.data(), 0);
  r.verifyOutput(expected_ret1.data(), 1);
}

INSTANTIATE_TEST_SUITE_P(
    ExecutorTestInstances, ExecutorTest,
    testing::Combine(testing::Values(4, 3, 2),
                     testing::Values(FieldType::FM64, FieldType::FM128),
                     testing::Values(ProtocolKind::REF2K,
                                     ProtocolKind::SEMI2K)),
    [](const testing::TestParamInfo<ExecutorTest::ParamType> &p) {
      return fmt::format("{}x{}x{}", std::get<0>(p.param), std::get<1>(p.param),
                         std::get<2>(p.param));
    });

// NOTE(junfeng): ABY3 is 3pc only.
INSTANTIATE_TEST_SUITE_P(
    ExecutorTestABY3Instances, ExecutorTest,
    testing::Combine(testing::Values(3),
                     testing::Values(FieldType::FM64, FieldType::FM128),
                     testing::Values(ProtocolKind::ABY3)),
    [](const testing::TestParamInfo<ExecutorTest::ParamType> &p) {
      return fmt::format("{}x{}x{}", std::get<0>(p.param), std::get<1>(p.param),
                         std::get<2>(p.param));
    });

}  // namespace spu::device::pphlo::test
