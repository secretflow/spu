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

#include "libspu/device/pphlo/pphlo_executor.h"

#include <array>
#include <cstddef>
#include <exception>
#include <memory>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xtensor/xrandom.hpp"

#include "libspu/device/pphlo/pphlo_executor_test_runner.h"
#include "libspu/device/symbol_table.h"
#include "libspu/mpc/ref2k/ref2k.h"

namespace spu::device::pphlo::test {

class ExecutorTest : public ::testing::TestWithParam<
                         std::tuple<size_t, FieldType, ProtocolKind>> {};

TEST_P(ExecutorTest, Basic) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));
  r.addInput(1);
  r.addInput(2);

  r.run(R"(
func.func @main(%arg0: tensor<!pphlo.pub<i32>>, %arg1: tensor<!pphlo.pub<i32>>) -> (tensor<!pphlo.pub<i32>>) {
  %0 = "pphlo.add"(%arg0, %arg1) : (tensor<!pphlo.pub<i32>>, tensor<!pphlo.pub<i32>>) -> tensor<!pphlo.pub<i32>>
  return %0 : tensor<!pphlo.pub<i32>>
})");

  r.verifyScalarOutput(3);
}

TEST_P(ExecutorTest, BoolSplatConstant) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  r.getConfig().set_enable_type_checker(false);

  r.run(R"(
func.func @main() -> (tensor<!pphlo.pub<i32>>) {
  %0 = "pphlo.constant"() {value = dense<true> : tensor<i1>} : () ->tensor<!pphlo.pub<i1>>
  %1 = "pphlo.constant"() {value = dense<1> : tensor<i32>} : () ->tensor<!pphlo.pub<i32>>
  %2 = "pphlo.constant"() {value = dense<0> : tensor<i32>} : () ->tensor<!pphlo.pub<i32>>
  %3 = "pphlo.select"(%0, %1, %2) : (tensor<!pphlo.pub<i1>>, tensor<!pphlo.pub<i32>>, tensor<!pphlo.pub<i32>>) -> tensor<!pphlo.pub<i32>>
  return %3 : tensor<!pphlo.pub<i32>>
})");

  int32_t expected = 1;
  r.verifyOutput(&expected);
}

TEST_P(ExecutorTest, EmptyConstant) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  r.run(R"(
func.func @main() -> tensor<0x!pphlo.pub<f32>> {
  %0 = "pphlo.constant"() {value = dense<> : tensor<0xf32>} : () -> tensor<0x!pphlo.pub<f32>>
  return %0 : tensor<0x!pphlo.pub<f32>>
})");

  r.verifyOutput<float>(nullptr);
}

TEST_P(ExecutorTest, BoolConstant) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  r.getConfig().set_enable_type_checker(false);

  r.run(R"(
func.func @main() -> (tensor<2x!pphlo.pub<i32>>) {
  %0 = "pphlo.constant"() {value = dense<[true,false]> : tensor<2xi1>} : () ->tensor<2x!pphlo.pub<i1>>
  %1 = "pphlo.constant"() {value = dense<1> : tensor<2xi32>} : () ->tensor<2x!pphlo.pub<i32>>
  %2 = "pphlo.constant"() {value = dense<0> : tensor<2xi32>} : () ->tensor<2x!pphlo.pub<i32>>
  %3 = "pphlo.select"(%0, %1, %2) : (tensor<2x!pphlo.pub<i1>>, tensor<2x!pphlo.pub<i32>>, tensor<2x!pphlo.pub<i32>>) -> tensor<2x!pphlo.pub<i32>>
  return %3 : tensor<2x!pphlo.pub<i32>>
})");

  std::array<int32_t, 2> expected{1, 0};
  r.verifyOutput(expected.data());
}

TEST_P(ExecutorTest, InvalidIR) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  ASSERT_THROW(r.run(R"(
func.func @main() -> tensor<!pphlo.pub<i32>> {
  %2 = "pphlo.constant"() {value = dense<[0x41DA6E5887800000, 0x41C94E3940000000, 0x41C4BD2007000000, 0x41DC95133AC00000, 0x41D1650CEC000000, 0x41C9DF42E7800000, 0x41D46C43B6800000, 0x41C467EE0E800000, 0x41DC705F14400000]> : tensor<9xf64>} : () -> tensor<9x!pphlo.pub<f64>>
  %3 = "pphlo.floor"(%2) : (tensor<9x!pphlo.pub<f64>>) -> tensor<9x!pphlo.pub<f64>>
  %9 = "pphlo.concatenate"(%3) {dimension = 0 : i64} : (tensor<9x!pphlo.pub<f64>>) -> tensor<9x!pphlo.pub<f64>>
  %10 = "pphlo.broadcast"(%9) {broadcast_dimensions = dense<13> : tensor<1xi64>} : (tensor<9x!pphlo.pub<f64>>) -> tensor<9x!pphlo.pub<f64>>
  %51 = "pphlo.constant"() {value = dense<5> : tensor<i32>} : () -> tensor<!pphlo.pub<i32>>
  "pphlo.return"(%51) : (tensor<!pphlo.pub<i32>>) -> ()
})"),
               std::exception);
}

TEST_P(ExecutorTest, WithConst) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));
  r.addInput(xt::xarray<int>({{1, 1}, {1, 1}}));

  r.run(R"(
func.func @main(%arg0: tensor<2x2x!pphlo.pub<i32>>) -> (tensor<2x2x!pphlo.pub<i32>>) {
    %0 = "pphlo.constant"() {value = dense<[[1,2],[3,4]]> : tensor<2x2xi32>} : () -> tensor<2x2x!pphlo.pub<i32>>
    %1 = "pphlo.add"(%arg0, %0) : (tensor<2x2x!pphlo.pub<i32>>, tensor<2x2x!pphlo.pub<i32>>) -> tensor<2x2x!pphlo.pub<i32>>
    return %1 : tensor<2x2x!pphlo.pub<i32>>
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
func.func @main(%arg0: tensor<2x3x!pphlo.pub<i32>>, %arg1: tensor<2x3x!pphlo.pub<i32>>) -> (tensor<4x3x!pphlo.pub<i32>>) {
  %0 = "pphlo.concatenate"(%arg0, %arg1) {dimension = 0 : i64} : (tensor<2x3x!pphlo.pub<i32>>, tensor<2x3x!pphlo.pub<i32>>) -> tensor<4x3x!pphlo.pub<i32>>
  return %0 : tensor<4x3x!pphlo.pub<i32>>
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
func.func @main(%arg0: tensor<2x3x!pphlo.pub<i32>>, %arg1: tensor<2x3x!pphlo.pub<i32>>) -> (tensor<2x6x!pphlo.pub<i32>>) {
  %0 = "pphlo.concatenate"(%arg0, %arg1) {dimension = 1 : i64} : (tensor<2x3x!pphlo.pub<i32>>, tensor<2x3x!pphlo.pub<i32>>) -> tensor<2x6x!pphlo.pub<i32>>
  return %0 : tensor<2x6x!pphlo.pub<i32>>
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
func.func @main(%arg0: tensor<3x!pphlo.pub<i32>>) -> (tensor<9x!pphlo.pub<i32>>) {
  %0 = "pphlo.concatenate"(%arg0, %arg0, %arg0) {dimension = 0 : i64} : (tensor<3x!pphlo.pub<i32>>, tensor<3x!pphlo.pub<i32>>, tensor<3x!pphlo.pub<i32>>) -> tensor<9x!pphlo.pub<i32>>
  return %0 : tensor<9x!pphlo.pub<i32>>
})");

  std::array<int, 12> expect = {1, 2, 3, 1, 2, 3, 1, 2, 3};
  r.verifyOutput(expect.data());
}

TEST_P(ExecutorTest, Slice) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));
  r.addInput(xt::xarray<int>({{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}}));

  r.run(R"(
func.func @main(%arg0: tensor<4x3x!pphlo.pub<i32>>) -> (tensor<2x2x!pphlo.pub<i32>>) {
  %0 = "pphlo.slice"(%arg0) {limit_indices = dense<[4, 3]> : tensor<2xi64>, start_indices = dense<[2, 1]> : tensor<2xi64>, strides = dense<[1, 1]> : tensor<2xi64>} : (tensor<4x3x!pphlo.pub<i32>>) -> tensor<2x2x!pphlo.pub<i32>>
  return %0 : tensor<2x2x!pphlo.pub<i32>>
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
func.func @main(%arg0: tensor<4x6x!pphlo.pub<i32>>) -> (tensor<2x3x!pphlo.pub<i32>>) {
  %0 = "pphlo.slice"(%arg0) {limit_indices = dense<[4, 6]> : tensor<2xi64>, start_indices = dense<[0, 0]> : tensor<2xi64>, strides = dense<[2, 2]> : tensor<2xi64>} : (tensor<4x6x!pphlo.pub<i32>>) -> tensor<2x3x!pphlo.pub<i32>>
  return %0 : tensor<2x3x!pphlo.pub<i32>>
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
func.func @main(%arg0: tensor<4x3x!pphlo.pub<i32>>) -> (tensor<2x6x!pphlo.pub<i32>>) {
  %0 = "pphlo.reshape"(%arg0) : (tensor<4x3x!pphlo.pub<i32>>) -> tensor<2x6x!pphlo.pub<i32>>
  return %0 : tensor<2x6x!pphlo.pub<i32>>
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
func.func @main(%arg0: tensor<!pphlo.pub<i32>>, %arg1: tensor<!pphlo.pub<i32>>) -> tensor<!pphlo.pub<i32>> {
  %0, %1 = "pphlo.while"(%arg0, %arg1) ( {
  ^bb0(%arg2: tensor<!pphlo.pub<i32>>, %arg3: tensor<!pphlo.pub<i32>>):  // no predecessors
    %2 = "pphlo.less"(%arg2, %arg3) : (tensor<!pphlo.pub<i32>>, tensor<!pphlo.pub<i32>>) -> tensor<!pphlo.pub<i1>>
    "pphlo.return"(%2) : (tensor<!pphlo.pub<i1>>) -> ()
  },  {
  ^bb0(%arg2: tensor<!pphlo.pub<i32>>, %arg3: tensor<!pphlo.pub<i32>>):  // no predecessors
    %2 = "pphlo.constant"() {value = dense<1> : tensor<i32>} : () -> tensor<!pphlo.pub<i32>>
    %3 = "pphlo.add"(%arg2, %2) {name = "compare.0"} : (tensor<!pphlo.pub<i32>>, tensor<!pphlo.pub<i32>>) -> tensor<!pphlo.pub<i32>>
    "pphlo.return"(%3, %arg3) : (tensor<!pphlo.pub<i32>>, tensor<!pphlo.pub<i32>>) -> ()
  }) : (tensor<!pphlo.pub<i32>>, tensor<!pphlo.pub<i32>>) -> (tensor<!pphlo.pub<i32>>, tensor<!pphlo.pub<i32>>)
  return %0 : tensor<!pphlo.pub<i32>>
})");

  r.verifyScalarOutput(3);
}

TEST_P(ExecutorTest, Reduce1D) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  const xt::xarray<int> in1({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  r.addInput(in1);

  r.run(R"(
func.func @main(%arg0: tensor<10x!pphlo.pub<i32>>) -> (tensor<!pphlo.pub<i32>>) {
  %0 = "pphlo.constant"() {value = dense<0> : tensor<i32>} : () -> tensor<!pphlo.pub<i32>>
  %1 = "pphlo.reduce"(%arg0, %0) ( {
        ^bb0(%arg1: tensor<!pphlo.pub<i32>>, %arg2: tensor<!pphlo.pub<i32>>): // no predecessors
         %2 = "pphlo.add"(%arg1, %arg2) : (tensor<!pphlo.pub<i32>>, tensor<!pphlo.pub<i32>>) -> tensor<!pphlo.pub<i32>>
         "pphlo.return"(%2) : (tensor<!pphlo.pub<i32>>) -> ()
  }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<10x!pphlo.pub<i32>>, tensor<!pphlo.pub<i32>>) -> tensor<!pphlo.pub<i32>>
  return %1 :  tensor<!pphlo.pub<i32>>
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
func.func @main(%arg0: tensor<2x3x!pphlo.pub<i32>>) -> (tensor<2x!pphlo.pub<i32>>) {
  %0 = "pphlo.constant"() {value = dense<0> : tensor<i32>} : () -> tensor<!pphlo.pub<i32>>
  %1 = "pphlo.reduce"(%arg0, %0) ( {
        ^bb0(%arg1: tensor<!pphlo.pub<i32>>, %arg2: tensor<!pphlo.pub<i32>>): // no predecessors
         %2 = "pphlo.add"(%arg1, %arg2) : (tensor<!pphlo.pub<i32>>, tensor<!pphlo.pub<i32>>) -> tensor<!pphlo.pub<i32>>
         "pphlo.return"(%2) : (tensor<!pphlo.pub<i32>>) -> ()
  }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<2x3x!pphlo.pub<i32>>, tensor<!pphlo.pub<i32>>) -> tensor<2x!pphlo.pub<i32>>
  return %1 :  tensor<2x!pphlo.pub<i32>>
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
func.func @main(%arg0: tensor<2x3x!pphlo.pub<i32>>) -> (tensor<3x!pphlo.pub<i32>>) {
  %0 = "pphlo.constant"() {value = dense<0> : tensor<i32>} : () -> tensor<!pphlo.pub<i32>>
  %1 = "pphlo.reduce"(%arg0, %0) ( {
        ^bb0(%arg1: tensor<!pphlo.pub<i32>>, %arg2: tensor<!pphlo.pub<i32>>): // no predecessors
         %2 = "pphlo.add"(%arg1, %arg2) : (tensor<!pphlo.pub<i32>>, tensor<!pphlo.pub<i32>>) -> tensor<!pphlo.pub<i32>>
         "pphlo.return"(%2) : (tensor<!pphlo.pub<i32>>) -> ()
  }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<2x3x!pphlo.pub<i32>>, tensor<!pphlo.pub<i32>>) -> tensor<3x!pphlo.pub<i32>>
  return %1 :  tensor<3x!pphlo.pub<i32>>
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
func.func @main(%arg0: tensor<10x!pphlo.pub<i32>>) -> (tensor<!pphlo.pub<i32>>, tensor<!pphlo.pub<i32>>) {
  %0 = "pphlo.constant"() {value = dense<0> : tensor<i32>} : () -> tensor<!pphlo.pub<i32>>
  %1:2 = "pphlo.reduce"(%arg0, %arg0, %0, %0) ( {
        ^bb0(%arg1: tensor<!pphlo.pub<i32>>, %arg2: tensor<!pphlo.pub<i32>>, %arg3: tensor<!pphlo.pub<i32>>, %arg4: tensor<!pphlo.pub<i32>>): // no predecessors
         %2 = "pphlo.add"(%arg1, %arg3) : (tensor<!pphlo.pub<i32>>, tensor<!pphlo.pub<i32>>) -> tensor<!pphlo.pub<i32>>
         %3 = "pphlo.maximum"(%arg2, %arg4) : (tensor<!pphlo.pub<i32>>, tensor<!pphlo.pub<i32>>) -> tensor<!pphlo.pub<i32>>
         "pphlo.return"(%2, %3) : (tensor<!pphlo.pub<i32>>, tensor<!pphlo.pub<i32>>) -> ()
  }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<10x!pphlo.pub<i32>>, tensor<10x!pphlo.pub<i32>>, tensor<!pphlo.pub<i32>>, tensor<!pphlo.pub<i32>>) -> (tensor<!pphlo.pub<i32>>, tensor<!pphlo.pub<i32>>)
  return %1#0, %1#1 : tensor<!pphlo.pub<i32>>, tensor<!pphlo.pub<i32>>
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
func.func @main(%arg0: tensor<1x10x!pphlo.pub<f32>>) -> (tensor<1x!pphlo.pub<f32>>) {
  // Initial value is -inf
  %0 = "pphlo.constant"() {value = dense<0xFF800000> : tensor<f32>} : () -> tensor<!pphlo.pub<f32>>
  %1 = "pphlo.reduce"(%arg0, %0) ( {
  ^bb0(%arg1: tensor<!pphlo.pub<f32>>, %arg2: tensor<!pphlo.pub<f32>>):  // no predecessors
    %2 = "pphlo.maximum"(%arg1, %arg2) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    "pphlo.return"(%2) : (tensor<!pphlo.pub<f32>>) -> ()
  }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<1x10x!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<1x!pphlo.pub<f32>>
  return %1 :  tensor<1x!pphlo.pub<f32>>
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
func.func @main(%arg0: tensor<2x3x4x!pphlo.sec<f32>>, %arg1: tensor<2x3x4x!pphlo.sec<f32>>) -> (tensor<!pphlo.sec<i1>>) {
  %0 = "pphlo.constant"() {value = dense<true> : tensor<i1>} : () -> tensor<!pphlo.pub<i1>>
  %1 = "pphlo.equal"(%arg0, %arg1) : (tensor<2x3x4x!pphlo.sec<f32>>, tensor<2x3x4x!pphlo.sec<f32>>) -> tensor<2x3x4x!pphlo.sec<i1>>
  %2 = "pphlo.convert"(%0) : (tensor<!pphlo.pub<i1>>) -> tensor<!pphlo.sec<i1>>
  %3 = "pphlo.reduce"(%1, %2) ({
  ^bb0(%arg2: tensor<!pphlo.sec<i1>>, %arg3: tensor<!pphlo.sec<i1>>):
    %4 = "pphlo.and"(%arg2, %arg3) : (tensor<!pphlo.sec<i1>>, tensor<!pphlo.sec<i1>>) -> tensor<!pphlo.sec<i1>>
    "pphlo.return"(%4) : (tensor<!pphlo.sec<i1>>) -> ()
  }) {dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<2x3x4x!pphlo.sec<i1>>, tensor<!pphlo.sec<i1>>) -> tensor<!pphlo.sec<i1>>
  return %3 :  tensor<!pphlo.sec<i1>>
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
func.func @main(%arg0: tensor<4x6x!pphlo.pub<i32>>) -> (tensor<2x2x!pphlo.pub<i32>>) {
  %0 = "pphlo.constant"() {value = dense<0> : tensor<i32>} : () -> tensor<!pphlo.pub<i32>>
  %1 = "pphlo.reduce_window"(%arg0, %0) ( {
    ^bb0(%arg1: tensor<!pphlo.pub<i32>>, %arg2: tensor<!pphlo.pub<i32>>):  // no predecessors
      %2 = "pphlo.add"(%arg1, %arg2) : (tensor<!pphlo.pub<i32>>, tensor<!pphlo.pub<i32>>) -> tensor<!pphlo.pub<i32>>
      "pphlo.return"(%2) : (tensor<!pphlo.pub<i32>>) -> ()
    }) {base_dilations = dense<1> : tensor<2xi64>, padding = dense<0> : tensor<2x2xi64>, window_dilations = dense<1> : tensor<2xi64>, window_dimensions = dense<[2,3]> : tensor<2xi64>, window_strides = dense<[2,3]> : tensor<2xi64>} : (tensor<4x6x!pphlo.pub<i32>>, tensor<!pphlo.pub<i32>>) -> tensor<2x2x!pphlo.pub<i32>>

  return %1 :  tensor<2x2x!pphlo.pub<i32>>
})");

  xt::xarray<int> expect = {{-38, -23}, {25, -7}};
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
func.func @main(%arg0: tensor<4x6x!pphlo.pub<i32>>) -> (tensor<3x4x!pphlo.pub<i32>>) {
  %0 = "pphlo.constant"() {value = dense<0> : tensor<i32>} : () -> tensor<!pphlo.pub<i32>>
  %1 = "pphlo.reduce_window"(%arg0, %0) ( {
    ^bb0(%arg1: tensor<!pphlo.pub<i32>>, %arg2: tensor<!pphlo.pub<i32>>):  // no predecessors
      %2 = "pphlo.maximum"(%arg1, %arg2) : (tensor<!pphlo.pub<i32>>, tensor<!pphlo.pub<i32>>) -> tensor<!pphlo.pub<i32>>
      "pphlo.return"(%2) : (tensor<!pphlo.pub<i32>>) -> ()
    }) {base_dilations = dense<1> : tensor<2xi64>, padding = dense<0> : tensor<2x2xi64>, window_dilations = dense<1> : tensor<2xi64>, window_dimensions = dense<[2,3]> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<4x6x!pphlo.pub<i32>>, tensor<!pphlo.pub<i32>>) -> tensor<3x4x!pphlo.pub<i32>>

  return %1 :  tensor<3x4x!pphlo.pub<i32>>
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
func.func @main(%arg0: tensor<4x4x!pphlo.pub<i32>>) -> (tensor<2x2x!pphlo.pub<i32>>) {
  %0 = "pphlo.constant"() {value = dense<0> : tensor<i32>} : () -> tensor<!pphlo.pub<i32>>
  %1 = "pphlo.reduce_window"(%arg0, %0) ( {
    ^bb0(%arg1: tensor<!pphlo.pub<i32>>, %arg2: tensor<!pphlo.pub<i32>>):  // no predecessors
      %2 = "pphlo.maximum"(%arg1, %arg2) : (tensor<!pphlo.pub<i32>>, tensor<!pphlo.pub<i32>>) -> tensor<!pphlo.pub<i32>>
      "pphlo.return"(%2) : (tensor<!pphlo.pub<i32>>) -> ()
    }) {base_dilations = dense<1> : tensor<2xi64>, padding = dense<0> : tensor<2x2xi64>, window_dilations = dense<2> : tensor<2xi64>, window_dimensions = dense<2> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<4x4x!pphlo.pub<i32>>, tensor<!pphlo.pub<i32>>) -> tensor<2x2x!pphlo.pub<i32>>

  return %1 :  tensor<2x2x!pphlo.pub<i32>>
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
func.func @main(%arg0: tensor<4x4x!pphlo.pub<i32>>) -> (tensor<1x1x!pphlo.pub<i32>>) {
  %0 = "pphlo.constant"() {value = dense<0> : tensor<i32>} : () -> tensor<!pphlo.pub<i32>>
  %1 = "pphlo.reduce_window"(%arg0, %0) ( {
    ^bb0(%arg1: tensor<!pphlo.pub<i32>>, %arg2: tensor<!pphlo.pub<i32>>):  // no predecessors
      %2 = "pphlo.maximum"(%arg1, %arg2) : (tensor<!pphlo.pub<i32>>, tensor<!pphlo.pub<i32>>) -> tensor<!pphlo.pub<i32>>
      "pphlo.return"(%2) : (tensor<!pphlo.pub<i32>>) -> ()
    }) {base_dilations = dense<1> : tensor<2xi64>, padding = dense<0> : tensor<2x2xi64>, window_dilations = dense<2> : tensor<2xi64>, window_dimensions = dense<2> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : (tensor<4x4x!pphlo.pub<i32>>, tensor<!pphlo.pub<i32>>) -> tensor<1x1x!pphlo.pub<i32>>

  return %1 :  tensor<1x1x!pphlo.pub<i32>>
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

  r.run(R"(
func.func @main(%arg0: tensor<4x4x!pphlo.pub<i32>>) -> (tensor<6x6x!pphlo.pub<i32>>) {
  %0 = "pphlo.constant"() {value = dense<0> : tensor<i32>} : () -> tensor<!pphlo.pub<i32>>
  %1 = "pphlo.reduce_window"(%arg0, %0) ( {
    ^bb0(%arg1: tensor<!pphlo.pub<i32>>, %arg2: tensor<!pphlo.pub<i32>>):  // no predecessors
      %2 = "pphlo.maximum"(%arg1, %arg2) : (tensor<!pphlo.pub<i32>>, tensor<!pphlo.pub<i32>>) -> tensor<!pphlo.pub<i32>>
      "pphlo.return"(%2) : (tensor<!pphlo.pub<i32>>) -> ()
    }) {base_dilations = dense<2> : tensor<2xi64>, padding = dense<0> : tensor<2x2xi64>, window_dilations = dense<1> : tensor<2xi64>, window_dimensions = dense<2> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<4x4x!pphlo.pub<i32>>, tensor<!pphlo.pub<i32>>) -> tensor<6x6x!pphlo.pub<i32>>

  return %1 :  tensor<6x6x!pphlo.pub<i32>>
})");

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

  r.run(R"(
func.func @main(%arg0: tensor<4x4x!pphlo.pub<i32>>) -> (tensor<3x3x!pphlo.pub<i32>>) {
  %0 = "pphlo.constant"() {value = dense<0> : tensor<i32>} : () -> tensor<!pphlo.pub<i32>>
  %1 = "pphlo.reduce_window"(%arg0, %0) ( {
    ^bb0(%arg1: tensor<!pphlo.pub<i32>>, %arg2: tensor<!pphlo.pub<i32>>):  // no predecessors
      %2 = "pphlo.maximum"(%arg1, %arg2) : (tensor<!pphlo.pub<i32>>, tensor<!pphlo.pub<i32>>) -> tensor<!pphlo.pub<i32>>
      "pphlo.return"(%2) : (tensor<!pphlo.pub<i32>>) -> ()
    }) {base_dilations = dense<2> : tensor<2xi64>, padding = dense<0> : tensor<2x2xi64>, window_dilations = dense<1> : tensor<2xi64>, window_dimensions = dense<2> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : (tensor<4x4x!pphlo.pub<i32>>, tensor<!pphlo.pub<i32>>) -> tensor<3x3x!pphlo.pub<i32>>

  return %1 :  tensor<3x3x!pphlo.pub<i32>>
})");

  xt::xarray<int> expect = {{0, 1, 2}, {4, 5, 6}, {8, 9, 10}};
  r.verifyOutput(expect.data());
}

TEST_P(ExecutorTest, ReduceWindowMaxIotaStrideBothDilation) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  const xt::xarray<int> in1 = {
      {0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}, {12, 13, 14, 15}};
  r.addInput(in1);

  r.run(R"(
func.func @main(%arg0: tensor<4x4x!pphlo.pub<i32>>) -> (tensor<3x3x!pphlo.pub<i32>>) {
  %0 = "pphlo.constant"() {value = dense<0> : tensor<i32>} : () -> tensor<!pphlo.pub<i32>>
  %1 = "pphlo.reduce_window"(%arg0, %0) ( {
    ^bb0(%arg1: tensor<!pphlo.pub<i32>>, %arg2: tensor<!pphlo.pub<i32>>):  // no predecessors
      %2 = "pphlo.maximum"(%arg1, %arg2) : (tensor<!pphlo.pub<i32>>, tensor<!pphlo.pub<i32>>) -> tensor<!pphlo.pub<i32>>
      "pphlo.return"(%2) : (tensor<!pphlo.pub<i32>>) -> ()
    }) {base_dilations = dense<2> : tensor<2xi64>, padding = dense<0> : tensor<2x2xi64>, window_dilations = dense<2> : tensor<2xi64>, window_dimensions = dense<2> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : (tensor<4x4x!pphlo.pub<i32>>, tensor<!pphlo.pub<i32>>) -> tensor<3x3x!pphlo.pub<i32>>

  return %1 :  tensor<3x3x!pphlo.pub<i32>>
})");

  xt::xarray<int> expect = {{5, 6, 7}, {9, 10, 11}, {13, 14, 15}};
  r.verifyOutput(expect.data());
}

TEST_P(ExecutorTest, ReduceWindowMaxIotaPaddingStrideBaseDilation) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  const xt::xarray<int> in1 = {
      {0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}, {12, 13, 14, 15}};
  r.addInput(in1);

  r.run(R"(
func.func @main(%arg0: tensor<4x4x!pphlo.pub<i32>>) -> (tensor<3x3x!pphlo.pub<i32>>) {
  %0 = "pphlo.constant"() {value = dense<0> : tensor<i32>} : () -> tensor<!pphlo.pub<i32>>
  %1 = "pphlo.reduce_window"(%arg0, %0) ( {
    ^bb0(%arg1: tensor<!pphlo.pub<i32>>, %arg2: tensor<!pphlo.pub<i32>>):  // no predecessors
      %2 = "pphlo.maximum"(%arg1, %arg2) : (tensor<!pphlo.pub<i32>>, tensor<!pphlo.pub<i32>>) -> tensor<!pphlo.pub<i32>>
      "pphlo.return"(%2) : (tensor<!pphlo.pub<i32>>) -> ()
    }) {base_dilations = dense<2> : tensor<2xi64>, padding = dense<1> : tensor<2x2xi64>, window_dilations = dense<1> : tensor<2xi64>, window_dimensions = dense<3> : tensor<2xi64>, window_strides = dense<3> : tensor<2xi64>} : (tensor<4x4x!pphlo.pub<i32>>, tensor<!pphlo.pub<i32>>) -> tensor<3x3x!pphlo.pub<i32>>

  return %1 :  tensor<3x3x!pphlo.pub<i32>>
})");

  xt::xarray<int> expect = {{0, 2, 3}, {8, 10, 11}, {12, 14, 15}};
  r.verifyOutput(expect.data());
}

TEST_P(ExecutorTest, If) {
  const auto *prog = R"(
 func.func @main(%arg0: tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>> {
  %0 = "pphlo.constant"() {value = dense<1.000000e+01> : tensor<f32>} : () -> tensor<!pphlo.pub<f32>>
  %1 = "pphlo.less"(%arg0, %0) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<i1>>
  %2 = "pphlo.if"(%1) ( {
    %3 = "pphlo.multiply"(%arg0, %arg0) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    "pphlo.return"(%3) : (tensor<!pphlo.pub<f32>>) -> ()
  },  {
    %3 = "pphlo.add"(%arg0, %arg0) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    "pphlo.return"(%3) : (tensor<!pphlo.pub<f32>>) -> ()
  }) : (tensor<!pphlo.pub<i1>>) -> tensor<!pphlo.pub<f32>>
  return %2 : tensor<!pphlo.pub<f32>>
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
func.func @main(%arg0: tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.sec<f32>> {
  %0 = "pphlo.constant"() {value = dense<1.000000e+01> : tensor<f32>} : () -> tensor<!pphlo.pub<f32>>
  %1 = "pphlo.convert"(%arg0) : (tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.sec<f32>>
  %2 = "pphlo.less"(%1, %0) : (tensor<!pphlo.sec<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.sec<i1>>
  %3 = "pphlo.if"(%2) ( {
    %4 = "pphlo.multiply"(%arg0, %arg0) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    "pphlo.return"(%4) : (tensor<!pphlo.pub<f32>>) -> ()
  },  {
    %4 = "pphlo.add"(%arg0, %arg0) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    "pphlo.return"(%4) : (tensor<!pphlo.pub<f32>>) -> ()
  }) : (tensor<!pphlo.sec<i1>>) -> tensor<!pphlo.sec<f32>>
  return %3 : tensor<!pphlo.sec<f32>>
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
func.func @main() -> (tensor<4x!pphlo.pub<i32>>) {
    %0 = "pphlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<4x!pphlo.pub<i32>>
    return %0 : tensor<4x!pphlo.pub<i32>>
})");

  std::array<int, 4> expect = {0, 1, 2, 3};
  r.verifyOutput(expect.data());
}

TEST_P(ExecutorTest, Iota2D) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  r.run(R"(
func.func @main() -> (tensor<4x2x!pphlo.pub<i32>>) {
    %0 = "pphlo.iota"() {iota_dimension = 1 : i64} : () -> tensor<4x2x!pphlo.pub<i32>>
    return %0 : tensor<4x2x!pphlo.pub<i32>>
})");

  std::array<int, 8> expect = {0, 1, 0, 1, 0, 1, 0, 1};
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
func.func @main(%arg0: tensor<!pphlo.pub<f32>>) -> (tensor<!pphlo.pub<i32>>) {
    %0 = "pphlo.bitcast_convert"(%arg0) {elsize = 32 : i64} : (tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<i32>>
    return %0 : tensor<!pphlo.pub<i32>>
})");

  r.verifyOutput(reinterpret_cast<int32_t *>(&in));
}

void testGatherImpl(size_t world_size, FieldType field, ProtocolKind protocol,
                    const xt::xarray<int> &operand,
                    const xt::xarray<int> &indices,
                    const xt::xarray<int> &expected, const std::string &mhlo) {
  // Public index
  {
    Runner r(world_size, field, protocol);

    r.addInput(operand);
    // Start indices
    r.addInput(indices);

    auto compiled = r.compileMHlo(mhlo, {VIS_PUBLIC, VIS_PUBLIC});

    EXPECT_THAT(compiled, testing::HasSubstr("pphlo.gather"));

    r.run(compiled);

    r.verifyOutput(expected.data());
  }

  // Secret index
  {
    Runner r(world_size, field, protocol);

    r.addInput(operand);
    // Start indices
    r.addInput(indices, VIS_SECRET);

    auto compiled = r.compileMHlo(mhlo, {VIS_PUBLIC, VIS_SECRET});

    EXPECT_THAT(compiled, testing::Not(testing::HasSubstr("pphlo.gather")));

    r.run(compiled);

    r.verifyOutput(expected.data());
  }
}

TEST_P(ExecutorTest, Gather1) {
  std::string mhlo = R"(
func.func @main(%arg0: tensor<3x3xi32>, %arg1: tensor<2xi32>) -> (tensor<2x3xi32>) {
    %0 = "mhlo.gather"(%arg0, %arg1) {dimension_numbers = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 3]> : tensor<2xi64>} : (tensor<3x3xi32>, tensor<2xi32>) -> tensor<2x3xi32>
    return %0 : tensor<2x3xi32>
})";

  auto operand = xt::xarray<int>{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  auto indices = xt::xarray<int>{0, 2};
  xt::xarray<int> expected = {{1, 2, 3}, {7, 8, 9}};

  testGatherImpl(std::get<0>(GetParam()), std::get<1>(GetParam()),
                 std::get<2>(GetParam()), operand, indices, expected, mhlo);
}

TEST_P(ExecutorTest, Gather2) {
  std::string mhlo = R"(
func.func @main(%arg0: tensor<3x3xi32>, %arg1: tensor<2xi32>) -> (tensor<3x2xi32>) {
    %0 = "mhlo.gather"(%arg0, %arg1) {dimension_numbers = #mhlo.gather<offset_dims = [0], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[3,1]> : tensor<2xi64>} : (tensor<3x3xi32>, tensor<2xi32>) -> tensor<3x2xi32>
    return %0 : tensor<3x2xi32>
})";

  auto operand = xt::xarray<int>{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  auto indices = xt::xarray<int>{0, 2};
  xt::xarray<int> expected = {{1, 3}, {4, 6}, {7, 9}};

  testGatherImpl(std::get<0>(GetParam()), std::get<1>(GetParam()),
                 std::get<2>(GetParam()), operand, indices, expected, mhlo);
}

TEST_P(ExecutorTest, GatherBatch) {
  std::string mhlo = R"(
func.func @main(%arg0: tensor<3x3xi32>, %arg1: tensor<2x2xi32>) -> (tensor<2x3x2xi32>) {
    %0 = "mhlo.gather"(%arg0, %arg1) {dimension_numbers = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = dense<[3,1]> : tensor<2xi64>} : (tensor<3x3xi32>, tensor<2x2xi32>) -> tensor<2x3x2xi32>
    return %0 : tensor<2x3x2xi32>
})";

  auto operand = xt::xarray<int>{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  auto indices = xt::xarray<int>{{0, 2}, {2, 1}};

  xt::xarray<int> expected = {{{1, 3}, {4, 6}, {7, 9}},
                              {{3, 2}, {6, 5}, {9, 8}}};

  testGatherImpl(std::get<0>(GetParam()), std::get<1>(GetParam()),
                 std::get<2>(GetParam()), operand, indices, expected, mhlo);
}

TEST_P(ExecutorTest, GatherNd) {
  std::string mhlo = R"(
func.func @main(%arg0: tensor<3x3x2xi32>, %arg1: tensor<2x2xi32>) -> (tensor<2x2xi32>) {
    %0 = "mhlo.gather"(%arg0, %arg1) {dimension_numbers = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [0,1], start_index_map = [0,1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1,1,2]> : tensor<3xi64>} : (tensor<3x3x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
    return %0 : tensor<2x2xi32>
})";
  const xt::xarray<int> operand = {{{-1, 1}, {-2, 2}, {-3, 3}},
                                   {{-4, 4}, {-5, 5}, {-6, 6}},
                                   {{-7, 7}, {-8, 8}, {-9, 9}}};
  auto indices = xt::xarray<int>{{0, 0}, {1, 0}};
  xt::xarray<int> expected = {{-1, 1}, {-4, 4}};

  testGatherImpl(std::get<0>(GetParam()), std::get<1>(GetParam()),
                 std::get<2>(GetParam()), operand, indices, expected, mhlo);
}

TEST_P(ExecutorTest, GatherNdNonDefaultIndexVectorDim) {
  std::string mhlo = R"(
func.func @main(%arg0: tensor<3x3x2xi32>, %arg1: tensor<2x2xi32>) -> (tensor<2x2xi32>) {
    %0 = "mhlo.gather"(%arg0, %arg1) {dimension_numbers = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [0,1], start_index_map = [0,1], index_vector_dim = 0>, indices_are_sorted = false, slice_sizes = dense<[1,1,2]> : tensor<3xi64>} : (tensor<3x3x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
    return %0 : tensor<2x2xi32>
})";
  xt::xarray<int> operand = {{{-1, 1}, {-2, 2}, {-3, 3}},
                             {{-4, 4}, {-5, 5}, {-6, 6}},
                             {{-7, 7}, {-8, 8}, {-9, 9}}};
  auto indices = xt::xarray<int>{{0, 0}, {1, 0}};
  xt::xarray<int> expected = {{-2, 2}, {-1, 1}};

  testGatherImpl(std::get<0>(GetParam()), std::get<1>(GetParam()),
                 std::get<2>(GetParam()), operand, indices, expected, mhlo);
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
    %0 = mhlo.convolution(%arg0, %arg1)
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
    %0 = mhlo.convolution(%arg0, %arg1)
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
    %0 = mhlo.convolution(%arg0, %arg1)
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
    %0 = mhlo.convolution(%arg0, %arg1)
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
    %0 = mhlo.convolution(%arg0, %arg1)
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
func.func @main(%arg0: tensor<2x!pphlo.pub<i32>>, %arg1: tensor<2x!pphlo.pub<i32>>) -> (tensor<2x!pphlo.pub<i32>>) {
    %0 = "pphlo.shift_left"(%arg0, %arg1) : (tensor<2x!pphlo.pub<i32>>, tensor<2x!pphlo.pub<i32>>) -> tensor<2x!pphlo.pub<i32>>
    return %0 : tensor<2x!pphlo.pub<i32>>
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
func.func @main(%arg0: tensor<2x!pphlo.pub<i32>>, %arg1: tensor<2x!pphlo.pub<i32>>) -> (tensor<2x!pphlo.pub<i32>>) {
    %0 = "pphlo.shift_right_logical"(%arg0, %arg1) : (tensor<2x!pphlo.pub<i32>>, tensor<2x!pphlo.pub<i32>>) -> tensor<2x!pphlo.pub<i32>>
    return %0 : tensor<2x!pphlo.pub<i32>>
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
func.func @main(%arg0: tensor<!pphlo.pub<i32>>) -> (tensor<!pphlo.pub<i32>>) {
  %0 = "pphlo.constant"() {value = dense<-2147483648> : tensor<i32>} : () -> tensor<!pphlo.pub<i32>>
  %1 = "pphlo.maximum"(%0, %arg0) : (tensor<!pphlo.pub<i32>>, tensor<!pphlo.pub<i32>>) -> tensor<!pphlo.pub<i32>>
  return %1 :  tensor<!pphlo.pub<i32>>
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
func.func @main(%arg0: tensor<!pphlo.pub<i32>>) -> (tensor<!pphlo.pub<i32>>) {
  %0 = "pphlo.constant"() {value = dense<2147483647> : tensor<i32>} : () -> tensor<!pphlo.pub<i32>>
  %1 = "pphlo.minimum"(%0, %arg0) : (tensor<!pphlo.pub<i32>>, tensor<!pphlo.pub<i32>>) -> tensor<!pphlo.pub<i32>>
  return %1 :  tensor<!pphlo.pub<i32>>
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
func.func @main(%arg0: tensor<5x!pphlo.pub<i32>>, %arg1: tensor<!pphlo.pub<i32>>) -> tensor<2x!pphlo.pub<i32>> {
  %0 = "pphlo.dynamic-slice"(%arg0, %arg1) {slice_sizes = dense<2> : tensor<1xi64>} : (tensor<5x!pphlo.pub<i32>>, tensor<!pphlo.pub<i32>>) -> tensor<2x!pphlo.pub<i32>>
  return %0 : tensor<2x!pphlo.pub<i32>>
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
func.func @main(%arg0: tensor<4x3x!pphlo.pub<f32>>, %arg1: tensor<!pphlo.pub<i32>>, %arg2: tensor<!pphlo.pub<i32>>) -> tensor<2x2x!pphlo.pub<f32>> {
  %0 = "pphlo.dynamic-slice"(%arg0, %arg1, %arg2) {slice_sizes = dense<[2, 2]> : tensor<2xi64>} : (tensor<4x3x!pphlo.pub<f32>>, tensor<!pphlo.pub<i32>>, tensor<!pphlo.pub<i32>>) -> tensor<2x2x!pphlo.pub<f32>>
  return %0 : tensor<2x2x!pphlo.pub<f32>>
})");

  xt::xarray<float> expected = {{7.0, 8.0}, {10.0, 11.0}};
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
func.func @main(%arg0: tensor<5x!pphlo.pub<i32>>, %arg1: tensor<2x!pphlo.pub<i32>>, %arg2: tensor<!pphlo.pub<i32>>) -> tensor<5x!pphlo.pub<i32>> {
  %0 = "pphlo.dynamic-update-slice"(%arg0, %arg1, %arg2) : (tensor<5x!pphlo.pub<i32>>, tensor<2x!pphlo.pub<i32>>, tensor<!pphlo.pub<i32>>) -> tensor<5x!pphlo.pub<i32>>
  return %0 : tensor<5x!pphlo.pub<i32>>
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
func.func @main(%arg0: tensor<4x3x!pphlo.pub<f32>>, %arg1: tensor<3x2x!pphlo.pub<f32>>, %arg2: tensor<!pphlo.pub<i32>>, %arg3: tensor<!pphlo.pub<i32>>) -> tensor<4x3x!pphlo.pub<f32>> {
  %0 = "pphlo.dynamic-update-slice"(%arg0, %arg1, %arg2, %arg3) : (tensor<4x3x!pphlo.pub<f32>>, tensor<3x2x!pphlo.pub<f32>>, tensor<!pphlo.pub<i32>>, tensor<!pphlo.pub<i32>>) -> tensor<4x3x!pphlo.pub<f32>>
  return %0 : tensor<4x3x!pphlo.pub<f32>>
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
func.func @main(%arg0: tensor<4x!pphlo.pub<f32>>) -> tensor<4x!pphlo.pub<f32>> {
    %0 = "pphlo.sort"(%arg0) ( {
    ^bb0(%arg1: tensor<!pphlo.pub<f32>>, %arg2: tensor<!pphlo.pub<f32>>):  // no predecessors
      %1 = "pphlo.less"(%arg1, %arg2) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<i1>>
      "pphlo.return"(%1) : (tensor<!pphlo.pub<i1>>) -> ()
    }) {dimension = 0 : i64, is_stable = true} : (tensor<4x!pphlo.pub<f32>>) -> (tensor<4x!pphlo.pub<f32>>)
    return %0 : tensor<4x!pphlo.pub<f32>>
})");
    r.verifyOutput(expected.data());
  }

  {
    Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
             std::get<2>(GetParam()));

    r.addInput(op, VIS_SECRET);

    r.run(R"(
func.func @main(%arg0: tensor<4x!pphlo.sec<f32>>) -> tensor<4x!pphlo.sec<f32>> {
    %0 = "pphlo.sort"(%arg0) ( {
    ^bb0(%arg1: tensor<!pphlo.sec<f32>>, %arg2: tensor<!pphlo.sec<f32>>):  // no predecessors
      %1 = "pphlo.less"(%arg1, %arg2) : (tensor<!pphlo.sec<f32>>, tensor<!pphlo.sec<f32>>) -> tensor<!pphlo.sec<i1>>
      "pphlo.return"(%1) : (tensor<!pphlo.sec<i1>>) -> ()
    }) {dimension = 0 : i64, is_stable = true} : (tensor<4x!pphlo.sec<f32>>) -> (tensor<4x!pphlo.sec<f32>>)
    return %0 : tensor<4x!pphlo.sec<f32>>
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
func.func @main(%arg0: tensor<2x5x!pphlo.pub<f32>>) -> tensor<2x5x!pphlo.pub<f32>> {
    %0 = "pphlo.sort"(%arg0) ( {
    ^bb0(%arg1: tensor<!pphlo.pub<f32>>, %arg2: tensor<!pphlo.pub<f32>>):  // no predecessors
      %1 = "pphlo.less"(%arg1, %arg2) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<i1>>
      "pphlo.return"(%1) : (tensor<!pphlo.pub<i1>>) -> ()
    }) {dimension = 1 : i64, is_stable = true} : (tensor<2x5x!pphlo.pub<f32>>) -> (tensor<2x5x!pphlo.pub<f32>>)
    return %0 : tensor<2x5x!pphlo.pub<f32>>
})");
    r.verifyOutput(expected.data());
  }

  {
    Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
             std::get<2>(GetParam()));
    // Row sort
    r.addInput(op, VIS_SECRET);
    r.run(R"(
func.func @main(%arg0: tensor<2x5x!pphlo.sec<f32>>) -> tensor<2x5x!pphlo.sec<f32>> {
    %0 = "pphlo.sort"(%arg0) ( {
    ^bb0(%arg1: tensor<!pphlo.sec<f32>>, %arg2: tensor<!pphlo.sec<f32>>):  // no predecessors
      %1 = "pphlo.less"(%arg1, %arg2) : (tensor<!pphlo.sec<f32>>, tensor<!pphlo.sec<f32>>) -> tensor<!pphlo.sec<i1>>
      "pphlo.return"(%1) : (tensor<!pphlo.sec<i1>>) -> ()
    }) {dimension = 1 : i64, is_stable = true} : (tensor<2x5x!pphlo.sec<f32>>) -> (tensor<2x5x!pphlo.sec<f32>>)
    return %0 : tensor<2x5x!pphlo.sec<f32>>
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
func.func @main(%arg0: tensor<2x4x!pphlo.pub<f32>>) -> tensor<2x4x!pphlo.pub<f32>> {
    %0 = "pphlo.sort"(%arg0) ( {
    ^bb0(%arg1: tensor<!pphlo.pub<f32>>, %arg2: tensor<!pphlo.pub<f32>>):  // no predecessors
      %1 = "pphlo.less"(%arg1, %arg2) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<i1>>
      "pphlo.return"(%1) : (tensor<!pphlo.pub<i1>>) -> ()
    }) {dimension = 0 : i64, is_stable = true} : (tensor<2x4x!pphlo.pub<f32>>) -> (tensor<2x4x!pphlo.pub<f32>>)
    return %0 : tensor<2x4x!pphlo.pub<f32>>
})");

    r.verifyOutput(expected.data());
  }

  {
    Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
             std::get<2>(GetParam()));
    r.addInput(op, VIS_SECRET);

    // Column sort
    r.run(R"(
func.func @main(%arg0: tensor<2x4x!pphlo.sec<f32>>) -> tensor<2x4x!pphlo.sec<f32>> {
    %0 = "pphlo.sort"(%arg0) ( {
    ^bb0(%arg1: tensor<!pphlo.sec<f32>>, %arg2: tensor<!pphlo.sec<f32>>):  // no predecessors
      %1 = "pphlo.less"(%arg1, %arg2) : (tensor<!pphlo.sec<f32>>, tensor<!pphlo.sec<f32>>) -> tensor<!pphlo.sec<i1>>
      "pphlo.return"(%1) : (tensor<!pphlo.sec<i1>>) -> ()
    }) {dimension = 0 : i64, is_stable = true} : (tensor<2x4x!pphlo.sec<f32>>) -> (tensor<2x4x!pphlo.sec<f32>>)
    return %0 : tensor<2x4x!pphlo.sec<f32>>
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
func.func @main(%arg0: tensor<2x!pphlo.pub<i32>>, %arg1: tensor<2x!pphlo.pub<i32>>, %arg2: tensor<2x!pphlo.pub<f32>>) -> (tensor<2x!pphlo.pub<i32>>, tensor<2x!pphlo.pub<i32>>, tensor<2x!pphlo.pub<f32>>) {
    %0:3 = "pphlo.sort"(%arg0, %arg1, %arg2) ( {
    ^bb0(%arg3: tensor<!pphlo.pub<i32>>, %arg4: tensor<!pphlo.pub<i32>>, %arg5: tensor<!pphlo.pub<i32>>, %arg6: tensor<!pphlo.pub<i32>>, %arg7: tensor<!pphlo.pub<f32>>, %arg8: tensor<!pphlo.pub<f32>>):  // no predecessors
      %1 = "pphlo.less"(%arg3, %arg4) : (tensor<!pphlo.pub<i32>>, tensor<!pphlo.pub<i32>>) -> tensor<!pphlo.pub<i1>>
      "pphlo.return"(%1) : (tensor<!pphlo.pub<i1>>) -> ()
    }) {dimension = 0 : i64, is_stable = true} : (tensor<2x!pphlo.pub<i32>>, tensor<2x!pphlo.pub<i32>>, tensor<2x!pphlo.pub<f32>>) -> (tensor<2x!pphlo.pub<i32>>, tensor<2x!pphlo.pub<i32>>, tensor<2x!pphlo.pub<f32>>)
    return %0#0, %0#1, %0#2 : tensor<2x!pphlo.pub<i32>>, tensor<2x!pphlo.pub<i32>>, tensor<2x!pphlo.pub<f32>>
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
func.func @main(%arg0: tensor<2x!pphlo.sec<i32>>, %arg1: tensor<2x!pphlo.sec<i32>>, %arg2: tensor<2x!pphlo.sec<f32>>) -> (tensor<2x!pphlo.sec<i32>>, tensor<2x!pphlo.sec<i32>>, tensor<2x!pphlo.sec<f32>>) {
    %0:3 = "pphlo.sort"(%arg0, %arg1, %arg2) ( {
    ^bb0(%arg3: tensor<!pphlo.sec<i32>>, %arg4: tensor<!pphlo.sec<i32>>, %arg5: tensor<!pphlo.sec<i32>>, %arg6: tensor<!pphlo.sec<i32>>, %arg7: tensor<!pphlo.sec<f32>>, %arg8: tensor<!pphlo.sec<f32>>):  // no predecessors
      %1 = "pphlo.less"(%arg3, %arg4) : (tensor<!pphlo.sec<i32>>, tensor<!pphlo.sec<i32>>) -> tensor<!pphlo.sec<i1>>
      "pphlo.return"(%1) : (tensor<!pphlo.sec<i1>>) -> ()
    }) {dimension = 0 : i64, is_stable = true} : (tensor<2x!pphlo.sec<i32>>, tensor<2x!pphlo.sec<i32>>, tensor<2x!pphlo.sec<f32>>) -> (tensor<2x!pphlo.sec<i32>>, tensor<2x!pphlo.sec<i32>>, tensor<2x!pphlo.sec<f32>>)
    return %0#0, %0#1, %0#2 : tensor<2x!pphlo.sec<i32>>, tensor<2x!pphlo.sec<i32>>, tensor<2x!pphlo.sec<f32>>
})",
          3);

    r.verifyOutput(expected_x.data(), 0);
    r.verifyOutput(expected_y.data(), 1);
    r.verifyOutput(expected_z.data(), 2);
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
func.func @main(%arg0: tensor<4x!pphlo.pub<i32>>, %arg1: tensor<4x!pphlo.pub<i32>>) -> (tensor<4x!pphlo.pub<i32>>, tensor<4x!pphlo.pub<i32>>) {
    %0:2 = "pphlo.sort"(%arg0, %arg1) ( {
    ^bb0(%arg2: tensor<!pphlo.pub<i32>>, %arg3: tensor<!pphlo.pub<i32>>, %arg4: tensor<!pphlo.pub<i32>>, %arg5: tensor<!pphlo.pub<i32>>):  // no predecessors
      %1 = "pphlo.add"(%arg2, %arg4) : (tensor<!pphlo.pub<i32>>, tensor<!pphlo.pub<i32>>) -> tensor<!pphlo.pub<i32>>
      %2 = "pphlo.add"(%arg3, %arg5) : (tensor<!pphlo.pub<i32>>, tensor<!pphlo.pub<i32>>) -> tensor<!pphlo.pub<i32>>
      %3 = "pphlo.less" (%1, %2) : (tensor<!pphlo.pub<i32>>, tensor<!pphlo.pub<i32>>) -> tensor<!pphlo.pub<i1>>
      "pphlo.return"(%3) : (tensor<!pphlo.pub<i1>>) -> ()
    }) {dimension = 0 : i64, is_stable = true} : (tensor<4x!pphlo.pub<i32>>, tensor<4x!pphlo.pub<i32>>) -> (tensor<4x!pphlo.pub<i32>>, tensor<4x!pphlo.pub<i32>>)
    return %0#0, %0#1 : tensor<4x!pphlo.pub<i32>>, tensor<4x!pphlo.pub<i32>>
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
func.func @main(%arg0: tensor<4x!pphlo.sec<i32>>, %arg1: tensor<4x!pphlo.sec<i32>>) -> (tensor<4x!pphlo.sec<i32>>, tensor<4x!pphlo.sec<i32>>) {
    %0:2 = "pphlo.sort"(%arg0, %arg1) ( {
    ^bb0(%arg2: tensor<!pphlo.sec<i32>>, %arg3: tensor<!pphlo.sec<i32>>, %arg4: tensor<!pphlo.sec<i32>>, %arg5: tensor<!pphlo.sec<i32>>):  // no predecessors
      %1 = "pphlo.add"(%arg2, %arg4) : (tensor<!pphlo.sec<i32>>, tensor<!pphlo.sec<i32>>) -> tensor<!pphlo.sec<i32>>
      %2 = "pphlo.add"(%arg3, %arg5) : (tensor<!pphlo.sec<i32>>, tensor<!pphlo.sec<i32>>) -> tensor<!pphlo.sec<i32>>
      %3 = "pphlo.less" (%1, %2) : (tensor<!pphlo.sec<i32>>, tensor<!pphlo.sec<i32>>) -> tensor<!pphlo.sec<i1>>
      "pphlo.return"(%3) : (tensor<!pphlo.sec<i1>>) -> ()
    }) {dimension = 0 : i64, is_stable = true} : (tensor<4x!pphlo.sec<i32>>, tensor<4x!pphlo.sec<i32>>) -> (tensor<4x!pphlo.sec<i32>>, tensor<4x!pphlo.sec<i32>>)
    return %0#0, %0#1 : tensor<4x!pphlo.sec<i32>>, tensor<4x!pphlo.sec<i32>>
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
func.func @main(%arg0: tensor<3x!pphlo.pub<f32>>, %arg1: tensor<3x!pphlo.pub<f32>>) -> (tensor<3x!pphlo.pub<f32>>) {
  %0 = "pphlo.remainder"(%arg0, %arg1) : (tensor<3x!pphlo.pub<f32>>, tensor<3x!pphlo.pub<f32>>) -> tensor<3x!pphlo.pub<f32>>
  return %0 : tensor<3x!pphlo.pub<f32>>
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
func.func @main(%arg0: tensor<8x!pphlo.pub<i32>>, %arg1: tensor<8x!pphlo.pub<i32>>) -> (tensor<8x!pphlo.pub<i32>>) {
  %0 = "pphlo.remainder"(%arg0, %arg1) : (tensor<8x!pphlo.pub<i32>>, tensor<8x!pphlo.pub<i32>>) -> tensor<8x!pphlo.pub<i32>>
  return %0 : tensor<8x!pphlo.pub<i32>>
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
func.func @main(%arg0: tensor<8x!pphlo.pub<i32>>, %arg1: tensor<8x!pphlo.pub<i32>>) -> (tensor<8x!pphlo.pub<i32>>) {
  %0 = "pphlo.shift_left"(%arg0, %arg1) : (tensor<8x!pphlo.pub<i32>>, tensor<8x!pphlo.pub<i32>>) -> tensor<8x!pphlo.pub<i32>>
  return %0 : tensor<8x!pphlo.pub<i32>>
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
func.func @main(%arg0: tensor<8x!pphlo.pub<ui32>>, %arg1: tensor<8x!pphlo.pub<ui32>>) -> (tensor<8x!pphlo.pub<ui32>>) {
  %0 = "pphlo.shift_left"(%arg0, %arg1) : (tensor<8x!pphlo.pub<ui32>>, tensor<8x!pphlo.pub<ui32>>) -> tensor<8x!pphlo.pub<ui32>>
  return %0 : tensor<8x!pphlo.pub<ui32>>
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
func.func @main(%arg0: tensor<8x!pphlo.pub<i32>>, %arg1: tensor<8x!pphlo.pub<i32>>) -> (tensor<8x!pphlo.pub<i32>>) {
  %0 = "pphlo.shift_right_logical"(%arg0, %arg1) : (tensor<8x!pphlo.pub<i32>>, tensor<8x!pphlo.pub<i32>>) -> tensor<8x!pphlo.pub<i32>>
  return %0 : tensor<8x!pphlo.pub<i32>>
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
func.func @main(%arg0: tensor<8x!pphlo.pub<ui32>>, %arg1: tensor<8x!pphlo.pub<ui32>>) -> (tensor<8x!pphlo.pub<ui32>>) {
  %0 = "pphlo.shift_right_logical"(%arg0, %arg1) : (tensor<8x!pphlo.pub<ui32>>, tensor<8x!pphlo.pub<ui32>>) -> tensor<8x!pphlo.pub<ui32>>
  return %0 : tensor<8x!pphlo.pub<ui32>>
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
func.func @main(%arg0: tensor<8x!pphlo.pub<i32>>, %arg1: tensor<8x!pphlo.pub<i32>>) -> (tensor<8x!pphlo.pub<i32>>) {
  %0 = "pphlo.shift_right_arithmetic"(%arg0, %arg1) : (tensor<8x!pphlo.pub<i32>>, tensor<8x!pphlo.pub<i32>>) -> tensor<8x!pphlo.pub<i32>>
  return %0 : tensor<8x!pphlo.pub<i32>>
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
func.func @main(%arg0: tensor<8x!pphlo.pub<ui32>>, %arg1: tensor<8x!pphlo.pub<ui32>>) -> (tensor<8x!pphlo.pub<ui32>>) {
  %0 = "pphlo.shift_right_arithmetic"(%arg0, %arg1) : (tensor<8x!pphlo.pub<ui32>>, tensor<8x!pphlo.pub<ui32>>) -> tensor<8x!pphlo.pub<ui32>>
  return %0 : tensor<8x!pphlo.pub<ui32>>
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
func.func @main(%arg0: tensor<8x!pphlo.sec<ui32>>, %arg1: tensor<8x!pphlo.sec<ui32>>) -> (tensor<8x!pphlo.sec<ui32>>) {
  %0 = "pphlo.shift_right_arithmetic"(%arg0, %arg1) : (tensor<8x!pphlo.sec<ui32>>, tensor<8x!pphlo.sec<ui32>>) -> tensor<8x!pphlo.sec<ui32>>
  return %0 : tensor<8x!pphlo.sec<ui32>>
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
func.func @main(%arg0: tensor<12x!pphlo.pub<i32>>, %arg1: tensor<60x!pphlo.pub<i32>>) -> (tensor<3x5x!pphlo.pub<i32>>) {
  %0 = "pphlo.reshape" (%arg0) : (tensor<12x!pphlo.pub<i32>>) -> tensor<3x1x4x!pphlo.pub<i32>>
  %1 = "pphlo.reshape" (%arg1) : (tensor<60x!pphlo.pub<i32>>) -> tensor<3x4x5x!pphlo.pub<i32>>
  %2 = "pphlo.dot_general"(%0, %1) {dot_dimension_numbers = #pphlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<3x1x4x!pphlo.pub<i32>>, tensor<3x4x5x!pphlo.pub<i32>>) -> tensor<3x5x!pphlo.pub<i32>>
  return %2 : tensor<3x5x!pphlo.pub<i32>>
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
func.func @main(%arg0: tensor<4x6x!pphlo.pub<i32>>, %arg1: tensor<2x2x!pphlo.pub<i32>>, %arg2: tensor<!pphlo.pub<i32>>) -> (tensor<4x6x!pphlo.pub<i32>>) {
  %0 = "pphlo.select_and_scatter"(%arg0, %arg1, %arg2) ({
    ^bb0(%arg3: tensor<!pphlo.pub<i32>>, %arg4: tensor<!pphlo.pub<i32>>):
      %1 = "pphlo.less"(%arg3, %arg4) : (tensor<!pphlo.pub<i32>>, tensor<!pphlo.pub<i32>>) -> tensor<!pphlo.pub<i1>>
      %2 = "pphlo.not"(%1) : (tensor<!pphlo.pub<i1>>) -> tensor<!pphlo.pub<i1>>
      "pphlo.return"(%2) : (tensor<!pphlo.pub<i1>>) -> ()
    }, {
    ^bb0(%arg3: tensor<!pphlo.pub<i32>>, %arg4: tensor<!pphlo.pub<i32>>):
      %1 = "pphlo.add"(%arg3, %arg4) : (tensor<!pphlo.pub<i32>>, tensor<!pphlo.pub<i32>>) -> tensor<!pphlo.pub<i32>>
      "pphlo.return"(%1) : (tensor<!pphlo.pub<i32>>) -> ()
    }) {padding = dense<0> : tensor<2x2xi64>, window_dimensions = dense<[2,3]> : tensor<2xi64>, window_strides = dense<[2,3]> : tensor<2xi64>} : (tensor<4x6x!pphlo.pub<i32>>, tensor<2x2x!pphlo.pub<i32>>, tensor<!pphlo.pub<i32>>) -> tensor<4x6x!pphlo.pub<i32>>
    return %0 : tensor<4x6x!pphlo.pub<i32>>
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
func.func @main(%arg0: tensor<4x5x!pphlo.pub<i32>>, %arg1: tensor<2x2x!pphlo.pub<i32>>, %arg2: tensor<!pphlo.pub<i32>>) -> (tensor<4x5x!pphlo.pub<i32>>) {
  %0 = "pphlo.select_and_scatter"(%arg0, %arg1, %arg2) ({
    ^bb0(%arg3: tensor<!pphlo.pub<i32>>, %arg4: tensor<!pphlo.pub<i32>>):
      %1 = "pphlo.less"(%arg3, %arg4) : (tensor<!pphlo.pub<i32>>, tensor<!pphlo.pub<i32>>) -> tensor<!pphlo.pub<i1>>
      %2 = "pphlo.not"(%1) : (tensor<!pphlo.pub<i1>>) -> tensor<!pphlo.pub<i1>>
      "pphlo.return"(%2) : (tensor<!pphlo.pub<i1>>) -> ()
    }, {
    ^bb0(%arg3: tensor<!pphlo.pub<i32>>, %arg4: tensor<!pphlo.pub<i32>>):
      %1 = "pphlo.add"(%arg3, %arg4) : (tensor<!pphlo.pub<i32>>, tensor<!pphlo.pub<i32>>) -> tensor<!pphlo.pub<i32>>
      "pphlo.return"(%1) : (tensor<!pphlo.pub<i32>>) -> ()
    }) {padding = dense<0> : tensor<2x2xi64>, window_dimensions = dense<[2,3]> : tensor<2xi64>, window_strides = dense<[2,2]> : tensor<2xi64>} : (tensor<4x5x!pphlo.pub<i32>>, tensor<2x2x!pphlo.pub<i32>>, tensor<!pphlo.pub<i32>>) -> tensor<4x5x!pphlo.pub<i32>>
    return %0 : tensor<4x5x!pphlo.pub<i32>>
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

  r.addInput(xt::xarray<int8_t>{
      {{0, 0, 0, 0, 0, 1}, {0, 1, 0, 0, 0, 0}},  //
      {{0, 0, 1, 0, 0, 0}, {0, 0, 0, 0, 0, 1}},  //
  });
  r.addInput(xt::xarray<int32_t>{
      {2, 6},  //
      {3, 1}   //
  });

  r.run(R"(
func.func @main(%arg0: tensor<2x2x6x!pphlo.pub<i8>>, %arg1: tensor<2x2x!pphlo.pub<i32>>) -> (tensor<4x6x!pphlo.pub<i32>>) {
    %0 = "pphlo.maxpool_scatter"(%arg0, %arg1) {padding = dense<0> : tensor<2x2xi64>, window_dimensions = dense<[2,3]> : tensor<2xi64>, window_strides = dense<[2,3]> : tensor<2xi64>} : (tensor<2x2x6x!pphlo.pub<i8>>, tensor<2x2x!pphlo.pub<i32>>) -> tensor<4x6x!pphlo.pub<i32>>
    return %0 : tensor<4x6x!pphlo.pub<i32>>
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

  r.addInput(xt::xarray<int8_t>{
      {{0, 0, 0, 0, 0, 1}, {0, 0, 0, 1, 0, 0}},  //
      {{0, 0, 1, 0, 0, 0}, {0, 0, 0, 0, 1, 0}},  //
  });

  r.addInput(xt::xarray<int32_t>{
      {2, 6},  //
      {3, 1}   //
  });

  r.run(R"(
func.func @main(%arg0: tensor<2x2x6x!pphlo.pub<i8>>, %arg1: tensor<2x2x!pphlo.pub<i32>>) -> (tensor<4x5x!pphlo.pub<i32>>) {
    %0 = "pphlo.maxpool_scatter"(%arg0, %arg1) {padding = dense<0> : tensor<2x2xi64>, window_dimensions = dense<[2,3]> : tensor<2xi64>, window_strides = dense<[2,2]> : tensor<2xi64>} : (tensor<2x2x6x!pphlo.pub<i8>>, tensor<2x2x!pphlo.pub<i32>>) -> tensor<4x5x!pphlo.pub<i32>>
    return %0 : tensor<4x5x!pphlo.pub<i32>>
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
func.func @main(%arg0: tensor<4x6x!pphlo.pub<i32>>) -> (tensor<2x2x!pphlo.pub<i32>>, tensor<2x2x6x!pphlo.pub<i1>>) {
    %4:2 = "pphlo.argmax"(%arg0) {base_dilations = dense<1> : tensor<4xi64>, padding = dense<0> : tensor<2x2xi64>, window_dilations = dense<1> : tensor<2xi64>, window_dimensions = dense<[2, 3]> : tensor<2xi64>, window_strides = dense<[2, 3]> : tensor<2xi64>} : (tensor<4x6x!pphlo.pub<i32>>) -> (tensor<2x2x!pphlo.pub<i32>>, tensor<2x2x6x!pphlo.pub<i1>>)
    return %4#0, %4#1: tensor<2x2x!pphlo.pub<i32>>, tensor<2x2x6x!pphlo.pub<i1>>
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
func.func @main(%arg0: tensor<4x5x!pphlo.pub<i32>>) -> (tensor<2x2x!pphlo.pub<i32>>, tensor<2x2x6x!pphlo.pub<i1>>) {
    %4:2 = "pphlo.argmax"(%arg0) {base_dilations = dense<1> : tensor<4xi64>, padding = dense<0> : tensor<2x2xi64>, window_dilations = dense<1> : tensor<2xi64>, window_dimensions = dense<[2, 3]> : tensor<2xi64>, window_strides = dense<[2, 2]> : tensor<2xi64>} : (tensor<4x5x!pphlo.pub<i32>>) -> (tensor<2x2x!pphlo.pub<i32>>, tensor<2x2x6x!pphlo.pub<i1>>)
    return %4#0, %4#1: tensor<2x2x!pphlo.pub<i32>>, tensor<2x2x6x!pphlo.pub<i1>>
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
  %0 = mhlo.constant dense<0> : tensor<i32>
  %1 = "mhlo.reduce_window"(%arg0, %0) ({
    ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>):
      %3 = mhlo.maximum %arg2, %arg3 : tensor<i32>
      "mhlo.return"(%3) : (tensor<i32>) -> ()
    }) {base_dilations = dense<1> : tensor<2xi64>, padding = dense<0> : tensor<2x2xi64>, window_dilations = dense<1> : tensor<2xi64>,
        window_dimensions = dense<[2,3]> : tensor<2xi64>, window_strides = dense<[2, 3]> : tensor<2xi64>} : (tensor<4x6xi32>, tensor<i32>) -> tensor<2x2xi32>
  %2 = "mhlo.select_and_scatter"(%arg0, %arg1, %0) ({
    ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):
      %3 = "mhlo.compare"(%arg3, %arg4) {comparison_direction = #mhlo<comparison_direction GE>} : (tensor<i32>, tensor<i32>) -> tensor<i1>
      "mhlo.return"(%3) : (tensor<i1>) -> ()
    }, {
    ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):
      %3 = mhlo.add %arg3, %arg4 : tensor<i32>
      "mhlo.return"(%3) : (tensor<i32>) -> ()
    }) {padding = dense<0> : tensor<2x2xi64>, window_dimensions = dense<[2,3]> : tensor<2xi64>, window_strides = dense<[2,3]> : tensor<2xi64>} : (tensor<4x6xi32>, tensor<2x2xi32>, tensor<i32>) -> tensor<4x6xi32>
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
func.func @main(%arg0: tensor<1x4x4x1x!pphlo.pub<i32>>, %arg1: tensor<1x3x3x1x!pphlo.pub<i32>>) -> (tensor<1x3x3x1x!pphlo.pub<i32>>, tensor<1x3x3x1x4x!pphlo.pub<i1>>, tensor<1x4x4x1x!pphlo.pub<i32>>) {
    %0:2 = "pphlo.argmax"(%arg0) {base_dilations = dense<1> : tensor<4xi64>, padding = dense<0> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 2, 2, 1]> : tensor<4xi64>, window_strides = dense<1> : tensor<4xi64>} : (tensor<1x4x4x1x!pphlo.pub<i32>>) -> (tensor<1x3x3x1x!pphlo.pub<i32>>, tensor<1x3x3x1x4x!pphlo.pub<i1>>)
    %1 = "pphlo.maxpool_scatter"(%0#1, %arg1) {padding = dense<0> : tensor<4x2xi64>, window_dimensions = dense<[1, 2, 2, 1]> : tensor<4xi64>, window_strides = dense<1> : tensor<4xi64>} : (tensor<1x3x3x1x4x!pphlo.pub<i1>>, tensor<1x3x3x1x!pphlo.pub<i32>>) -> tensor<1x4x4x1x!pphlo.pub<i32>>
    return %0#0, %0#1, %1: tensor<1x3x3x1x!pphlo.pub<i32>>, tensor<1x3x3x1x4x!pphlo.pub<i1>>, tensor<1x4x4x1x!pphlo.pub<i32>>
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
func.func @main() -> (tensor<!pphlo.pub<i32>>, tensor<!pphlo.pub<i1>>) {
    %0 = "pphlo.constant"() {value = dense<5> : tensor<i32>} : () -> tensor<!pphlo.pub<i32>>
    %1 = "pphlo.not"(%0) : (tensor<!pphlo.pub<i32>>) -> tensor<!pphlo.pub<i32>>
    %2 = "pphlo.constant"() {value = dense<0> : tensor<i1>} : () -> tensor<!pphlo.pub<i1>>
    %3 = "pphlo.not"(%2) : (tensor<!pphlo.pub<i1>>) -> tensor<!pphlo.pub<i1>>
    return %1, %3: tensor<!pphlo.pub<i32>>, tensor<!pphlo.pub<i1>>
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
 func.func @main(%arg0: tensor<!pphlo.pub<i32>>) -> (tensor<!pphlo.pub<i32>>,tensor<!pphlo.pub<i32>>) {
  %0:2 = "pphlo.case"(%arg0) ({
    %1 = "pphlo.constant"() {value = dense<1> : tensor<i32>} : () -> tensor<!pphlo.pub<i32>>
    %2 = "pphlo.constant"() {value = dense<11> : tensor<i32>} : () -> tensor<!pphlo.pub<i32>>
    "pphlo.return"(%1, %2) : (tensor<!pphlo.pub<i32>>, tensor<!pphlo.pub<i32>>) -> ()
  }, {
    %1 = "pphlo.constant"() {value = dense<2> : tensor<i32>} : () -> tensor<!pphlo.pub<i32>>
    %2 = "pphlo.constant"() {value = dense<12> : tensor<i32>} : () -> tensor<!pphlo.pub<i32>>
    "pphlo.return"(%1, %2) : (tensor<!pphlo.pub<i32>>, tensor<!pphlo.pub<i32>>) -> ()
  }, {
    %1 = "pphlo.constant"() {value = dense<3> : tensor<i32>} : () -> tensor<!pphlo.pub<i32>>
    %2 = "pphlo.constant"() {value = dense<13> : tensor<i32>} : () -> tensor<!pphlo.pub<i32>>
    "pphlo.return"(%1, %2) : (tensor<!pphlo.pub<i32>>, tensor<!pphlo.pub<i32>>) -> ()
  }) : (tensor<!pphlo.pub<i32>>) -> (tensor<!pphlo.pub<i32>>, tensor<!pphlo.pub<i32>>)
  return %0#0, %0#1: tensor<!pphlo.pub<i32>>, tensor<!pphlo.pub<i32>>
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
 func.func @main(%arg0: tensor<!pphlo.sec<i32>>) -> (tensor<!pphlo.sec<i32>>, tensor<!pphlo.sec<i32>>) {
  %0:2 = "pphlo.case"(%arg0) ({
    %1 = "pphlo.constant"() {value = dense<1> : tensor<i32>} : () -> tensor<!pphlo.pub<i32>>
    %2 = "pphlo.convert"(%1) : (tensor<!pphlo.pub<i32>>) -> tensor<!pphlo.sec<i32>>
    %3 = "pphlo.constant"() {value = dense<11> : tensor<i32>} : () -> tensor<!pphlo.pub<i32>>
    %4 = "pphlo.convert"(%3) : (tensor<!pphlo.pub<i32>>) -> tensor<!pphlo.sec<i32>>
    "pphlo.return"(%2, %4) : (tensor<!pphlo.sec<i32>>, tensor<!pphlo.sec<i32>>) -> ()
  }, {
    %1 = "pphlo.constant"() {value = dense<2> : tensor<i32>} : () -> tensor<!pphlo.pub<i32>>
    %2 = "pphlo.convert"(%1) : (tensor<!pphlo.pub<i32>>) -> tensor<!pphlo.sec<i32>>
    %3 = "pphlo.constant"() {value = dense<12> : tensor<i32>} : () -> tensor<!pphlo.pub<i32>>
    %4 = "pphlo.convert"(%3) : (tensor<!pphlo.pub<i32>>) -> tensor<!pphlo.sec<i32>>
    "pphlo.return"(%2, %4) : (tensor<!pphlo.sec<i32>>, tensor<!pphlo.sec<i32>>) -> ()
  }, {
    %1 = "pphlo.constant"() {value = dense<3> : tensor<i32>} : () -> tensor<!pphlo.pub<i32>>
    %2 = "pphlo.convert"(%1) : (tensor<!pphlo.pub<i32>>) -> tensor<!pphlo.sec<i32>>
    %3 = "pphlo.constant"() {value = dense<13> : tensor<i32>} : () -> tensor<!pphlo.pub<i32>>
    %4 = "pphlo.convert"(%3) : (tensor<!pphlo.pub<i32>>) -> tensor<!pphlo.sec<i32>>
    "pphlo.return"(%2, %4) : (tensor<!pphlo.sec<i32>>, tensor<!pphlo.sec<i32>>) -> ()
  }) : (tensor<!pphlo.sec<i32>>) -> (tensor<!pphlo.sec<i32>>, tensor<!pphlo.sec<i32>>)
  return %0#0, %0#1: tensor<!pphlo.sec<i32>>, tensor<!pphlo.sec<i32>>
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
  xt::xarray<int32_t> op = {10, 9, 8, 7, 6,  5,  4,  3,  2,  1,
                            99, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  xt::xarray<int32_t> expected_ret0 = {1, 2, 3,  4,  5,  6,  7,  7,  8,  8,
                                       9, 9, 10, 10, 11, 12, 13, 14, 15, 99};
  xt::xarray<int32_t> expected_ret1 = {9, 8,  7,  6, 5,  4,  3,  11, 12, 2,
                                       1, 13, 14, 0, 15, 16, 17, 18, 19, 10};

  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  r.addInput(op, VIS_SECRET);

  r.run(r.compileMHlo(
            R"(
func.func @main(%arg0: tensor<20xi32>) -> (tensor<20xi32>, tensor<20xi32>) {
    %0 = "mhlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<20xi32>
    %1:2 = "mhlo.sort"(%arg0, %0) ({
    ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<i32>, %arg4: tensor<i32>):
      %2 = mhlo.compare  LT, %arg1, %arg2 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      mhlo.return %2 : tensor<i1>
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
