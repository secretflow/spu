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
#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "xtensor/xrandom.hpp"

#include "spu/device/pphlo_executor.h"
#include "spu/device/symbol_table.h"
#include "spu/device/test_utils.h"
#include "spu/mpc/ref2k/ref2k.h"
#include "spu/mpc/util/simulate.h"

namespace spu::device {
namespace {

class Runner {
public:
  Runner(size_t world_size, FieldType field, ProtocolKind protocol)
      : world_size_(world_size) {
    config_.set_field(field);
    config_.set_protocol(protocol);
    config_.set_enable_type_checker(true);
    io_ = std::make_unique<LocalIo>(world_size_, config_);
  }

  auto &getConfig() { return config_; }

  template <typename T>
  void addInput(const T &input, Visibility vis = Visibility::VIS_PUBLIC) {
    const std::string name = fmt::format("input{}", input_idx_++);
    io_->InFeed(name, input, vis);
    exec_.add_input_names(name);
  }

  void run(const std::string &mlir, size_t num_output = 1) {
    for (size_t idx = 0; idx < num_output; ++idx) {
      exec_.add_output_names(fmt::format("output{}", idx));
    }
    exec_.set_code(mlir);
    ::spu::mpc::util::simulate(
        world_size_, [&](const std::shared_ptr<yasl::link::Context> &lctx) {
          RuntimeConfig conf;
          conf.CopyFrom(config_);
          if (lctx->Rank() == 0) {
            // conf.set_enable_action_trace(true);
          }
          HalContext hctx(conf, lctx);
          PPHloExecutor executor(&hctx);
          auto *env = io_->GetSymbolTable(lctx->Rank());
          executor.runWithEnv(exec_, env);
        });
  }

  template <typename T>
  void verifyOutput(const T *expected, size_t idx = 0) {
    const auto &out = io_->OutFeed(fmt::format("output{}", idx));

    size_t numel = out.numel();
    const auto *in_ptr = static_cast<const T *>(out.data());

    // TODO: handle strides
    for (size_t i = 0; i < numel; ++i) {
      if constexpr (std::is_integral_v<T>) {
        EXPECT_EQ(in_ptr[i], expected[i]) << "i = " << i << "\n";
      } else {
        EXPECT_TRUE(std::abs(in_ptr[i] - expected[i]) <= 1e-2)
            << "i = " << i << " in = " << in_ptr[i]
            << " expected = " << expected[i] << "\n";
      }
    }
  }

  template <typename T, std::enable_if_t<std::is_scalar_v<T>, bool> = true>
  void verifyScalarOutput(T expected, size_t idx = 0) {
    verifyOutput(&expected, idx);
  }

private:
  size_t world_size_;
  RuntimeConfig config_;
  std::unique_ptr<LocalIo> io_;
  size_t input_idx_{0};
  ExecutableProto exec_;
};

} // namespace

class ProcessorTest : public ::testing::TestWithParam<
                          std::tuple<size_t, FieldType, ProtocolKind>> {};

TEST_P(ProcessorTest, Basic) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));
  r.addInput(1);
  r.addInput(2);

  r.run(R"(
func @main(%arg0: tensor<!pphlo.pub<i32>>, %arg1: tensor<!pphlo.pub<i32>>) -> (tensor<!pphlo.pub<i32>>) {
  %0 = "pphlo.add"(%arg0, %arg1) : (tensor<!pphlo.pub<i32>>, tensor<!pphlo.pub<i32>>) -> tensor<!pphlo.pub<i32>>
  return %0 : tensor<!pphlo.pub<i32>>
})");

  r.verifyScalarOutput(3);
}

TEST_P(ProcessorTest, WithConst) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));
  r.addInput(xt::xarray<int>({{1, 1}, {1, 1}}));

  r.run(R"(
func @main(%arg0: tensor<2x2x!pphlo.pub<i32>>) -> (tensor<2x2x!pphlo.pub<i32>>) {
    %0 = "pphlo.constant"() {value = dense<[[1,2],[3,4]]> : tensor<2x2xi32>} : () -> tensor<2x2x!pphlo.pub<i32>>
    %1 = "pphlo.add"(%arg0, %0) : (tensor<2x2x!pphlo.pub<i32>>, tensor<2x2x!pphlo.pub<i32>>) -> tensor<2x2x!pphlo.pub<i32>>
    return %1 : tensor<2x2x!pphlo.pub<i32>>
})");

  std::array<int, 4> expect = {2, 3, 4, 5};
  r.verifyOutput(expect.data());
}

TEST_P(ProcessorTest, RowConcate) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));
  r.addInput(xt::xarray<int>({{1, 2, 3}, {4, 5, 6}}));
  r.addInput(xt::xarray<int>({{7, 8, 9}, {10, 11, 12}}));

  r.run(R"(
func @main(%arg0: tensor<2x3x!pphlo.pub<i32>>, %arg1: tensor<2x3x!pphlo.pub<i32>>) -> (tensor<4x3x!pphlo.pub<i32>>) {
  %0 = "pphlo.concatenate"(%arg0, %arg1) {dimension = 0 : i64} : (tensor<2x3x!pphlo.pub<i32>>, tensor<2x3x!pphlo.pub<i32>>) -> tensor<4x3x!pphlo.pub<i32>>
  return %0 : tensor<4x3x!pphlo.pub<i32>>
})");

  std::array<int, 12> expect = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  r.verifyOutput(expect.data());
}

TEST_P(ProcessorTest, ColConcate) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));
  r.addInput(xt::xarray<int>({{1, 2, 3}, {4, 5, 6}}));
  r.addInput(xt::xarray<int>({{7, 8, 9}, {10, 11, 12}}));

  r.run(R"(
func @main(%arg0: tensor<2x3x!pphlo.pub<i32>>, %arg1: tensor<2x3x!pphlo.pub<i32>>) -> (tensor<2x6x!pphlo.pub<i32>>) {
  %0 = "pphlo.concatenate"(%arg0, %arg1) {dimension = 1 : i64} : (tensor<2x3x!pphlo.pub<i32>>, tensor<2x3x!pphlo.pub<i32>>) -> tensor<2x6x!pphlo.pub<i32>>
  return %0 : tensor<2x6x!pphlo.pub<i32>>
}
  )");

  std::array<int, 12> expect = {1, 2, 3, 7, 8, 9, 4, 5, 6, 10, 11, 12};
  r.verifyOutput(expect.data());
}

TEST_P(ProcessorTest, VariadicConcate) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));
  r.addInput(xt::xarray<int>({1, 2, 3}));

  r.run(R"(
func @main(%arg0: tensor<3x!pphlo.pub<i32>>) -> (tensor<9x!pphlo.pub<i32>>) {
  %0 = "pphlo.concatenate"(%arg0, %arg0, %arg0) {dimension = 0 : i64} : (tensor<3x!pphlo.pub<i32>>, tensor<3x!pphlo.pub<i32>>, tensor<3x!pphlo.pub<i32>>) -> tensor<9x!pphlo.pub<i32>>
  return %0 : tensor<9x!pphlo.pub<i32>>
})");

  std::array<int, 12> expect = {1, 2, 3, 1, 2, 3, 1, 2, 3};
  r.verifyOutput(expect.data());
}

TEST_P(ProcessorTest, Slice) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));
  r.addInput(xt::xarray<int>({{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}}));

  r.run(R"(
func @main(%arg0: tensor<4x3x!pphlo.pub<i32>>) -> (tensor<2x2x!pphlo.pub<i32>>) {
  %0 = "pphlo.slice"(%arg0) {limit_indices = dense<[4, 5]> : tensor<2xi64>, start_indices = dense<[2, 1]> : tensor<2xi64>, strides = dense<[1, 1]> : tensor<2xi64>} : (tensor<4x3x!pphlo.pub<i32>>) -> tensor<2x2x!pphlo.pub<i32>>
  return %0 : tensor<2x2x!pphlo.pub<i32>>
})");

  std::array<int, 4> expect = {7, 8, 10, 11};
  r.verifyOutput(expect.data());
}

TEST_P(ProcessorTest, SliceStride) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));
  r.addInput(xt::xarray<int>({                          //
                              {0, 1, 2, 3, 4, 5},       //
                              {6, 7, 8, 9, 10, 11},     //
                              {12, 13, 14, 15, 16, 17}, //
                              {18, 19, 20, 21, 22, 23}}));

  r.run(R"(
func @main(%arg0: tensor<4x6x!pphlo.pub<i32>>) -> (tensor<2x3x!pphlo.pub<i32>>) {
  %0 = "pphlo.slice"(%arg0) {limit_indices = dense<[4, 7]> : tensor<2xi64>, start_indices = dense<[0, 0]> : tensor<2xi64>, strides = dense<[2, 2]> : tensor<2xi64>} : (tensor<4x6x!pphlo.pub<i32>>) -> tensor<2x3x!pphlo.pub<i32>>
  return %0 : tensor<2x3x!pphlo.pub<i32>>
})");

  std::array<int, 6> expect = {0,  2,  4, //
                               12, 14, 16};
  r.verifyOutput(expect.data());
}

TEST_P(ProcessorTest, Reshape) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));
  r.addInput(xt::xarray<int>({{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}}));

  // Reshape to 2x6
  r.run(R"(
func @main(%arg0: tensor<4x3x!pphlo.pub<i32>>) -> (tensor<2x6x!pphlo.pub<i32>>) {
  %0 = "pphlo.reshape"(%arg0) : (tensor<4x3x!pphlo.pub<i32>>) -> tensor<2x6x!pphlo.pub<i32>>
  return %0 : tensor<2x6x!pphlo.pub<i32>>
}
  )");

  std::array<int, 12> expected = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  r.verifyOutput(expected.data());
}

TEST_P(ProcessorTest, While) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));
  r.addInput(1);
  r.addInput(3);

  // while(x < y) { x = x + 1; }
  r.run(R"(
func @main(%arg0: tensor<!pphlo.pub<i32>>, %arg1: tensor<!pphlo.pub<i32>>) -> tensor<!pphlo.pub<i32>> {
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

TEST_P(ProcessorTest, Reduce1D) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  const xt::xarray<int> in1({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  r.addInput(in1);

  r.run(R"(
func @main(%arg0: tensor<10x!pphlo.pub<i32>>) -> (tensor<!pphlo.pub<i32>>) {
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

TEST_P(ProcessorTest, Reduce2D1) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  const xt::xarray<int> in1({{1, 2, 3}, {4, 5, 6}});
  r.addInput(in1);

  r.run(R"(
func @main(%arg0: tensor<2x3x!pphlo.pub<i32>>) -> (tensor<2x!pphlo.pub<i32>>) {
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

TEST_P(ProcessorTest, Reduce2D2) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  const xt::xarray<int> in1({{1, 2, 3}, {4, 5, 6}});
  r.addInput(in1);

  r.run(R"(
func @main(%arg0: tensor<2x3x!pphlo.pub<i32>>) -> (tensor<3x!pphlo.pub<i32>>) {
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

TEST_P(ProcessorTest, VReduce) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  const xt::xarray<int> in1({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  r.addInput(in1);

  r.run(R"(
func @main(%arg0: tensor<10x!pphlo.pub<i32>>) -> (tensor<!pphlo.pub<i32>>, tensor<!pphlo.pub<i32>>) {
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

TEST_P(ProcessorTest, MaxReduce) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  const xt::xarray<float> in1({{0, 0, 0, 0, 0, 0, 0, 0, 0, 0}});
  r.addInput(in1);

  r.run(R"(
func @main(%arg0: tensor<1x10x!pphlo.pub<f32>>) -> (tensor<1x!pphlo.pub<f32>>) {
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

TEST_P(ProcessorTest, ReduceWindow) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  const xt::xarray<int> in1 = {{-7, 6, 1, -14, -7, 5},
                               {-13, -14, -11, 13, -13, -7},
                               {8, -11, 12, -2, 14, 4},
                               {0, 13, 3, -13, -7, -3}};
  r.addInput(in1);

  r.run(R"(
func @main(%arg0: tensor<4x6x!pphlo.pub<i32>>) -> (tensor<2x2x!pphlo.pub<i32>>) {
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

TEST_P(ProcessorTest, ReduceWindowDefaultStrides) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  const xt::xarray<int> in1 = {{-7, 6, 1, -14, -7, 5},
                               {-13, -14, -11, 13, -13, -7},
                               {8, -11, 12, -2, 14, 4},
                               {0, 13, 3, -13, -7, -3}};
  r.addInput(in1);

  r.run(R"(
func @main(%arg0: tensor<4x6x!pphlo.pub<i32>>) -> (tensor<3x4x!pphlo.pub<i32>>) {
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

TEST_P(ProcessorTest, ReduceWindowIotaWindowDilation) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  const xt::xarray<int> in1 = {
      {0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}, {12, 13, 14, 15}};
  r.addInput(in1);

  r.run(R"(
func @main(%arg0: tensor<4x4x!pphlo.pub<i32>>) -> (tensor<2x2x!pphlo.pub<i32>>) {
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

TEST_P(ProcessorTest, ReduceWindowIotaStrideWindowDilation) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  const xt::xarray<int> in1 = {
      {0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}, {12, 13, 14, 15}};
  r.addInput(in1);

  r.run(R"(
func @main(%arg0: tensor<4x4x!pphlo.pub<i32>>) -> (tensor<1x1x!pphlo.pub<i32>>) {
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

TEST_P(ProcessorTest, ReduceWindowMaxIotaBaseDilation) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  const xt::xarray<int> in1 = {
      {0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}, {12, 13, 14, 15}};
  r.addInput(in1);

  r.run(R"(
func @main(%arg0: tensor<4x4x!pphlo.pub<i32>>) -> (tensor<6x6x!pphlo.pub<i32>>) {
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

TEST_P(ProcessorTest, ReduceWindowMaxIotaStrideBaseDilation) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  const xt::xarray<int> in1 = {
      {0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}, {12, 13, 14, 15}};
  r.addInput(in1);

  r.run(R"(
func @main(%arg0: tensor<4x4x!pphlo.pub<i32>>) -> (tensor<3x3x!pphlo.pub<i32>>) {
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

TEST_P(ProcessorTest, ReduceWindowMaxIotaStrideBothDilation) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  const xt::xarray<int> in1 = {
      {0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}, {12, 13, 14, 15}};
  r.addInput(in1);

  r.run(R"(
func @main(%arg0: tensor<4x4x!pphlo.pub<i32>>) -> (tensor<3x3x!pphlo.pub<i32>>) {
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

TEST_P(ProcessorTest, ReduceWindowMaxIotaPaddingStrideBaseDilation) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  const xt::xarray<int> in1 = {
      {0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}, {12, 13, 14, 15}};
  r.addInput(in1);

  r.run(R"(
func @main(%arg0: tensor<4x4x!pphlo.pub<i32>>) -> (tensor<3x3x!pphlo.pub<i32>>) {
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

TEST_P(ProcessorTest, If) {
  const auto *prog = R"(
 func @main(%arg0: tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>> {
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

    r.verifyScalarOutput(2.5f * 2.5f);
  }

  {
    // False case
    Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
             std::get<2>(GetParam()));

    r.addInput(12.5F);

    r.run(prog);

    r.verifyScalarOutput(12.5f + 12.5f);
  }
}

TEST_P(ProcessorTest, SecretControlflow) {
  const auto *prog = R"(
func @main(%arg0: tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>> {
  %0 = "pphlo.constant"() {value = dense<1.000000e+01> : tensor<f32>} : () -> tensor<!pphlo.pub<f32>>
  %1 = "pphlo.convert"(%arg0) : (tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.sec<f32>>
  %2 = "pphlo.less"(%1, %0) : (tensor<!pphlo.sec<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.sec<i1>>
  %3 = "pphlo.if"(%2) ( {
    %4 = "pphlo.multiply"(%arg0, %arg0) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    "pphlo.return"(%4) : (tensor<!pphlo.pub<f32>>) -> ()
  },  {
    %4 = "pphlo.add"(%arg0, %arg0) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    "pphlo.return"(%4) : (tensor<!pphlo.pub<f32>>) -> ()
  }) : (tensor<!pphlo.sec<i1>>) -> tensor<!pphlo.pub<f32>>
  return %3 : tensor<!pphlo.pub<f32>>
}
)";
  // default
  {
    Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
             std::get<2>(GetParam()));

    r.addInput(2.5F);

    EXPECT_THROW(r.run(prog), yasl::EnforceNotMet);
  }
  // reveal
  {
    Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
             std::get<2>(GetParam()));

    r.getConfig().set_reveal_secret_condition(true);

    r.addInput(2.5F);

    r.run(prog);

    r.verifyScalarOutput(2.5f * 2.5f);
  }
}

TEST_P(ProcessorTest, Iota1D) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  r.run(R"(
func @main() -> (tensor<4x!pphlo.pub<i32>>) {
    %0 = "pphlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<4x!pphlo.pub<i32>>
    return %0 : tensor<4x!pphlo.pub<i32>>
})");

  std::array<int, 4> expect = {0, 1, 2, 3};
  r.verifyOutput(expect.data());
}

TEST_P(ProcessorTest, Iota2D) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  r.run(R"(
func @main() -> (tensor<4x2x!pphlo.pub<i32>>) {
    %0 = "pphlo.iota"() {iota_dimension = 1 : i64} : () -> tensor<4x2x!pphlo.pub<i32>>
    return %0 : tensor<4x2x!pphlo.pub<i32>>
})");

  std::array<int, 8> expect = {0, 1, 0, 1, 0, 1, 0, 1};
  r.verifyOutput(expect.data());
}

TEST_P(ProcessorTest, SimpleBitcast) {
  GTEST_SKIP();

  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  // the fixed-point bitcast behaves differently then floating point ones.
  float in = 2.0F;
  r.addInput(in);

  r.run(R"(
func @main(%arg0: tensor<!pphlo.pub<f32>>) -> (tensor<!pphlo.pub<i32>>) {
    %0 = "pphlo.bitcast_convert"(%arg0) {elsize = 32 : i64} : (tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<i32>>
    return %0 : tensor<!pphlo.pub<i32>>
})");

  r.verifyOutput(reinterpret_cast<int32_t *>(&in));
}

TEST_P(ProcessorTest, Gather1) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  r.addInput(xt::xarray<int>{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  // Start indices
  r.addInput(xt::xarray<int>{0, 2});

  r.run(R"(
func @main(%arg0: tensor<3x3x!pphlo.pub<i32>>, %arg1: tensor<2x!pphlo.pub<i32>>) -> (tensor<2x3x!pphlo.pub<i32>>) {
    %0 = "pphlo.gather"(%arg0, %arg1) {dimension_numbers = #pphlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 3]> : tensor<2xi64>} : (tensor<3x3x!pphlo.pub<i32>>, tensor<2x!pphlo.pub<i32>>) -> tensor<2x3x!pphlo.pub<i32>>
    return %0 : tensor<2x3x!pphlo.pub<i32>>
})");

  xt::xarray<int> expected = {{1, 2, 3}, {7, 8, 9}};
  r.verifyOutput(expected.data());
}

TEST_P(ProcessorTest, Gather2) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  r.addInput(xt::xarray<int>{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  // Start indices
  r.addInput(xt::xarray<int>{0, 2});

  r.run(R"(
func @main(%arg0: tensor<3x3x!pphlo.pub<i32>>, %arg1: tensor<2x!pphlo.pub<i32>>) -> (tensor<3x2x!pphlo.pub<i32>>) {
    %0 = "pphlo.gather"(%arg0, %arg1) {dimension_numbers = #pphlo.gather<offset_dims = [0], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[3,1]> : tensor<2xi64>} : (tensor<3x3x!pphlo.pub<i32>>, tensor<2x!pphlo.pub<i32>>) -> tensor<3x2x!pphlo.pub<i32>>
    return %0 : tensor<3x2x!pphlo.pub<i32>>
})");

  xt::xarray<int> expected = {{1, 3}, {4, 6}, {7, 9}};
  r.verifyOutput(expected.data());
}

TEST_P(ProcessorTest, GatherBatch) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  r.addInput(xt::xarray<int>{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  // Start indices
  r.addInput(xt::xarray<int>{{0, 2}, {2, 1}});

  r.run(R"(
func @main(%arg0: tensor<3x3x!pphlo.pub<i32>>, %arg1: tensor<2x2x!pphlo.pub<i32>>) -> (tensor<2x3x2x!pphlo.pub<i32>>) {
    %0 = "pphlo.gather"(%arg0, %arg1) {dimension_numbers = #pphlo.gather<offset_dims = [1], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = dense<[3,1]> : tensor<2xi64>} : (tensor<3x3x!pphlo.pub<i32>>, tensor<2x2x!pphlo.pub<i32>>) -> tensor<2x3x2x!pphlo.pub<i32>>
    return %0 : tensor<2x3x2x!pphlo.pub<i32>>
})");

  xt::xarray<int> expected = {{{1, 3}, {4, 6}, {7, 9}},
                              {{3, 2}, {6, 5}, {9, 8}}};
  r.verifyOutput(expected.data());
}

TEST_P(ProcessorTest, GatherNd) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  const xt::xarray<int> in = {{{-1, 1}, {-2, 2}, {-3, 3}},
                              {{-4, 4}, {-5, 5}, {-6, 6}},
                              {{-7, 7}, {-8, 8}, {-9, 9}}};
  r.addInput(in);
  // Start indices
  r.addInput(xt::xarray<int>{{0, 0}, {1, 0}});

  r.run(R"(
func @main(%arg0: tensor<3x3x2x!pphlo.pub<i32>>, %arg1: tensor<2x2x!pphlo.pub<i32>>) -> (tensor<2x2x!pphlo.pub<i32>>) {
    %0 = "pphlo.gather"(%arg0, %arg1) {dimension_numbers = #pphlo.gather<offset_dims = [1], collapsed_slice_dims = [0,1], start_index_map = [0,1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1,1,2]> : tensor<3xi64>} : (tensor<3x3x2x!pphlo.pub<i32>>, tensor<2x2x!pphlo.pub<i32>>) -> tensor<2x2x!pphlo.pub<i32>>
    return %0 : tensor<2x2x!pphlo.pub<i32>>
})");

  xt::xarray<int> expected = {{-1, 1}, {-4, 4}};
  r.verifyOutput(expected.data());
}

TEST_P(ProcessorTest, GatherNdNonDefaultIndexVectorDim) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  xt::xarray<int> in = {{{-1, 1}, {-2, 2}, {-3, 3}},
                        {{-4, 4}, {-5, 5}, {-6, 6}},
                        {{-7, 7}, {-8, 8}, {-9, 9}}};
  r.addInput(in);
  // Start indices
  r.addInput(xt::xarray<int>{{0, 0}, {1, 0}});

  r.run(R"(
func @main(%arg0: tensor<3x3x2x!pphlo.pub<i32>>, %arg1: tensor<2x2x!pphlo.pub<i32>>) -> (tensor<2x2x!pphlo.pub<i32>>) {
    %0 = "pphlo.gather"(%arg0, %arg1) {dimension_numbers = #pphlo.gather<offset_dims = [1], collapsed_slice_dims = [0,1], start_index_map = [0,1], index_vector_dim = 0>, indices_are_sorted = false, slice_sizes = dense<[1,1,2]> : tensor<3xi64>} : (tensor<3x3x2x!pphlo.pub<i32>>, tensor<2x2x!pphlo.pub<i32>>) -> tensor<2x2x!pphlo.pub<i32>>
    return %0 : tensor<2x2x!pphlo.pub<i32>>
})");

  xt::xarray<int> expected = {{-2, 2}, {-1, 1}};
  r.verifyOutput(expected.data());
}

TEST_P(ProcessorTest, Simple4x4Conv2DWith2x2Kernel) {
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

  r.run(R"(
func @main(%arg0: tensor<1x1x4x4x!pphlo.pub<f32>>, %arg1: tensor<1x1x2x2x!pphlo.pub<f32>>) -> (tensor<1x1x4x4x!pphlo.pub<f32>>) {
    %0 = pphlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x1x4x4x!pphlo.pub<f32>>, tensor<1x1x2x2x!pphlo.pub<f32>>) -> tensor<1x1x4x4x!pphlo.pub<f32>>
    return %0 : tensor<1x1x4x4x!pphlo.pub<f32>>
})");

  xt::xarray<float> expected = {{{
      {100, 126, 152, 76},
      {204, 230, 256, 124},
      {308, 334, 360, 172},
      {149, 160, 171, 80},
  }}};
  r.verifyOutput(expected.data());
}

TEST_P(ProcessorTest, Conv2DGeneralDimensions) {
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

  r.run(R"(
func @main(%arg0: tensor<2x3x1x4x!pphlo.pub<f32>>, %arg1: tensor<1x3x2x3x!pphlo.pub<f32>>) -> (tensor<1x1x1x2x!pphlo.pub<f32>>) {
    %0 = pphlo.convolution(%arg0, %arg1) dim_numbers = [f, 0, b, 1]x[o, 1, i, 0]->[f, 0, b, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<2x3x1x4x!pphlo.pub<f32>>, tensor<1x3x2x3x!pphlo.pub<f32>>) -> tensor<1x1x1x2x!pphlo.pub<f32>>
    return %0 : tensor<1x1x1x2x!pphlo.pub<f32>>
})");

  xt::xarray<float> expected = {{{{2514, 2685}}}};
  r.verifyOutput(expected.data());
}

TEST_P(ProcessorTest, DilatedBaseConv2DWithHighPadding) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  xt::xarray<float> lhs = {{{
      {1, 2, 3, 4}, //
      {5, 6, 7, 8},
      {9, 10, 11, 12},
      {13, 14, 15, 16},
  }}};
  r.addInput(lhs);

  xt::xarray<float> rhs = {{{
      {5, 6}, //
      {7, 8},
  }}};

  r.addInput(rhs);

  r.run(R"(
func @main(%arg0: tensor<1x1x4x4x!pphlo.pub<f32>>, %arg1: tensor<1x1x2x2x!pphlo.pub<f32>>) -> (tensor<1x1x7x7x!pphlo.pub<f32>>) {
    %0 = pphlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x1x4x4x!pphlo.pub<f32>>, tensor<1x1x2x2x!pphlo.pub<f32>>) -> tensor<1x1x7x7x!pphlo.pub<f32>>
    return %0 : tensor<1x1x7x7x!pphlo.pub<f32>>
})");

  xt::xarray<float> expected = {{{5, 12, 10, 18, 15, 24, 20},
                                 {35, 48, 42, 56, 49, 64, 56},
                                 {25, 36, 30, 42, 35, 48, 40},
                                 {63, 80, 70, 88, 77, 96, 84},
                                 {45, 60, 50, 66, 55, 72, 60},
                                 {91, 112, 98, 120, 105, 128, 112},
                                 {65, 84, 70, 90, 75, 96, 80}}};
  r.verifyOutput(expected.data());
}

TEST_P(ProcessorTest, DilatedBaseConv2DWithLowAndHighPadding) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  xt::xarray<float> lhs = {{{
      {1, 2, 3, 4}, //
      {5, 6, 7, 8},
      {9, 10, 11, 12},
      {13, 14, 15, 16},
  }}};
  r.addInput(lhs);

  xt::xarray<float> rhs = {{{
      {5, 6}, //
      {7, 8},
  }}};

  r.addInput(rhs);

  r.run(R"(
func @main(%arg0: tensor<1x1x4x4x!pphlo.pub<f32>>, %arg1: tensor<1x1x2x2x!pphlo.pub<f32>>) -> (tensor<1x1x8x8x!pphlo.pub<f32>>) {
    %0 = pphlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x1x4x4x!pphlo.pub<f32>>, tensor<1x1x2x2x!pphlo.pub<f32>>) -> tensor<1x1x8x8x!pphlo.pub<f32>>
    return %0 : tensor<1x1x8x8x!pphlo.pub<f32>>
})");

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

TEST_P(ProcessorTest, FlatRhsDilation) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  xt::xarray<float> lhs = {{{
      {0, 1, 2, 3, 4, 5}, //
      {6, 7, 8, 9, 10, 11},
      {12, 13, 14, 15, 16, 17},
      {18, 19, 20, 21, 22, 23},

  }}};
  r.addInput(lhs);

  xt::xarray<float> rhs = {{{{1, 10, 100}, //
                             {2, 20, 200}}}};

  r.addInput(rhs);

  r.run(R"(
func @main(%arg0: tensor<1x1x4x6x!pphlo.pub<f32>>, %arg1: tensor<1x1x2x3x!pphlo.pub<f32>>) -> (tensor<1x1x2x2x!pphlo.pub<f32>>) {
    %0 = pphlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x1x4x6x!pphlo.pub<f32>>, tensor<1x1x2x3x!pphlo.pub<f32>>) -> tensor<1x1x2x2x!pphlo.pub<f32>>
    return %0 : tensor<1x1x2x2x!pphlo.pub<f32>>
})");

  xt::xarray<float> expected = {{3924, 4257}, {5922, 6255}};
  r.verifyOutput(expected.data());
}

TEST_P(ProcessorTest, ShiftLeft) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  xt::xarray<int> lhs = {1, 1};
  r.addInput(lhs);

  xt::xarray<int> rhs = {1, 2};
  r.addInput(rhs);

  r.run(R"(
func @main(%arg0: tensor<2x!pphlo.pub<i32>>, %arg1: tensor<2x!pphlo.pub<i32>>) -> (tensor<2x!pphlo.pub<i32>>) {
    %0 = "pphlo.shift_left"(%arg0, %arg1) : (tensor<2x!pphlo.pub<i32>>, tensor<2x!pphlo.pub<i32>>) -> tensor<2x!pphlo.pub<i32>>
    return %0 : tensor<2x!pphlo.pub<i32>>
})");

  xt::xarray<int> expected = {1 << 1, 1 << 2};
  r.verifyOutput(expected.data());
}

TEST_P(ProcessorTest, RightShiftLogical) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  xt::xarray<int> lhs = {1 << 4, 1 << 4};
  r.addInput(lhs);

  xt::xarray<int> rhs = {1, 2};
  r.addInput(rhs);

  r.run(R"(
func @main(%arg0: tensor<2x!pphlo.pub<i32>>, %arg1: tensor<2x!pphlo.pub<i32>>) -> (tensor<2x!pphlo.pub<i32>>) {
    %0 = "pphlo.shift_right_logical"(%arg0, %arg1) : (tensor<2x!pphlo.pub<i32>>, tensor<2x!pphlo.pub<i32>>) -> tensor<2x!pphlo.pub<i32>>
    return %0 : tensor<2x!pphlo.pub<i32>>
})");

  xt::xarray<int> expected = {1 << 3, 1 << 2};
  r.verifyOutput(expected.data());
}

TEST_P(ProcessorTest, Maximum) {
  if (std::get<1>(GetParam()) == FM32) {
    return; // Ring type is not large enough to hold value
  }

  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  r.addInput(10);

  r.run(R"(
func @main(%arg0: tensor<!pphlo.pub<i32>>) -> (tensor<!pphlo.pub<i32>>) {
  %0 = "pphlo.constant"() {value = dense<-2147483648> : tensor<i32>} : () -> tensor<!pphlo.pub<i32>>
  %1 = "pphlo.maximum"(%0, %arg0) : (tensor<!pphlo.pub<i32>>, tensor<!pphlo.pub<i32>>) -> tensor<!pphlo.pub<i32>>
  return %1 :  tensor<!pphlo.pub<i32>>
})");

  int expected = 10;
  r.verifyOutput(&expected);
}

TEST_P(ProcessorTest, Minimum) {
  if (std::get<1>(GetParam()) == FM32) {
    return; // Ring type is not large enough to hold value
  }

  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  r.addInput(10);

  r.run(R"(
func @main(%arg0: tensor<!pphlo.pub<i32>>) -> (tensor<!pphlo.pub<i32>>) {
  %0 = "pphlo.constant"() {value = dense<2147483647> : tensor<i32>} : () -> tensor<!pphlo.pub<i32>>
  %1 = "pphlo.minimum"(%0, %arg0) : (tensor<!pphlo.pub<i32>>, tensor<!pphlo.pub<i32>>) -> tensor<!pphlo.pub<i32>>
  return %1 :  tensor<!pphlo.pub<i32>>
})");

  int expected = 10;
  r.verifyOutput(&expected);
}

TEST_P(ProcessorTest, DynamicSlice1D) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  std::vector<int> op = {0, 1, 2, 3, 4};
  r.addInput(op);

  r.addInput(2);

  r.run(R"(
func @main(%arg0: tensor<5x!pphlo.pub<i32>>, %arg1: tensor<!pphlo.pub<i32>>) -> tensor<2x!pphlo.pub<i32>> {
  %0 = "pphlo.dynamic-slice"(%arg0, %arg1) {slice_sizes = dense<2> : tensor<i64>} : (tensor<5x!pphlo.pub<i32>>, tensor<!pphlo.pub<i32>>) -> tensor<2x!pphlo.pub<i32>>
  return %0 : tensor<2x!pphlo.pub<i32>>
})");

  std::vector<int> expected = {2, 3};
  r.verifyOutput(expected.data());
}

TEST_P(ProcessorTest, DynamicSlice2D) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  xt::xarray<float> op = {{0.0, 1.0, 2.0}, //
                          {3.0, 4.0, 5.0},
                          {6.0, 7.0, 8.0},
                          {9.0, 10.0, 11.0}};
  r.addInput(op);

  r.addInput(2);
  r.addInput(1);

  r.run(R"(
func @main(%arg0: tensor<4x3x!pphlo.pub<f32>>, %arg1: tensor<!pphlo.pub<i32>>, %arg2: tensor<!pphlo.pub<i32>>) -> tensor<2x2x!pphlo.pub<f32>> {
  %0 = "pphlo.dynamic-slice"(%arg0, %arg1, %arg2) {slice_sizes = dense<[2, 2]> : tensor<2xi64>} : (tensor<4x3x!pphlo.pub<f32>>, tensor<!pphlo.pub<i32>>, tensor<!pphlo.pub<i32>>) -> tensor<2x2x!pphlo.pub<f32>>
  return %0 : tensor<2x2x!pphlo.pub<f32>>
})");

  xt::xarray<float> expected = {{7.0, 8.0}, {10.0, 11.0}};
  r.verifyOutput(expected.data());
}

TEST_P(ProcessorTest, DynamicUpdateSlice1D) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  std::vector<int> op = {0, 1, 2, 3, 4};
  r.addInput(op);

  std::vector<int> u = {5, 6};
  r.addInput(u);

  r.addInput(2);

  r.run(R"(
func @main(%arg0: tensor<5x!pphlo.pub<i32>>, %arg1: tensor<2x!pphlo.pub<i32>>, %arg2: tensor<!pphlo.pub<i32>>) -> tensor<5x!pphlo.pub<i32>> {
  %0 = "pphlo.dynamic-update-slice"(%arg0, %arg1, %arg2) : (tensor<5x!pphlo.pub<i32>>, tensor<2x!pphlo.pub<i32>>, tensor<!pphlo.pub<i32>>) -> tensor<5x!pphlo.pub<i32>>
  return %0 : tensor<5x!pphlo.pub<i32>>
})");

  std::vector<int> expected = {0, 1, 5, 6, 4};
  r.verifyOutput(expected.data());
}

TEST_P(ProcessorTest, DynamicUpdateSlice2D) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  xt::xarray<float> op = {{0.0, 1.0, 2.0}, //
                          {3.0, 4.0, 5.0},
                          {6.0, 7.0, 8.0},
                          {9.0, 10.0, 11.0}};
  r.addInput(op);

  xt::xarray<float> u = {{12.0, 13.0}, //
                         {14.0, 15.0},
                         {16.0, 17.0}};
  r.addInput(u);

  r.addInput(1);
  r.addInput(1);

  r.run(R"(
func @main(%arg0: tensor<4x3x!pphlo.pub<f32>>, %arg1: tensor<3x2x!pphlo.pub<f32>>, %arg2: tensor<!pphlo.pub<i32>>, %arg3: tensor<!pphlo.pub<i32>>) -> tensor<4x3x!pphlo.pub<f32>> {
  %0 = "pphlo.dynamic-update-slice"(%arg0, %arg1, %arg2, %arg3) : (tensor<4x3x!pphlo.pub<f32>>, tensor<3x2x!pphlo.pub<f32>>, tensor<!pphlo.pub<i32>>, tensor<!pphlo.pub<i32>>) -> tensor<4x3x!pphlo.pub<f32>>
  return %0 : tensor<4x3x!pphlo.pub<f32>>
})");

  xt::xarray<float> expected = {{0.0, 1.0, 2.0}, //
                                {3.0, 12.0, 13.0},
                                {6.0, 14.0, 15.0},
                                {9.0, 16.0, 17.0}};
  r.verifyOutput(expected.data());
}

TEST_P(ProcessorTest, Sort1D) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  xt::xarray<float> op = {2.0, 1.0, 3.0, -10.0};
  r.addInput(op);

  r.run(R"(
func @main(%arg0: tensor<4x!pphlo.pub<f32>>) -> tensor<4x!pphlo.pub<f32>> {
    %0 = "pphlo.sort"(%arg0) ( {
    ^bb0(%arg1: tensor<!pphlo.pub<f32>>, %arg2: tensor<!pphlo.pub<f32>>):  // no predecessors
      %1 = "pphlo.less"(%arg1, %arg2) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<i1>>
      "pphlo.return"(%1) : (tensor<!pphlo.pub<i1>>) -> ()
    }) {dimension = 0 : i64, is_stable = true} : (tensor<4x!pphlo.pub<f32>>) -> (tensor<4x!pphlo.pub<f32>>)
    return %0 : tensor<4x!pphlo.pub<f32>>
})");
  xt::xarray<float> expected = {-10.0, 1.0, 2.0, 3.0};
  r.verifyOutput(expected.data());
}

TEST_P(ProcessorTest, Sort2DRow) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  xt::xarray<float> op = {{2.0, 1.0, 3.0, -10.0}, //
                          {4.0, 3.0, 2.0, 1.0}};

  r.addInput(op);

  // Row sort
  r.run(R"(
func @main(%arg0: tensor<2x4x!pphlo.pub<f32>>) -> tensor<2x4x!pphlo.pub<f32>> {
    %0 = "pphlo.sort"(%arg0) ( {
    ^bb0(%arg1: tensor<!pphlo.pub<f32>>, %arg2: tensor<!pphlo.pub<f32>>):  // no predecessors
      %1 = "pphlo.less"(%arg1, %arg2) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<i1>>
      "pphlo.return"(%1) : (tensor<!pphlo.pub<i1>>) -> ()
    }) {dimension = 1 : i64, is_stable = true} : (tensor<2x4x!pphlo.pub<f32>>) -> (tensor<2x4x!pphlo.pub<f32>>)
    return %0 : tensor<2x4x!pphlo.pub<f32>>
})");
  xt::xarray<float> expected = {{-10.0, 1.0, 2.0, 3.0}, //
                                {1.0, 2.0, 3.0, 4.0}};
  r.verifyOutput(expected.data());
}

TEST_P(ProcessorTest, Sort2DCol) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  xt::xarray<float> op = {{2.0, 1.0, 3.0, -10.0}, //
                          {4.0, 3.0, 2.0, 1.0}};

  r.addInput(op);

  // Column sort
  r.run(R"(
func @main(%arg0: tensor<2x4x!pphlo.pub<f32>>) -> tensor<2x4x!pphlo.pub<f32>> {
    %0 = "pphlo.sort"(%arg0) ( {
    ^bb0(%arg1: tensor<!pphlo.pub<f32>>, %arg2: tensor<!pphlo.pub<f32>>):  // no predecessors
      %1 = "pphlo.less"(%arg1, %arg2) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<i1>>
      "pphlo.return"(%1) : (tensor<!pphlo.pub<i1>>) -> ()
    }) {dimension = 0 : i64, is_stable = true} : (tensor<2x4x!pphlo.pub<f32>>) -> (tensor<2x4x!pphlo.pub<f32>>)
    return %0 : tensor<2x4x!pphlo.pub<f32>>
})");
  xt::xarray<float> expected = {{2.0, 1.0, 2.0, -10.0}, //
                                {4.0, 3.0, 3.0, 1.0}};
  r.verifyOutput(expected.data());
}

TEST_P(ProcessorTest, SortMultiOperands) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  r.addInput(xt::xarray<int>{3, 1});
  r.addInput(xt::xarray<int>{42, 50});
  r.addInput(xt::xarray<float>{-3.0, 1.5});

  // Row sort
  r.run(R"(
func @main(%arg0: tensor<2x!pphlo.pub<i32>>, %arg1: tensor<2x!pphlo.pub<i32>>, %arg2: tensor<2x!pphlo.pub<f32>>) -> (tensor<2x!pphlo.pub<i32>>, tensor<2x!pphlo.pub<i32>>, tensor<2x!pphlo.pub<f32>>) {
    %0:3 = "pphlo.sort"(%arg0, %arg1, %arg2) ( {
    ^bb0(%arg3: tensor<!pphlo.pub<i32>>, %arg4: tensor<!pphlo.pub<i32>>, %arg5: tensor<!pphlo.pub<i32>>, %arg6: tensor<!pphlo.pub<i32>>, %arg7: tensor<!pphlo.pub<f32>>, %arg8: tensor<!pphlo.pub<f32>>):  // no predecessors
      %1 = "pphlo.less"(%arg3, %arg4) : (tensor<!pphlo.pub<i32>>, tensor<!pphlo.pub<i32>>) -> tensor<!pphlo.pub<i1>>
      "pphlo.return"(%1) : (tensor<!pphlo.pub<i1>>) -> ()
    }) {dimension = 0 : i64, is_stable = true} : (tensor<2x!pphlo.pub<i32>>, tensor<2x!pphlo.pub<i32>>, tensor<2x!pphlo.pub<f32>>) -> (tensor<2x!pphlo.pub<i32>>, tensor<2x!pphlo.pub<i32>>, tensor<2x!pphlo.pub<f32>>)
    return %0#0, %0#1, %0#2 : tensor<2x!pphlo.pub<i32>>, tensor<2x!pphlo.pub<i32>>, tensor<2x!pphlo.pub<f32>>
})",
        3);
  xt::xarray<int> expected0 = {1, 3};
  r.verifyOutput(expected0.data(), 0);

  xt::xarray<int> expected1 = {50, 42};
  r.verifyOutput(expected1.data(), 1);

  xt::xarray<float> expected2 = {1.5, -3.0};
  r.verifyOutput(expected2.data(), 2);
}

TEST_P(ProcessorTest, SortComplicatedComparator) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  r.addInput(xt::xarray<int>{3, 1, 4, 2});
  r.addInput(xt::xarray<int>{42, 50, 49, 47});

  // Row sort
  r.run(R"(
func @main(%arg0: tensor<4x!pphlo.pub<i32>>, %arg1: tensor<4x!pphlo.pub<i32>>) -> (tensor<4x!pphlo.pub<i32>>, tensor<4x!pphlo.pub<i32>>) {
    %0:2 = "pphlo.sort"(%arg0, %arg1) ( {
    ^bb0(%arg2: tensor<!pphlo.pub<i32>>, %arg3: tensor<!pphlo.pub<i32>>, %arg4: tensor<!pphlo.pub<i32>>, %arg5: tensor<!pphlo.pub<i32>>):  // no predecessors
      %1 = "pphlo.less"(%arg2, %arg3) : (tensor<!pphlo.pub<i32>>, tensor<!pphlo.pub<i32>>) -> tensor<!pphlo.pub<i1>>
      %2 = "pphlo.less"(%arg4, %arg5) : (tensor<!pphlo.pub<i32>>, tensor<!pphlo.pub<i32>>) -> tensor<!pphlo.pub<i1>>
      %3 = "pphlo.and" (%1, %2) : (tensor<!pphlo.pub<i1>>, tensor<!pphlo.pub<i1>>) -> tensor<!pphlo.pub<i1>>
      "pphlo.return"(%3) : (tensor<!pphlo.pub<i1>>) -> ()
    }) {dimension = 0 : i64, is_stable = true} : (tensor<4x!pphlo.pub<i32>>, tensor<4x!pphlo.pub<i32>>) -> (tensor<4x!pphlo.pub<i32>>, tensor<4x!pphlo.pub<i32>>)
    return %0#0, %0#1 : tensor<4x!pphlo.pub<i32>>, tensor<4x!pphlo.pub<i32>>
})",
        2);
  xt::xarray<int> expected0 = {3, 1, 2, 4};
  r.verifyOutput(expected0.data(), 0);

  xt::xarray<int> expected1 = {42, 50, 47, 49};
  r.verifyOutput(expected1.data(), 1);
}

TEST_P(ProcessorTest, RemainderFxp) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));
  r.addInput(std::vector<float>{2.5, 18.5, 5.3});
  r.addInput(std::vector<float>{5.0, 4.2, 2.0});

  r.run(R"(
func @main(%arg0: tensor<3x!pphlo.pub<f32>>, %arg1: tensor<3x!pphlo.pub<f32>>) -> (tensor<3x!pphlo.pub<f32>>) {
  %0 = "pphlo.remainder"(%arg0, %arg1) : (tensor<3x!pphlo.pub<f32>>, tensor<3x!pphlo.pub<f32>>) -> tensor<3x!pphlo.pub<f32>> 
  return %0 : tensor<3x!pphlo.pub<f32>>
})");

  std::vector<float> expected{2.5, 1.7, 1.3};
  r.verifyOutput(expected.data());
}

TEST_P(ProcessorTest, RemainderInt) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));
  r.addInput(std::vector<int>{5, -5, 17, -17, 5, -5, 17, -17});
  r.addInput(std::vector<int>{2, 2, 3, 3, -2, -2, -3, -3});

  r.run(R"(
func @main(%arg0: tensor<8x!pphlo.pub<i32>>, %arg1: tensor<8x!pphlo.pub<i32>>) -> (tensor<8x!pphlo.pub<i32>>) {
  %0 = "pphlo.remainder"(%arg0, %arg1) : (tensor<8x!pphlo.pub<i32>>, tensor<8x!pphlo.pub<i32>>) -> tensor<8x!pphlo.pub<i32>>
  return %0 : tensor<8x!pphlo.pub<i32>>
})");

  std::vector<int> expected{1, -1, 2, -2, 1, -1, 2, -2};
  r.verifyOutput(expected.data());
}

TEST_P(ProcessorTest, ShiftLeftS32) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  r.addInput(std::vector<int32_t>{static_cast<int32_t>(0x12345678),
                                  static_cast<int32_t>(0xF0001000), 1, 3, 77, 1,
                                  -3, 77});
  r.addInput(std::vector<int32_t>{4, 8, 2, 7, 15, 32, 100, -1});

  r.run(R"(
func @main(%arg0: tensor<8x!pphlo.pub<i32>>, %arg1: tensor<8x!pphlo.pub<i32>>) -> (tensor<8x!pphlo.pub<i32>>) {
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

TEST_P(ProcessorTest, ShiftLeftU32) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  r.addInput(
      std::vector<uint32_t>{0x12345678, 0xF0001000, 1, 3, 77, 1, ~3U, 77});
  r.addInput(std::vector<uint32_t>{4, 8, 2, 7, 15, 32, 100, ~0U});

  r.run(R"(
func @main(%arg0: tensor<8x!pphlo.pub<ui32>>, %arg1: tensor<8x!pphlo.pub<ui32>>) -> (tensor<8x!pphlo.pub<ui32>>) {
  %0 = "pphlo.shift_left"(%arg0, %arg1) : (tensor<8x!pphlo.pub<ui32>>, tensor<8x!pphlo.pub<ui32>>) -> tensor<8x!pphlo.pub<ui32>>
  return %0 : tensor<8x!pphlo.pub<ui32>>
})");

  std::vector<uint32_t> expected{0x23456780, 0x00100000, 0x4, 0x180,
                                 2523136,    0,          0,   0};
  r.verifyOutput(expected.data());
}

TEST_P(ProcessorTest, ShiftRightLogicalS32) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  r.addInput(std::vector<int32_t>{static_cast<int32_t>(0x92345678),
                                  static_cast<int32_t>(0x10001000), 1, 3, 77, 1,
                                  -3, 77});
  r.addInput(std::vector<int32_t>{4, 8, 2, 7, 5, 32, /*100*/ 0, -1});

  r.run(R"(
func @main(%arg0: tensor<8x!pphlo.pub<i32>>, %arg1: tensor<8x!pphlo.pub<i32>>) -> (tensor<8x!pphlo.pub<i32>>) {
  %0 = "pphlo.shift_right_logical"(%arg0, %arg1) : (tensor<8x!pphlo.pub<i32>>, tensor<8x!pphlo.pub<i32>>) -> tensor<8x!pphlo.pub<i32>>
  return %0 : tensor<8x!pphlo.pub<i32>>
})");

  std::vector<int32_t> expected{
      static_cast<int>(0xF9234567), 0x00100010, 0, 0, 2, 0, -3, 0};
  r.verifyOutput(expected.data());
}

TEST_P(ProcessorTest, ShiftRightLogicalU32) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  r.addInput(
      std::vector<uint32_t>{0x92345678, 0x10001000, 1, 3, 77, 1, ~3U, 77});
  r.addInput(std::vector<uint32_t>{4, 8, 2, 7, 5, 32, /*100*/ 0, ~0U});

  r.run(R"(
func @main(%arg0: tensor<8x!pphlo.pub<ui32>>, %arg1: tensor<8x!pphlo.pub<ui32>>) -> (tensor<8x!pphlo.pub<ui32>>) {
  %0 = "pphlo.shift_right_logical"(%arg0, %arg1) : (tensor<8x!pphlo.pub<ui32>>, tensor<8x!pphlo.pub<ui32>>) -> tensor<8x!pphlo.pub<ui32>>
  return %0 : tensor<8x!pphlo.pub<ui32>>
})");

  std::vector<uint32_t> expected{0x09234567, 0x00100010, 0, 0, 2, 0, ~3U, 0};
  r.verifyOutput(expected.data());
}

TEST_P(ProcessorTest, ShiftRightArithmeticS32) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  r.addInput(std::vector<int32_t>{static_cast<int32_t>(0x92345678),
                                  static_cast<int32_t>(0x10001000), 1, 3, 77, 1,
                                  -3, 77});
  r.addInput(std::vector<int32_t>{4, 8, 2, 7, 2, 32, /*100*/ 0, -1});

  r.run(R"(
func @main(%arg0: tensor<8x!pphlo.pub<i32>>, %arg1: tensor<8x!pphlo.pub<i32>>) -> (tensor<8x!pphlo.pub<i32>>) {
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

TEST_P(ProcessorTest, ShiftRightArithmeticU32) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  r.addInput(
      std::vector<uint32_t>{0x92345678, 0x10001000, 1, 3, 77, 1, ~3U, 77});
  r.addInput(std::vector<uint32_t>{4, 8, 2, 7, 2, 32, /*100*/ 0, ~0U});

  r.run(R"(
func @main(%arg0: tensor<8x!pphlo.pub<ui32>>, %arg1: tensor<8x!pphlo.pub<ui32>>) -> (tensor<8x!pphlo.pub<ui32>>) {
  %0 = "pphlo.shift_right_arithmetic"(%arg0, %arg1) : (tensor<8x!pphlo.pub<ui32>>, tensor<8x!pphlo.pub<ui32>>) -> tensor<8x!pphlo.pub<ui32>>
  return %0 : tensor<8x!pphlo.pub<ui32>>
})");

  std::vector<uint32_t> expected{0x09234567, 0x00100010, 0, 0, 19, 0, ~3U, 0};
  r.verifyOutput(expected.data());
}

TEST_P(ProcessorTest, SelectAndScatter1) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  r.addInput(xt::xarray<int32_t>{
      {7, 2, 5, 3, 10, 2}, //
      {3, 8, 9, 3, 4, 2},  //
      {1, 5, 7, 5, 6, 1},  //
      {0, 6, 2, 7, 2, 8}   //
  });
  r.addInput(xt::xarray<int32_t>{
      {2, 6}, //
      {3, 1}  //
  });
  r.addInput(static_cast<int32_t>(0));

  r.run(R"(
func @main(%arg0: tensor<4x6x!pphlo.pub<i32>>, %arg1: tensor<2x2x!pphlo.pub<i32>>, %arg2: tensor<!pphlo.pub<i32>>) -> (tensor<4x6x!pphlo.pub<i32>>) {
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

  xt::xarray<int32_t> expected = {{0, 0, 0, 0, 6, 0}, //
                                  {0, 0, 2, 0, 0, 0}, //
                                  {0, 0, 3, 0, 0, 0}, //
                                  {0, 0, 0, 0, 0, 1}};
  r.verifyOutput(expected.data());
}

TEST_P(ProcessorTest, SelectAndScatter2) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  r.addInput(xt::xarray<int32_t>{
      {7, 2, 5, 3, 8}, //
      {3, 8, 9, 3, 4}, //
      {1, 5, 7, 5, 6}, //
      {0, 6, 2, 10, 2} //
  });
  r.addInput(xt::xarray<int32_t>{
      {2, 6}, //
      {3, 1}  //
  });
  r.addInput(static_cast<int32_t>(0));

  r.run(R"(
func @main(%arg0: tensor<4x5x!pphlo.pub<i32>>, %arg1: tensor<2x2x!pphlo.pub<i32>>, %arg2: tensor<!pphlo.pub<i32>>) -> (tensor<4x5x!pphlo.pub<i32>>) {
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

  xt::xarray<int32_t> expected = {{0, 0, 0, 0, 0}, //
                                  {0, 0, 8, 0, 0}, //
                                  {0, 0, 3, 0, 0}, //
                                  {0, 0, 0, 1, 0}};
  r.verifyOutput(expected.data());
}

INSTANTIATE_TEST_SUITE_P(
    ProcessorTestInstances, ProcessorTest,
    testing::Combine(testing::Values(4, 3, 2),
                     testing::Values(FieldType::FM64, FieldType::FM128),
                     testing::Values(ProtocolKind::REF2K,
                                     ProtocolKind::SEMI2K)),
    [](const testing::TestParamInfo<ProcessorTest::ParamType> &p) {
      return fmt::format("{}x{}x{}", std::get<0>(p.param),
                         std::get<1>(p.param), std::get<2>(p.param));
    });

// NOTE(junfeng): ABY3 is 3pc only.
INSTANTIATE_TEST_SUITE_P(
    ProcessorTestABY3Instances, ProcessorTest,
    testing::Combine(testing::Values(3),
                     testing::Values(FieldType::FM64, FieldType::FM128),
                     testing::Values(ProtocolKind::ABY3)),
    [](const testing::TestParamInfo<ProcessorTest::ParamType> &p) {
      return fmt::format("{}x{}x{}", std::get<0>(p.param),
                         std::get<1>(p.param), std::get<2>(p.param));
    });

} // namespace spu::device
