// Copyright 2023 Ant Group Co., Ltd.
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

#include "libspu/kernel/hlo/select_and_scatter.h"

#include "gtest/gtest.h"
#include "xtensor/xio.hpp"

#include "libspu/kernel/hal/polymorphic.h"
#include "libspu/kernel/hal/type_cast.h"
#include "libspu/kernel/test_util.h"

namespace spu::kernel::hlo {

struct SelectAndScatterTestParam {
  xt::xarray<float> operand;
  xt::xarray<float> source;
  std::vector<std::pair<int64_t, int64_t>> window_padding;
  Shape window_shape;
  Strides window_strides;
  xt::xarray<float> expected;
};

class SelectAndScatterTest
    : public ::testing::TestWithParam<SelectAndScatterTestParam> {
 public:
  SelectAndScatterTest() : ctx_(test::makeSPUContext()) {}

  SPUContext ctx_;
};

TEST_P(SelectAndScatterTest, ParamTest) {
  xt::xarray<float> operand = GetParam().operand;
  xt::xarray<float> source = GetParam().source;
  xt::xarray<float> expected = GetParam().expected;
  xt::xarray<float> init = 0;

  Value operand_s = test::makeValue(&ctx_, operand, VIS_SECRET);
  Value source_s = test::makeValue(&ctx_, source, VIS_SECRET);
  Value init_val = test::makeValue(&ctx_, init, VIS_SECRET);

  const auto ret = SelectAndScatter(
      &ctx_, operand_s, source_s, init_val, GetParam().window_shape,
      GetParam().window_strides, GetParam().window_padding,
      [&](const spu::Value &lhs, const spu::Value &rhs) {
        return hal::greater(&ctx_, lhs, rhs);
      },
      [&](const spu::Value &lhs, const spu::Value &rhs) {
        return hal::add(&ctx_, lhs, rhs);
      });
  auto ret_hat = hal::dump_public_as<float>(&ctx_, hal::reveal(&ctx_, ret));
  EXPECT_TRUE(xt::allclose(expected, ret_hat, 0.01, 0.001));
}

INSTANTIATE_TEST_CASE_P(
    SelectAndScatterTest_Instantiation, SelectAndScatterTest,
    ::testing::Values(
        SelectAndScatterTestParam{{1, 9, 3, 7, 5, 6},
                                  {34, 42},
                                  {{0, 0}},
                                  {3},
                                  {3},
                                  {0, 34, 0, 42, 0, 0}},
        SelectAndScatterTestParam{{{7, 2, 5, 3, 10, 2},
                                   {3, 8, 9, 3, 4, 2},
                                   {1, 5, 7, 5, 6, 1},
                                   {0, 6, 2, 7, 2, 8}},
                                  {{2, 6}, {3, 1}},
                                  {{0, 0}, {0, 0}},
                                  {2, 3},
                                  {2, 3},
                                  {{0, 0, 0, 0, 6, 0},
                                   {0, 0, 2, 0, 0, 0},
                                   {0, 0, 3, 0, 0, 0},
                                   {0, 0, 0, 0, 0, 1}}},
        SelectAndScatterTestParam{{1, 9, 3, 7, 5, 6},
                                  {34, 42, 53, 19},
                                  {{0, 0}},
                                  {3},
                                  {1},
                                  {0, 76, 0, 72, 0, 0}},
        SelectAndScatterTestParam{{{7, 2, 5, 3, 10, 2}, {3, 8, 9, 3, 4, 2}},
                                  {{2, 6}},
                                  {{0, 0}, {0, 0}},
                                  {2, 3},
                                  {2, 3},
                                  {{0, 0, 0, 0, 6, 0}, {0, 0, 2, 0, 0, 0}}},
        SelectAndScatterTestParam{{{7, 2, 5, 3, 8}, {3, 8, 9, 3, 4}},
                                  {{2, 6, 4}},
                                  {{0, 0}, {0, 0}},
                                  {2, 3},
                                  {1, 1},
                                  {{0, 0, 0, 0, 0}, {0, 0, 12, 0, 0}}},
        SelectAndScatterTestParam{
            {{1.5, 2.5, 1.5}, {3.5, 1.5, 3.5}, {4.5, 2.5, 4.5}},
            {{1.0, 2.0}, {3.0, 4.0}},
            {{0, 0}, {0, 0}},
            {2, 2},
            {1, 1},
            {{0.0, 0.0, 0.0}, {1.0, 0.0, 2.0}, {3.0, 0.0, 4.0}}}));

}  // namespace spu::kernel::hlo
