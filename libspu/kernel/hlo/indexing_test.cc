// Copyright 2022 Ant Group Co., Ltd.
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

#include "libspu/kernel/hlo/indexing.h"

#include "gtest/gtest.h"

#include "libspu/core/ndarray_ref.h"
#include "libspu/core/type.h"
#include "libspu/kernel/context.h"
#include "libspu/kernel/hal/test_util.h"
#include "libspu/kernel/hlo/const.h"
#include "libspu/kernel/value.h"

namespace spu::kernel::hlo {

TEST(IndexingTest, Take1) {
  Value a(NdArrayRef(makePtType(PT_I64), {5}), DT_I64);

  auto *data_ptr = static_cast<int64_t *>(a.data().data());
  std::iota(data_ptr, data_ptr + 5, 0);

  for (int64_t idx = 0; idx < 5; ++idx) {
    auto v = a.data().at<int64_t>({idx});
    EXPECT_EQ(v, idx);
  }

  auto r = FilterByMask(nullptr, a, {0U, 1U, 1U, 0U, 0U});

  EXPECT_EQ(r.shape()[0], 2);
  EXPECT_EQ(r.data().at<int64_t>({0}), 1);
  EXPECT_EQ(r.data().at<int64_t>({1}), 2);
}

TEST(IndexingTest, Take2) {
  Value a(NdArrayRef(makePtType(PT_I64), {10}), DT_I64);

  auto *data_ptr = static_cast<int64_t *>(a.data().data());
  std::iota(data_ptr, data_ptr + 10, 0);

  // Create a strided data
  Value b(NdArrayRef(a.data().buf(), a.data().eltype(), {5}, {2}, 0),
          a.dtype());
  // Sanity input...
  for (int64_t idx = 0; idx < 5; ++idx) {
    auto v = b.data().at<int64_t>({idx});
    EXPECT_EQ(v, 2 * idx);
  }

  auto r = FilterByMask(nullptr, b, {0U, 1U, 1U, 0U, 0U});

  EXPECT_EQ(r.shape()[0], 2);
  EXPECT_EQ(r.data().at<int64_t>({0}), 2);
  EXPECT_EQ(r.data().at<int64_t>({1}), 4);
}

TEST(DynamicUpdateSliceTest, UpdateSliceScalar) {
  HalContext hctx = hal::test::makeRefHalContext();
  auto input = Constant(&hctx, static_cast<int64_t>(1), {5});
  auto update = Constant(&hctx, static_cast<int64_t>(2), {1});

  auto output = UpdateSlice(&hctx, input, update, {1});

  EXPECT_EQ(output.numel(), input.numel());

  auto p_ret = hal::dump_public_as<int64_t>(&hctx, output);
  xt::xarray<int64_t> expected{1, 2, 1, 1, 1};
  EXPECT_EQ(p_ret, expected);
}

TEST(DynamicUpdateSliceTest, UpdateSliceMultiValues) {
  HalContext hctx = hal::test::makeRefHalContext();
  auto input = Constant(&hctx, static_cast<int64_t>(1), {5});
  auto update = Constant(&hctx, static_cast<int64_t>(2), {2});

  auto output = UpdateSlice(&hctx, input, update, {3});

  EXPECT_EQ(output.numel(), input.numel());

  auto p_ret = hal::dump_public_as<int64_t>(&hctx, output);
  xt::xarray<int64_t> expected{1, 1, 1, 2, 2};
  EXPECT_EQ(p_ret, expected);
}

}  // namespace spu::kernel::hlo
