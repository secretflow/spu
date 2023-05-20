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

#include "libspu/core/context.h"
#include "libspu/core/ndarray_ref.h"
#include "libspu/core/type.h"
#include "libspu/core/value.h"
#include "libspu/kernel/hlo/const.h"
#include "libspu/kernel/test_util.h"

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

TEST(IndexingTest, DynamicUpdateSliceScalarWithPublicIndices) {
  SPUContext sctx = test::makeSPUContext();
  auto input = Constant(&sctx, static_cast<int64_t>(1), {5});
  auto update = Constant(&sctx, static_cast<int64_t>(2), {1});
  std::vector<spu::Value> start_indices{
      Constant(&sctx, static_cast<int64_t>(1), {})};

  auto output = DynamicUpdateSlice(&sctx, input, update, start_indices);

  EXPECT_EQ(output.numel(), input.numel());

  auto p_ret = hal::dump_public_as<int64_t>(&sctx, output);
  xt::xarray<int64_t> expected{1, 2, 1, 1, 1};
  EXPECT_EQ(p_ret, expected);
}

TEST(IndexingTest, DynamicUpdateSliceScalarWithSecretIndices) {
  SPUContext sctx = test::makeSPUContext();
  auto input = Constant(&sctx, static_cast<int64_t>(1), {5});
  auto update = Constant(&sctx, static_cast<int64_t>(2), {1});
  std::vector<spu::Value> start_indices{
      Seal(&sctx, Constant(&sctx, static_cast<int64_t>(1), {}))};

  auto output = DynamicUpdateSlice(&sctx, input, update, start_indices);

  EXPECT_EQ(output.numel(), input.numel());

  auto p_ret = hal::dump_public_as<int64_t>(&sctx, Reveal(&sctx, output));
  xt::xarray<int64_t> expected{1, 2, 1, 1, 1};
  EXPECT_EQ(p_ret, expected);
}

TEST(IndexingTest, DynamicUpdateSliceMultiValuesWithPublicIndices) {
  SPUContext sctx = test::makeSPUContext();
  auto input = Constant(&sctx, static_cast<int64_t>(1), {5});
  auto update = Constant(&sctx, static_cast<int64_t>(2), {2});
  std::vector<spu::Value> start_indices{
      Constant(&sctx, static_cast<int64_t>(3), {})};

  auto output = DynamicUpdateSlice(&sctx, input, update, start_indices);

  EXPECT_EQ(output.numel(), input.numel());

  auto p_ret = hal::dump_public_as<int64_t>(&sctx, output);
  xt::xarray<int64_t> expected{1, 1, 1, 2, 2};
  EXPECT_EQ(p_ret, expected);
}

TEST(IndexingTest, DynamicUpdateSliceMultiValuesWithSecretIndices) {
  SPUContext sctx = test::makeSPUContext();
  auto input = Constant(&sctx, static_cast<int64_t>(1), {5});
  auto update = Constant(&sctx, static_cast<int64_t>(2), {2});
  std::vector<spu::Value> start_indices{
      Seal(&sctx, Constant(&sctx, static_cast<int64_t>(3), {}))};

  auto output = DynamicUpdateSlice(&sctx, input, update, start_indices);

  EXPECT_EQ(output.numel(), input.numel());

  auto p_ret = hal::dump_public_as<int64_t>(&sctx, Reveal(&sctx, output));
  xt::xarray<int64_t> expected{1, 1, 1, 2, 2};
  EXPECT_EQ(p_ret, expected);
}

TEST(DynamicSliceTest, DynamicSliceWithPublicIndices) {
  SPUContext sctx = test::makeSPUContext();
  xt::xarray<float> x = {{0.05, 0.24, 0.5}, {2, 5, 50}};
  auto input = test::makeValue(&sctx, x, VIS_SECRET);

  auto start_indices =
      std::vector<spu::Value>{Constant(&sctx, static_cast<int64_t>(1), {}),
                              Constant(&sctx, static_cast<int64_t>(1), {})};

  auto output = DynamicSlice(&sctx, input, {2, 2}, start_indices);

  auto p_ret = hal::dump_public_as<float>(&sctx, Reveal(&sctx, output));
  xt::xarray<float> expected{{0.24, 0.5}, {5, 50}};
  EXPECT_TRUE(xt::allclose(p_ret, expected, 0.01, 0.001))
      << p_ret << std::endl
      << expected << std::endl;
}

TEST(DynamicSliceTest, DynamicSliceWithSecretIndices) {
  SPUContext sctx = test::makeSPUContext();
  xt::xarray<float> x = {{0.05, 0.24, 0.5}, {2, 5, 50}, {7, 9, 10.1}};
  auto input = test::makeValue(&sctx, x, VIS_SECRET);

  auto start_indices = std::vector<spu::Value>{
      Seal(&sctx, Constant(&sctx, static_cast<int64_t>(0), {})),
      Seal(&sctx, Constant(&sctx, static_cast<int64_t>(1), {}))};

  auto output = DynamicSlice(&sctx, input, {2, 2}, start_indices);

  auto p_ret = hal::dump_public_as<float>(&sctx, Reveal(&sctx, output));
  xt::xarray<float> expected{{0.24, 0.5}, {5, 50}};
  EXPECT_TRUE(xt::allclose(p_ret, expected, 0.01, 0.001))
      << p_ret << std::endl
      << expected << std::endl;
}

TEST(DynamicSliceTest, DynamicSliceWithSecretIndicesFull) {
  SPUContext sctx = test::makeSPUContext();
  xt::xarray<float> x = {{0.05, 0.24, 0.5}, {2, 5, 50}, {7, 9, 10.1}};
  auto input = test::makeValue(&sctx, x, VIS_SECRET);

  auto start_indices = std::vector<spu::Value>{
      Seal(&sctx, Constant(&sctx, static_cast<int64_t>(0), {})),
      Seal(&sctx, Constant(&sctx, static_cast<int64_t>(0), {}))};

  auto output = DynamicSlice(&sctx, input, {2, 3}, start_indices);

  auto p_ret = hal::dump_public_as<float>(&sctx, Reveal(&sctx, output));
  xt::xarray<float> expected{{0.05, 0.24, 0.5}, {2, 5, 50}};
  EXPECT_TRUE(xt::allclose(p_ret, expected, 0.01, 0.001))
      << p_ret << std::endl
      << expected << std::endl;
}

TEST(DynamicSliceTest, DynamicSliceWithPublicIndicesOffRangeLow) {
  SPUContext sctx = test::makeSPUContext();
  xt::xarray<float> x = {{0.05, 0.24, 0.5}, {2, 5, 50}};
  auto input = test::makeValue(&sctx, x, VIS_SECRET);

  auto start_indices =
      std::vector<spu::Value>{Constant(&sctx, static_cast<int64_t>(-1), {}),
                              Constant(&sctx, static_cast<int64_t>(-1), {})};

  auto output = DynamicSlice(&sctx, input, {2, 2}, start_indices);

  auto p_ret = hal::dump_public_as<float>(&sctx, Reveal(&sctx, output));
  xt::xarray<float> expected{{0.05, 0.24}, {2, 5}};
  EXPECT_TRUE(xt::allclose(p_ret, expected, 0.01, 0.001))
      << p_ret << std::endl
      << expected << std::endl;
}

TEST(DynamicSliceTest, DynamicSliceWithPublicIndicesOffRangeHigh) {
  SPUContext sctx = test::makeSPUContext();
  xt::xarray<float> x = {{0.05, 0.24, 0.5}, {2, 5, 50}};
  auto input = test::makeValue(&sctx, x, VIS_SECRET);

  auto start_indices =
      std::vector<spu::Value>{Constant(&sctx, static_cast<int64_t>(10), {}),
                              Constant(&sctx, static_cast<int64_t>(10), {})};

  auto output = DynamicSlice(&sctx, input, {2, 2}, start_indices);

  auto p_ret = hal::dump_public_as<float>(&sctx, Reveal(&sctx, output));
  xt::xarray<float> expected{{0.24, 0.5}, {5, 50}};
  EXPECT_TRUE(xt::allclose(p_ret, expected, 0.01, 0.001))
      << p_ret << std::endl
      << expected << std::endl;
}

TEST(DynamicSliceTest, DynamicSliceWithSecretIndicesOffRangeLow) {
  SPUContext sctx = test::makeSPUContext();
  xt::xarray<float> x = {{0.05, 0.24, 0.5}, {2, 5, 50}};
  auto input = test::makeValue(&sctx, x, VIS_SECRET);

  auto start_indices = std::vector<spu::Value>{
      Seal(&sctx, Constant(&sctx, static_cast<int64_t>(-1), {})),
      Seal(&sctx, Constant(&sctx, static_cast<int64_t>(-1), {}))};

  auto output = DynamicSlice(&sctx, input, {2, 2}, start_indices);

  auto p_ret = hal::dump_public_as<float>(&sctx, Reveal(&sctx, output));
  xt::xarray<float> expected{{0.05, 0.24}, {2, 5}};
  EXPECT_TRUE(xt::allclose(p_ret, expected, 0.01, 0.001))
      << p_ret << std::endl
      << expected << std::endl;
}

TEST(DynamicSliceTest, DynamicSliceWithSecretIndicesOffRangeHigh) {
  SPUContext sctx = test::makeSPUContext();
  xt::xarray<float> x = {{0.05, 0.24, 0.5}, {2, 5, 50}};
  auto input = test::makeValue(&sctx, x, VIS_SECRET);

  auto start_indices = std::vector<spu::Value>{
      Seal(&sctx, Constant(&sctx, static_cast<int64_t>(10), {})),
      Seal(&sctx, Constant(&sctx, static_cast<int64_t>(10), {}))};

  auto output = DynamicSlice(&sctx, input, {2, 2}, start_indices);

  auto p_ret = hal::dump_public_as<float>(&sctx, Reveal(&sctx, output));
  xt::xarray<float> expected{{0.24, 0.5}, {5, 50}};
  EXPECT_TRUE(xt::allclose(p_ret, expected, 0.01, 0.001))
      << p_ret << std::endl
      << expected << std::endl;
}

}  // namespace spu::kernel::hlo
