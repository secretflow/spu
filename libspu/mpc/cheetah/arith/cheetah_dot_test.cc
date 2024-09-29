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

#include "libspu/mpc/cheetah/arith/cheetah_dot.h"

#include "gtest/gtest.h"

#include "libspu/core/type_util.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::mpc::cheetah {

class CheetahDotTest
    : public ::testing::TestWithParam<std::tuple<size_t, Shape3D>> {};

INSTANTIATE_TEST_SUITE_P(
    Cheetah, CheetahDotTest,
    testing::Combine(testing::Values(64, 128),
                     testing::Values(Shape3D{8, 7, 5}, Shape3D{57, 30, 1},
                                     Shape3D{30, 57, 1}, Shape3D{18, 8, 41},
                                     Shape3D{500, 13, 25},
                                     Shape3D{1, 2048, 768},
                                     Shape3D{18, 768, 78})),
    [](const testing::TestParamInfo<CheetahDotTest::ParamType>& p) {
      return fmt::format("{}x{}x{}x{}", std::get<0>(std::get<1>(p.param)),
                         std::get<1>(std::get<1>(p.param)),
                         std::get<2>(std::get<1>(p.param)),
                         std::get<0>(p.param));
    });

TEST_P(CheetahDotTest, Basic) {
  size_t kWorldSize = 2;
  auto field = std::get<0>(GetParam());
  auto dim3 = std::get<1>(GetParam());

  std::vector<MemRef> mat(kWorldSize);
  // NOTE(juhou): now Cheetah supports strided ArrayRef
  for (int64_t stride : {1, 3}) {
    MemRef tmp_0(makeType<RingTy>(SE_INVALID, field),
                 {stride * dim3[0] * dim3[1]});
    ring_rand(tmp_0);
    mat[0] = std::move(tmp_0);
    mat[0] = mat[0].slice({0}, {dim3[0] * dim3[1]}, {stride});
    mat[0] = mat[0].reshape({dim3[0], dim3[1]});

    MemRef tmp_1(makeType<RingTy>(SE_INVALID, field),
                 {1 + stride * dim3[1] * dim3[2]});
    ring_rand(tmp_1);
    mat[1] = std::move(tmp_1);
    mat[1] = mat[1].slice({1}, {dim3[1] * dim3[2]}, {stride});
    mat[1] = mat[1].reshape({dim3[1], dim3[2]});

    std::vector<MemRef> result(kWorldSize);
    utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> lctx) {
      int rank = lctx->Rank();
      auto dot = std::make_shared<CheetahDot>(lctx);
      result[rank] = dot->DotOLE(mat[rank], dim3, rank == 0);
    });

    auto expected = ring_mmul(mat[0], mat[1]);
    auto computed = ring_add(result[0], result[1]);
    EXPECT_EQ(expected.numel(), computed.numel());

    const int64_t kMaxDiff = 1;
    DISPATCH_ALL_STORAGE_TYPES(GetStorageType(field), [&]() {
      auto e = MemRefView<ScalarT>(expected);
      auto c = MemRefView<ScalarT>(computed);

      for (auto idx = 0; idx < expected.numel(); idx++) {
        EXPECT_NEAR(e[idx], c[idx], kMaxDiff);
      }
    });
  }
}

TEST_P(CheetahDotTest, BatchDot) {
  size_t kWorldSize = 2;
  auto field = std::get<0>(GetParam());
  auto dim3 = std::get<1>(GetParam());
  int64_t B = 12;
  Shape4D dim4 = {B, dim3[0], dim3[1], dim3[2]};

  std::vector<MemRef> mat(kWorldSize);
  MemRef tmp_0(makeType<RingTy>(SE_INVALID, field), {B, dim3[0], dim3[1]});
  ring_rand(tmp_0);
  mat[0] = std::move(tmp_0);
  MemRef tmp_1(makeType<RingTy>(SE_INVALID, field), {B, dim3[1], dim3[2]});
  ring_rand(tmp_1);
  mat[1] = std::move(tmp_1);

  std::vector<MemRef> result(kWorldSize);
  utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> lctx) {
    int rank = lctx->Rank();
    auto dot = std::make_shared<CheetahDot>(lctx);
    result[rank] = dot->BatchDotOLE(mat[rank], lctx.get(), dim4, rank == 0);
  });

  MemRef expected(makeType<RingTy>(SE_INVALID, field), {B * dim3[0] * dim3[2]});
  ring_zeros(expected);

  mat[0] = mat[0].reshape({B * dim3[0] * dim3[1]});
  mat[1] = mat[1].reshape({B * dim3[1] * dim3[2]});
  for (int64_t b = 0; b < B; ++b) {
    auto lhs = mat[0].slice({b * dim3[0] * dim3[1]}, {dim3[0] * dim3[1]}, {1});
    auto rhs = mat[1].slice({b * dim3[1] * dim3[2]}, {dim3[1] * dim3[2]}, {1});
    auto slice =
        expected.slice({b * dim3[0] * dim3[2]}, {dim3[0] * dim3[2]}, {1})
            .reshape({dim3[0], dim3[2]});

    ring_mmul_(slice, lhs.reshape({dim3[0], dim3[1]}),
               rhs.reshape({dim3[1], dim3[2]}));
  }
  auto computed = ring_add(result[0], result[1]);
  EXPECT_EQ(expected.numel(), computed.numel());

  [[maybe_unused]] constexpr int64_t kMaxDiff = 1;
  int64_t max_diff = 0;
  DISPATCH_ALL_STORAGE_TYPES(GetStorageType(field), [&]() {
    auto e = MemRefView<ScalarT>(expected);
    auto c = MemRefView<ScalarT>(computed);

    for (auto idx = 0; idx < expected.numel(); idx++) {
      if (e[idx] != c[idx]) {
        std::cout << fmt::format("expected {} got {} at {}\n", e[idx], c[idx],
                                 idx);
      }
      int64_t diff = e[idx] < c[idx] ? c[idx] - e[idx] : e[idx] - c[idx];
      max_diff = std::max(max_diff, diff);
      ASSERT_LE(diff, kMaxDiff);
    }
  });
}

}  // namespace spu::mpc::cheetah
