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
    : public ::testing::TestWithParam<std::tuple<FieldType, Shape3D>> {};

INSTANTIATE_TEST_SUITE_P(
    Cheetah, CheetahDotTest,
    testing::Combine(testing::Values(FieldType::FM32, FieldType::FM64,
                                     FieldType::FM128),
                     testing::Values(Shape3D{8, 7, 5}, Shape3D{57, 30, 1},
                                     Shape3D{30, 57, 1}, Shape3D{18, 8, 41},
                                     Shape3D{25088, 5, 25})),
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

  std::vector<ArrayRef> mat(kWorldSize);
  // NOTE(juhou): now Cheetah supports strided ArrayRef
  for (size_t stride : {1, 3}) {
    mat[0] = ring_rand(field, stride * dim3[0] * dim3[1]);
    mat[0] = mat[0].slice(0, mat[0].numel(), stride);

    mat[1] = ring_rand(field, 1 + stride * dim3[1] * dim3[2]);
    mat[1] = mat[1].slice(1, mat[1].numel(), stride);

    std::vector<ArrayRef> result(kWorldSize);
    utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> lctx) {
      int rank = lctx->Rank();
      auto dot = std::make_shared<CheetahDot>(lctx);
      result[rank] = dot->DotOLE(mat[rank], dim3, rank == 0);
    });

    auto expected = ring_mmul(mat[0], mat[1], dim3[0], dim3[2], dim3[1]);
    auto computed = ring_add(result[0], result[1]);
    EXPECT_EQ(expected.numel(), computed.numel());

    const int64_t kMaxDiff = 1;
    DISPATCH_ALL_FIELDS(field, "_", [&]() {
      auto e = ArrayView<ring2k_t>(expected);
      auto c = ArrayView<ring2k_t>(computed);

      for (auto idx = 0; idx < expected.numel(); idx++) {
        EXPECT_NEAR(e[idx], c[idx], kMaxDiff);
      }
    });
  }
}

}  // namespace spu::mpc::cheetah
