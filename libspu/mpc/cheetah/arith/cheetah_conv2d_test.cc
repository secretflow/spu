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

#include "gtest/gtest.h"

#include "libspu/core/type_util.h"
#include "libspu/core/xt_helper.h"
#include "libspu/mpc/cheetah/arith/cheetah_dot.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::mpc::cheetah::test {

class CheetahConv2dTest
    : public ::testing::TestWithParam<
          std::tuple<FieldType, int64_t, Shape3D, Shape3D, Shape2D>> {};

std::string to_string(const Shape3D& dim3) {
  return fmt::format("H{}W{}C{}", dim3[0], dim3[1], dim3[2]);
}

INSTANTIATE_TEST_SUITE_P(
    Cheetah, CheetahConv2dTest,
    testing::Combine(testing::Values(FieldType::FM64),
                     // input batch N
                     testing::Values(1LL, 2LL, 7LL),
                     // input_shape HxWxC
                     testing::Values(Shape3D{8, 8, 4}, Shape3D{38, 38, 3}),
                     // kernel_shape hxwxO
                     testing::Values(Shape3D{1, 1, 3}, Shape3D{2, 2, 1},
                                     Shape3D{27, 27, 5}),
                     // window_strides
                     testing::Values(Shape2D{2, 2}, Shape2D{1, 1})),
    [](const testing::TestParamInfo<CheetahConv2dTest::ParamType>& p) {
      return fmt::format("{}N{}{}h{}O{}s{}", std::get<0>(p.param),
                         std::get<1>(p.param), to_string(std::get<2>(p.param)),
                         std::get<3>(p.param)[0], std::get<3>(p.param)[2],
                         std::get<4>(p.param)[0]);
    });

TEST_P(CheetahConv2dTest, Basic) {
  constexpr size_t kWorldSize = 2;
  auto field = std::get<0>(GetParam());

  int64_t N = std::get<1>(GetParam());
  auto tshape = std::get<2>(GetParam());
  auto kshape = std::get<3>(GetParam());
  tshape[0] = std::max(tshape[0], kshape[0]);
  tshape[1] = std::max(tshape[1], kshape[1]);
  int64_t O = kshape[2];
  kshape[2] = tshape[2];
  auto window_strides = std::get<4>(GetParam());

  ArrayRef tensor = ring_rand(field, calcNumel(tshape) * N);
  ArrayRef kernel = ring_rand(field, calcNumel(kshape) * O);

  std::vector<ArrayRef> result(kWorldSize);
  utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> lctx) {
    int rank = lctx->Rank();
    auto dot = std::make_shared<CheetahDot>(lctx);
    if (rank == 0) {
      result[rank] =
          dot->Conv2dOLE(tensor, N, tshape, O, kshape, window_strides, true);
    } else {
      result[rank] =
          dot->Conv2dOLE(kernel, N, tshape, O, kshape, window_strides, false);
    }
  });

  ArrayRef expected =
      ring_conv2d(tensor, kernel, N, tshape, O, kshape, window_strides);
  ArrayRef computed = ring_add(result[0], result[1]);

  const int64_t kMaxDiff = 1;
  DISPATCH_ALL_FIELDS(field, "_", [&]() {
    auto e = ArrayView<ring2k_t>(expected);
    auto c = ArrayView<ring2k_t>(computed);

    for (auto idx = 0; idx < expected.numel(); idx++) {
      EXPECT_NEAR(c[idx], e[idx], kMaxDiff);
    }
  });
}

}  // namespace spu::mpc::cheetah::test
