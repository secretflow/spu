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

#include "libspu/mpc/utils/gfmp_ops.h"
#include "libspu/mpc/utils/ring_ops.h"

#include <random>

#include "gtest/gtest.h"

namespace spu::mpc {

class GfmpArrayRefTest
    : public ::testing::TestWithParam<
            std::tuple<FieldType,
                        int64_t,
                        int64_t>> {};

static NdArrayRef makeRandomArray(FieldType field, int64_t numel,
                                    int64_t stride) {
    const Type ty = makeType<GfmpTy>(field);
    const size_t buf_size = SizeOf(field) * numel * stride;

    auto buf = std::make_shared<yacl::Buffer>(buf_size);
    {
        size_t numOfInts = buf_size / sizeof(int32_t);
        auto* begin = buf->data<int32_t>();
        for(size_t idx = 0; idx < numOfInts; idx++) {
            *(begin + idx) = std::rand();
        }
    }
    return gfmp_mod(NdArrayRef(buf, ty, {numel}, {stride}, 0));
}

INSTANTIATE_TEST_SUITE_P(
    GfmpArrayRefTestSuite, GfmpArrayRefTest,
    testing::Combine(testing::Values(FM32, FM64, FM128),  //
                     testing::Values(1, 3, 10000),         // size of parameters
                     testing::Values(1, 3)  // stride of first param
                     ),
    [](const testing::TestParamInfo<GfmpArrayRefTest::ParamType>& p) {
      return fmt::format("{}x{}x{}", std::get<0>(p.param),
                         std::get<1>(p.param), std::get<2>(p.param));
    });

TEST_P(GfmpArrayRefTest, BatchInverse) {
  const FieldType field = std::get<0>(GetParam());
  const int64_t numel = std::get<1>(GetParam());
  const int64_t stride = std::get<2>(GetParam());
  const Type ty = makeType<RingTy>(field);

  // GIVEN
  const NdArrayRef y = makeRandomArray(field, numel, stride);
  DISPATCH_ALL_FIELDS(field, [&]() {
    NdArrayView<ring2k_t> _y(y);
    pforeach(0, numel, [&](int64_t idx) {_y[idx]=idx+1;});
  });

  NdArrayRef x = gfmp_batch_inverse(y);
  NdArrayRef ones = ring_ones(field, y.shape());
  // WHEN
  NdArrayRef z = gfmp_mul_mod(x, y);  // x = y;

  // THEN
  EXPECT_TRUE(ring_all_equal(z, ones));
}

}