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

#include "libspu/mpc/semi2k/sort_test.h"

#include "libspu/core/ndarray_ref.h"
#include "libspu/core/prelude.h"
#include "libspu/mpc/ab_api.h"
#include "libspu/mpc/api.h"
#include "libspu/mpc/utils/permute.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::mpc::test {

namespace {

#define EXPECT_VALUE_EQ(X, Y)                            \
  {                                                      \
    EXPECT_EQ((X).shape(), (Y).shape());                 \
    EXPECT_TRUE(ring_all_equal((X).data(), (Y).data())); \
  }

Shape kShape = {20};
const int64_t kInputsSize = 10;

}  // namespace

TEST_P(PermuteTest, RadixSort) {
  const auto factory = std::get<0>(GetParam());
  const RuntimeConfig& conf = std::get<1>(GetParam());
  const size_t npc = std::get<2>(GetParam());

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto obj = factory(conf, lctx);

    /* GIVEN */
    std::vector<Value> in_p(kInputsSize);
    std::vector<Value> in_s(kInputsSize);
    for (size_t i = 0; i < kInputsSize; ++i) {
      in_p[i] = rand_p(obj.get(), kShape);
      in_s[i] = p2a(obj.get(), in_p[i]);
    }

    /* WHEN */
    auto sorted_s = dynDispatch<std::vector<Value>>(obj.get(), "sort_a", in_s);

    /* THEN */
    const auto perm = genInversePerm(genPermBySort(UnwrapValue(in_p[0])));

    for (size_t i = 0; i < kInputsSize; ++i) {
      auto expected_sorted = applyInvPerm(UnwrapValue(in_p[i]), perm);
      auto actual_sorted = a2p(obj.get(), sorted_s[i]);
      EXPECT_VALUE_EQ(WrapValue(expected_sorted), actual_sorted);
    }
  });
}

}  // namespace spu::mpc::test