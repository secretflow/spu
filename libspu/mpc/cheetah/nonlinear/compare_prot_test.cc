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

#include "libspu/mpc/cheetah/nonlinear/compare_prot.h"

#include <random>

#include "gtest/gtest.h"

#include "libspu/core/xt_helper.h"
#include "libspu/mpc/cheetah/ot/basic_ot_prot.h"
#include "libspu/mpc/semi2k/type.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::mpc::cheetah {

class CompareProtTest
    : public ::testing::TestWithParam<std::tuple<FieldType, bool>> {
  void SetUp() override {}
};

INSTANTIATE_TEST_SUITE_P(
    Cheetah, CompareProtTest,
    testing::Combine(testing::Values(FieldType::FM32, FieldType::FM64,
                                     FieldType::FM128),
                     testing::Values(true, false)),
    [](const testing::TestParamInfo<CompareProtTest::ParamType> &p) {
      return fmt::format("{}Greater{}", std::get<0>(p.param),
                         (int)std::get<1>(p.param));
    });

TEST_P(CompareProtTest, Basic) {
  size_t kWorldSize = 2;
  size_t n = 100;
  FieldType field = std::get<0>(GetParam());
  bool greater_than = std::get<1>(GetParam());

  ArrayRef inp[2];
  inp[0] = ring_rand(field, n);
  inp[1] = ring_rand(field, n);

  DISPATCH_ALL_FIELDS(field, "", [&]() {
    auto xinp = xt_mutable_adapt<ring2k_t>(inp[0]);
    xinp[0] = 1;
    xinp[1] = 10;
    xinp[2] = 100;

    xinp = xt_mutable_adapt<ring2k_t>(inp[1]);
    xinp[0] = 1;
    xinp[1] = 9;
    xinp[2] = 1000;
  });

  ArrayRef cmp_oup[2];
  ArrayRef eq_oup[2];
  utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> ctx) {
    int rank = ctx->Rank();
    auto base = std::make_shared<BasicOTProtocols>(ctx);
    CompareProtocol comp_prot(base);
    auto [_c, _e] = comp_prot.ComputeWithEq(inp[rank], greater_than);
    cmp_oup[rank] = _c;
    eq_oup[rank] = _e;
  });

  DISPATCH_ALL_FIELDS(field, "", [&]() {
    auto xout0 = xt_adapt<ring2k_t>(cmp_oup[0]);
    auto xout1 = xt_adapt<ring2k_t>(cmp_oup[1]);
    auto xeq0 = xt_adapt<ring2k_t>(eq_oup[0]);
    auto xeq1 = xt_adapt<ring2k_t>(eq_oup[1]);
    auto xinp0 = xt_adapt<ring2k_t>(inp[0]);
    auto xinp1 = xt_adapt<ring2k_t>(inp[1]);

    for (size_t i = 0; i < n; ++i) {
      bool expected = greater_than ? xinp0[i] > xinp1[i] : xinp0[i] < xinp1[i];
      bool got_cmp = xout0[i] ^ xout1[i];
      bool got_eq = xeq0[i] ^ xeq1[i];
      EXPECT_EQ(expected, got_cmp);
      EXPECT_EQ((xinp0[i] == xinp1[i]), got_eq);
    }
  });
}

}  // namespace spu::mpc::cheetah
