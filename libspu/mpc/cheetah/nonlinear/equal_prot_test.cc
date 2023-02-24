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

#include "libspu/mpc/cheetah/nonlinear/equal_prot.h"

#include <random>

#include "gtest/gtest.h"

#include "libspu/core/xt_helper.h"
#include "libspu/mpc/cheetah/ot/basic_ot_prot.h"
#include "libspu/mpc/semi2k/type.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::mpc::cheetah {

class EqualProtTest : public ::testing::TestWithParam<FieldType> {
  void SetUp() override {}
};

INSTANTIATE_TEST_SUITE_P(
    Cheetah, EqualProtTest,
    testing::Values(FieldType::FM32, FieldType::FM64, FieldType::FM128),
    [](const testing::TestParamInfo<EqualProtTest::ParamType> &p) {
      return fmt::format("{}", p.param);
    });

TEST_P(EqualProtTest, Basic) {
  size_t kWorldSize = 2;
  size_t n = 11;
  FieldType field = GetParam();

  ArrayRef inp[2];
  inp[0] = ring_rand(field, n);
  inp[1] = ring_rand(field, n);

  DISPATCH_ALL_FIELDS(field, "", [&]() {
    auto xinp0 = xt_mutable_adapt<ring2k_t>(inp[0]);
    auto xinp1 = xt_mutable_adapt<ring2k_t>(inp[1]);
    std::copy_n(xinp1.data(), 5, xinp0.data());
  });

  ArrayRef eq_oup[2];
  utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> ctx) {
    auto conn = std::make_shared<Communicator>(ctx);
    int rank = ctx->Rank();
    auto base = std::make_shared<BasicOTProtocols>(conn);
    EqualProtocol eq_prot(base);
    eq_oup[rank] = eq_prot.Compute(inp[rank]);
  });

  DISPATCH_ALL_FIELDS(field, "", [&]() {
    auto xeq0 = xt_adapt<ring2k_t>(eq_oup[0]);
    auto xeq1 = xt_adapt<ring2k_t>(eq_oup[1]);
    auto xinp0 = xt_adapt<ring2k_t>(inp[0]);
    auto xinp1 = xt_adapt<ring2k_t>(inp[1]);

    for (size_t i = 0; i < n; ++i) {
      bool expected = xinp0[i] == xinp1[i];
      bool got_eq = xeq0[i] ^ xeq1[i];
      EXPECT_EQ(expected, got_eq);
    }
  });
}

}  // namespace spu::mpc::cheetah
