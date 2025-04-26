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

#include "yacl/crypto/rand/rand.h"

#include "libspu/mpc/api.h"
#include "libspu/mpc/api_test.h"
#include "libspu/mpc/cheetah/protocol.h"
#include "libspu/mpc/cheetah/type.h"
#include "libspu/mpc/utils/permute.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::mpc::test {
namespace {

RuntimeConfig makeConfig(FieldType field) {
  RuntimeConfig conf;
  conf.protocol = ProtocolKind::CHEETAH;
  conf.field = field;
  conf.cheetah_2pc_config.ot_kind = CheetahOtKind::YACL_Softspoken;
  return conf;
}

}  // namespace

INSTANTIATE_TEST_SUITE_P(
    Cheetah, ApiTest,
    testing::Combine(testing::Values(makeCheetahProtocol),           //
                     testing::Values(makeConfig(FieldType::FM32),    //
                                     makeConfig(FieldType::FM64),    //
                                     makeConfig(FieldType::FM128)),  //
                     testing::Values(2)),                            //
    [](const testing::TestParamInfo<ApiTest::ParamType>& p) {
      return fmt::format("{}x{}", std::get<1>(p.param).field,
                         std::get<2>(p.param));
    });

namespace {
NdArrayRef mockPshare(SPUContext* ctx, const Index& perm) {
  NdArrayRef out(makeType<cheetah::PShrTy>(),
                 {static_cast<int64_t>(perm.size())});
  const auto field = out.eltype().as<cheetah::PShrTy>()->field();

  DISPATCH_ALL_FIELDS(field, [&]() {
    NdArrayView<ring2k_t> _out(out);
    pforeach(0, out.numel(),
             [&](int64_t idx) { _out[idx] = ring2k_t(perm[idx]); });
  });

  return out;
}
}  // namespace

class PermuteTest : public ::testing::TestWithParam<OpTestParams> {};

INSTANTIATE_TEST_SUITE_P(
    Semi2k, PermuteTest,
    testing::Combine(testing::Values(CreateObjectFn(makeCheetahProtocol)),
                     testing::Values(makeConfig(FieldType::FM32),
                                     makeConfig(FieldType::FM64),
                                     makeConfig(FieldType::FM128)),
                     testing::Values(2)),
    [](const testing::TestParamInfo<PermuteTest::ParamType>& p) {
      return fmt::format("{}x{}", std::get<1>(p.param).field,
                         std::get<2>(p.param));
      ;
    });

TEST_P(PermuteTest, Perm_Work) {
  const auto factory = std::get<0>(GetParam());
  const RuntimeConfig& conf = std::get<1>(GetParam());
  const size_t npc = std::get<2>(GetParam());

  int64_t n = 537;
  const Shape shape = {n};

  uint64_t cnt = yacl::crypto::RandU64();
  uint128_t seed1 = yacl::crypto::RandU128();
  uint128_t seed2 = yacl::crypto::RandU128();
  const Index perm1 = genRandomPerm(n, seed1, &cnt);
  const Index perm2 = genRandomPerm(n, seed2, &cnt);

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto sctx = makeCheetahProtocol(conf, lctx);

    // GIVEN
    NdArrayRef perm;
    if (lctx->Rank() == 0) {
      perm = mockPshare(sctx.get(), perm1);
    } else {
      perm = mockPshare(sctx.get(), perm2);
    }

    auto x_p = rand_p(sctx.get(), shape);
    auto x_s = p2s(sctx.get(), x_p);

    // WHEN
    auto permuted_x = perm_ss(sctx.get(), x_s, WrapValue(perm));
    EXPECT_TRUE(permuted_x.has_value());

    auto permuted_x_p = s2p(sctx.get(), permuted_x.value());

    // THEN
    auto required = applyInvPerm(UnwrapValue(x_p), perm1);
    required = applyInvPerm(required, perm2);

    EXPECT_EQ(permuted_x_p.shape(), required.shape());
    EXPECT_TRUE(ring_all_equal(permuted_x_p.data(), required));
  });
}

// test whether inv_perm(perm(x)) == x
TEST_P(PermuteTest, InvPerm_Perm_Work) {
  const auto factory = std::get<0>(GetParam());
  const RuntimeConfig& conf = std::get<1>(GetParam());
  const size_t npc = std::get<2>(GetParam());

  int64_t n = 537;
  const Shape shape = {n};

  uint64_t cnt = yacl::crypto::RandU64();
  uint128_t seed1 = yacl::crypto::RandU128();
  uint128_t seed2 = yacl::crypto::RandU128();
  const Index perm1 = genRandomPerm(n, seed1, &cnt);
  const Index perm2 = genRandomPerm(n, seed2, &cnt);

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto sctx = makeCheetahProtocol(conf, lctx);

    // GIVEN
    NdArrayRef perm;
    if (lctx->Rank() == 0) {
      perm = mockPshare(sctx.get(), perm1);
    } else {
      perm = mockPshare(sctx.get(), perm2);
    }

    auto x_p = rand_p(sctx.get(), shape);
    auto x_s = p2s(sctx.get(), x_p);

    // WHEN
    auto permuted_x = perm_ss(sctx.get(), x_s, WrapValue(perm));
    EXPECT_TRUE(permuted_x.has_value());
    auto inv_permuted_x =
        inv_perm_ss(sctx.get(), permuted_x.value(), WrapValue(perm));
    EXPECT_TRUE(inv_permuted_x.has_value());

    auto inv_permuted_x_p = s2p(sctx.get(), inv_permuted_x.value());

    // THEN
    EXPECT_EQ(inv_permuted_x_p.shape(), x_p.shape());
    EXPECT_TRUE(ring_all_equal(inv_permuted_x_p.data(), x_p.data()));
  });
}

}  // namespace spu::mpc::test
