// Copyright 2024 Ant Group Co., Ltd.
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

#include "libspu/mpc/shamir/prot_shamir_test.h"

#include "libspu/core/prelude.h"
#include "libspu/mpc/ab_api.h"
#include "libspu/mpc/api.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/kernel.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::mpc::test {
namespace {

Shape kShape = {20, 30};
const std::vector<size_t> kShiftBits = {0, 1, 2, 31, 32, 33, 64, 1000};

#define EXPECT_VALUE_EQ(X, Y)                            \
  {                                                      \
    EXPECT_EQ((X).shape(), (Y).shape());                 \
    EXPECT_TRUE(ring_all_equal((X).data(), (Y).data())); \
  }

#define EXPECT_VALUE_ALMOST_EQ(X, Y, ERR)                     \
  {                                                           \
    EXPECT_EQ((X).shape(), (Y).shape());                      \
    EXPECT_TRUE(ring_all_equal((X).data(), (Y).data(), ERR)); \
  }

bool verifyCost(Kernel* kernel, std::string_view name, const ce::Params& params,
                const Communicator::Stats& cost, size_t repeated = 1) {
  if (kernel->kind() == Kernel::Kind::Dynamic) {
    return true;
  }

  auto comm = kernel->comm();
  auto latency = kernel->latency();

  bool succeed = true;
  constexpr size_t kBitsPerBytes = 8;
  const auto expectedComm = comm->eval(params) * repeated;
  const auto realComm = cost.comm * kBitsPerBytes;

  float diff;
  if (expectedComm == 0) {
    diff = realComm;
  } else {
    diff = std::abs(static_cast<float>(realComm - expectedComm)) / expectedComm;
  }
  if (realComm < expectedComm || diff > kernel->getCommTolerance()) {
    fmt::print("Failed: {} comm mismatch, expected={}, got={}\n", name,
               expectedComm, realComm);
    succeed = false;
  }

  if (latency->eval(params) != cost.latency) {
    fmt::print("Failed: {} latency mismatch, expected={}, got={}\n", name,
               latency->eval(params), cost.latency);
    succeed = false;
  }

  return succeed;
}

bool verifyCost(Kernel* kernel, std::string_view name, FieldType field,
                const Shape& shape, size_t npc,
                const Communicator::Stats& cost) {
  ce::Params params = {{"K", SizeOf(field) * 8}, {"N", npc}};
  return verifyCost(kernel, name, params, cost, shape.numel() /*repeated*/);
}

}  // namespace

TEST_P(ShamirProtTest, MulAAA) {
  const auto factory = std::get<0>(GetParam());
  const RuntimeConfig& conf = std::get<1>(GetParam());
  const size_t npc = std::get<2>(GetParam());

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto sctx = factory(conf, lctx);

    auto p0 = rand_p(sctx.get(), kShape);
    auto p1 = rand_p(sctx.get(), kShape);
    auto p2 = rand_p(sctx.get(), kShape);

    auto v0 = p2v(sctx.get(), p0, 0);
    auto v1 = p2v(sctx.get(), p1, 1);
    auto v2 = p2v(sctx.get(), p2, 2);

    auto a0 = v2a(sctx.get(), v0);
    auto a1 = v2a(sctx.get(), v1);
    auto a2 = v2a(sctx.get(), v2);

    auto prod = mul_aaa(sctx.get(), a0, a1, a2);
    auto p_prod = a2p(sctx.get(), prod);

    auto s = mul_pp(sctx.get(), p0, p1);
    auto s_prime = mul_pp(sctx.get(), s, p2);

    /* THEN */
    EXPECT_VALUE_EQ(s_prime, p_prod);
  });
}

TEST_P(ShamirProtTest, MulAAP) {
  const auto factory = std::get<0>(GetParam());
  const RuntimeConfig& conf = std::get<1>(GetParam());
  const size_t npc = std::get<2>(GetParam());

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto sctx = factory(conf, lctx);

    auto p0 = rand_p(sctx.get(), kShape);
    auto p1 = rand_p(sctx.get(), kShape);

    auto v0 = p2v(sctx.get(), p0, 0);
    auto v1 = p2v(sctx.get(), p1, 1);

    auto a0 = v2a(sctx.get(), v0);
    auto a1 = v2a(sctx.get(), v1);

    auto prod = mul_aa_p(sctx.get(), a0, a1);

    auto s = mul_pp(sctx.get(), p0, p1);

    /* THEN */
    EXPECT_VALUE_EQ(s, prod);
  });
}

TEST_P(ShamirProtTest, MulAATrunc) {
  const auto factory = std::get<0>(GetParam());
  const RuntimeConfig& conf = std::get<1>(GetParam());
  const size_t npc = std::get<2>(GetParam());

  // ArrayRef p0_large =
  //     ring_rand_range(conf.field(), kShape, -(1 << 28), -(1 << 27));
  // ArrayRef p0_small = ring_rand_range(conf.field(), kShape, 1, 10000);

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto obj = factory(conf, lctx);

    auto p0 = rand_p(obj.get(), kShape);
    auto p1 = rand_p(obj.get(), kShape);

    auto bits_range_gap = p0.elsize() * 8 - (p0.elsize() * 8) / 2;
    p0 = arshift_p(obj.get(), p0, {static_cast<int64_t>(bits_range_gap)});
    p1 = arshift_p(obj.get(), p1, {static_cast<int64_t>(bits_range_gap)});
    auto prod = mul_pp(obj.get(), p0, p1);

    auto v0 = p2v(obj.get(), p0, 0);
    auto v1 = p2v(obj.get(), p1, 1);

    /* GIVEN */
    auto a0 = v2a(obj.get(), v0);
    auto a1 = v2a(obj.get(), v1);

    /* WHEN */
    const size_t bits = 2;
    auto prev = obj->prot()->getState<Communicator>()->getStats();
    auto prod_a = mul_aa_trunc(obj.get(), a0, a1, bits, SignType::Unknown);
    auto cost = obj->prot()->getState<Communicator>()->getStats() - prev;

    auto r_a = a2p(obj.get(), prod_a);
    auto r_p = arshift_p(obj.get(), prod, {static_cast<int64_t>(bits)});

    /* THEN */
    EXPECT_VALUE_ALMOST_EQ(r_a, r_p, npc);
    EXPECT_TRUE(verifyCost(obj->prot()->getKernel("mul_aa_trunc"),
                           "mul_aa_trunc", conf.field(), kShape, npc, cost));
  });
}

TEST_P(ShamirProtTest, ReLU) {
  const auto factory = std::get<0>(GetParam());
  const RuntimeConfig& conf = std::get<1>(GetParam());
  const size_t npc = std::get<2>(GetParam());

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto sctx = factory(conf, lctx);

    auto p0 = rand_p(sctx.get(), kShape);

    // SECURENN has an msb input range requirement here
    if (conf.protocol() == ProtocolKind::SECURENN) {
      p0 = arshift_p(sctx.get(), p0, {1});
    }

    auto relu_s = relu(sctx.get(), p2s(sctx.get(), p0));

    auto r_p = msb_p(sctx.get(), p0);
    auto d_relu = add_pp(sctx.get(), make_p(sctx.get(), 1, kShape),
                         negate_p(sctx.get(), r_p));
    auto relu_p = mul_pp(sctx.get(), d_relu, p0);

    /* THEN */
    EXPECT_VALUE_EQ(s2p(sctx.get(), relu_s), relu_p);
  });
}
}