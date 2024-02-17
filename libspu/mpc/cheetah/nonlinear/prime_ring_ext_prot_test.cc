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

#include "libspu/mpc/cheetah/nonlinear/prime_ring_ext_prot.h"

#include <optional>
#include <random>
#include <type_traits>

#include "gtest/gtest.h"

#include "libspu/core/ndarray_ref.h"
#include "libspu/core/parallel_utils.h"
#include "libspu/core/type_util.h"
#include "libspu/mpc/cheetah/ot/basic_ot_prot.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::mpc::cheetah {

class PrimeRingExtendProtocolTest
    : public ::testing::TestWithParam<std::tuple<FieldType, bool>> {
 public:
  static constexpr int64_t kBenchSize = 1LL << 10;
  void SetUp() override {}
};

INSTANTIATE_TEST_SUITE_P(
    Cheetah, PrimeRingExtendProtocolTest,
    testing::Combine(testing::Values(FieldType::FM32, FieldType::FM64),
                     testing::Values(true, false)),
    [](const testing::TestParamInfo<PrimeRingExtendProtocolTest::ParamType>
           &p) {
      return fmt::format("{}Truc{}", std::get<0>(p.param),
                         (int)std::get<1>(p.param));
    });

template <typename T>
bool SignBit(T x) {
  using U = typename std::make_unsigned<T>::type;
  return static_cast<U>(x) >> (8 * sizeof(T) - 1) & 1;
}

void MaskItInplace(NdArrayRef a, size_t width) {
  auto field = a.eltype().as<Ring2k>()->field();
  if (width == SizeOf(field) * 8) {
    return;
  }
  DISPATCH_ALL_FIELDS(field, "mask", [&]() {
    NdArrayView<ring2k_t> _a(a);
    ring2k_t msk = (static_cast<ring2k_t>(1) << width) - 1;
    pforeach(0, _a.numel(), [&](int64_t i) { _a[i] &= msk; });
  });
}

// view [0, 2^k) as [-2^k/2, 2^k/2)
template <typename U>
auto ToSignType(U x, size_t width) {
  using S = typename std::make_signed<U>::type;
  if (sizeof(U) * 8 == width) {
    return static_cast<S>(x);
  }

  U half = static_cast<U>(1) << (width - 1);
  if (x >= half) {
    U upper = static_cast<U>(1) << width;
    x -= upper;
  }
  return static_cast<S>(x);
}

template <typename T>
T GetPrime();

template <>
uint32_t GetPrime() {
  return (static_cast<uint32_t>(1) << 31) - 1;
}

template <>
uint64_t GetPrime() {
  return (static_cast<uint64_t>(1) << 61) - 1;
}

template <>
uint128_t GetPrime() {
  return (static_cast<uint128_t>(1) << 61) - 1;
}

TEST_P(PrimeRingExtendProtocolTest, Basic) {
  size_t kWorldSize = 2;
  Shape shape = {kBenchSize};

  FieldType src_field = std::get<0>(GetParam());
  FieldType dst_field = src_field;

  bool do_trunc = std::get<1>(GetParam());
  int shft = do_trunc ? 11 : 0;

  NdArrayRef inp[2];
  inp[0] = ring_rand(src_field, shape);
  inp[1] = ring_zeros(src_field, shape);
  auto msg = ring_zeros(src_field, shape);

  DISPATCH_ALL_FIELDS(src_field, "setup", [&]() {
    using signedT = std::make_signed<ring2k_t>::type;
    signedT prime = GetPrime<ring2k_t>();
    std::uniform_int_distribution<signedT> uniform(
        -(prime >> PrimeRingExtendProtocol::kHeuristicBound),
        prime >> PrimeRingExtendProtocol::kHeuristicBound);
    std::default_random_engine rdv;

    auto xmsg = NdArrayView<signedT>(msg);
    auto rnd0 = NdArrayView<ring2k_t>(inp[0]);
    auto rnd1 = NdArrayView<ring2k_t>(inp[1]);
    for (int64_t i = 0; i < shape.numel(); ++i) {
      rnd0[i] %= prime;

      xmsg[i] = uniform(rdv);

      rnd1[i] = xmsg[i] > 0 ? xmsg[i] : prime - std::abs(xmsg[i]);
      rnd1[i] = (rnd1[i] + prime - rnd0[i]) % prime;
    }
  });

  NdArrayRef oup[2];
  utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> ctx) {
    int rank = ctx->Rank();
    auto conn = std::make_shared<Communicator>(ctx);
    auto base = std::make_shared<BasicOTProtocols>(conn);
    PrimeRingExtendProtocol prot(base);
    PrimeRingExtendProtocol::Meta meta;

    meta.dst_ring = dst_field;
    meta.dst_width = SizeOf(dst_field) * 8;
    meta.truncate_nbits = std::nullopt;
    if (do_trunc) {
      meta.truncate_nbits = shft;
    }
    DISPATCH_ALL_FIELDS(src_field, "set_prime",
                        [&]() { meta.prime = GetPrime<ring2k_t>(); });

    [[maybe_unused]] size_t b0 = ctx->GetStats()->sent_bytes;
    oup[rank] = prot.Compute(inp[rank], meta);
    [[maybe_unused]] size_t b1 = ctx->GetStats()->sent_bytes;
    SPDLOG_DEBUG("PrimeExt {} bits to {} bits sent {} bits per",
                 absl::bit_width(meta.prime), meta.dst_width,
                 (b1 - b0) * 8. / shape.numel());
  });

  EXPECT_EQ(oup[0].shape(), oup[1].shape());
  auto got = ring_add(oup[0], oup[1]);

  DISPATCH_ALL_FIELDS(src_field, "check", [&]() {
    using S0 = std::make_signed<ring2k_t>::type;
    NdArrayView<S0> expS(msg);
    DISPATCH_ALL_FIELDS(dst_field, "check", [&]() {
      using S1 = std::make_signed<ring2k_t>::type;

      NdArrayView<S1> gotS(got);
      if (do_trunc) {
        for (int64_t i = 0; i < shape.numel(); ++i) {
          ASSERT_NEAR(static_cast<S1>(expS[i]) >> shft, gotS[i], 1);
        }
      } else {
        for (int64_t i = 0; i < shape.numel(); ++i) {
          ASSERT_EQ(static_cast<S1>(expS[i]), gotS[i]);
        }
      }
    });
  });
}

TEST_P(PrimeRingExtendProtocolTest, RingUp) {
  size_t kWorldSize = 2;
  Shape shape = {kBenchSize};

  FieldType src_field = std::get<0>(GetParam());
  FieldType dst_field = FM128;
  if (src_field == FM32) {
    dst_field = FM64;
  }

  bool do_trunc = std::get<1>(GetParam());
  int shft = do_trunc ? 11 : 0;

  NdArrayRef inp[2];
  inp[0] = ring_rand(src_field, shape);
  inp[1] = ring_zeros(src_field, shape);
  auto msg = ring_zeros(src_field, shape);

  DISPATCH_ALL_FIELDS(src_field, "setup", [&]() {
    using signedT = std::make_signed<ring2k_t>::type;
    signedT prime = GetPrime<ring2k_t>();
    std::uniform_int_distribution<signedT> uniform(
        -(prime >> PrimeRingExtendProtocol::kHeuristicBound),
        prime >> PrimeRingExtendProtocol::kHeuristicBound);
    std::default_random_engine rdv;

    auto xmsg = NdArrayView<signedT>(msg);
    auto rnd0 = NdArrayView<ring2k_t>(inp[0]);
    auto rnd1 = NdArrayView<ring2k_t>(inp[1]);
    for (int64_t i = 0; i < shape.numel(); ++i) {
      rnd0[i] %= prime;

      xmsg[i] = uniform(rdv);

      rnd1[i] = xmsg[i] > 0 ? xmsg[i] : prime - std::abs(xmsg[i]);
      rnd1[i] = (rnd1[i] + prime - rnd0[i]) % prime;
    }
  });

  NdArrayRef oup[2];
  utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> ctx) {
    int rank = ctx->Rank();
    auto conn = std::make_shared<Communicator>(ctx);
    auto base = std::make_shared<BasicOTProtocols>(conn);
    PrimeRingExtendProtocol prot(base);
    PrimeRingExtendProtocol::Meta meta;

    meta.dst_ring = dst_field;
    meta.dst_width = SizeOf(dst_field) * 8;
    meta.truncate_nbits = std::nullopt;
    if (do_trunc) {
      meta.truncate_nbits = shft;
    }
    DISPATCH_ALL_FIELDS(src_field, "set_prime",
                        [&]() { meta.prime = GetPrime<ring2k_t>(); });

    [[maybe_unused]] size_t b0 = ctx->GetStats()->sent_bytes;
    oup[rank] = prot.Compute(inp[rank], meta);
    [[maybe_unused]] size_t b1 = ctx->GetStats()->sent_bytes;
    SPDLOG_DEBUG("PrimeExt {} bits to {} bits sent {} bits per",
                 absl::bit_width(meta.prime), meta.dst_width,
                 (b1 - b0) * 8. / shape.numel());
  });

  EXPECT_EQ(oup[0].shape(), oup[1].shape());
  auto got = ring_add(oup[0], oup[1]);

  DISPATCH_ALL_FIELDS(src_field, "check", [&]() {
    using S0 = std::make_signed<ring2k_t>::type;
    NdArrayView<S0> expS(msg);
    DISPATCH_ALL_FIELDS(dst_field, "check", [&]() {
      using S1 = std::make_signed<ring2k_t>::type;

      NdArrayView<S1> gotS(got);
      if (do_trunc) {
        for (int64_t i = 0; i < shape.numel(); ++i) {
          ASSERT_NEAR(static_cast<S1>(expS[i]) >> shft, gotS[i], 1);
        }
      } else {
        for (int64_t i = 0; i < shape.numel(); ++i) {
          ASSERT_EQ(static_cast<S1>(expS[i]), gotS[i]);
        }
      }
    });
  });
}

TEST_P(PrimeRingExtendProtocolTest, RingUpWithSpecificOutWidth) {
  size_t kWorldSize = 2;
  Shape shape = {kBenchSize};

  FieldType src_field = std::get<0>(GetParam());
  FieldType dst_field = FM128;
  if (src_field == FM32) {
    dst_field = FM64;
  }

  bool do_trunc = std::get<1>(GetParam());
  int shft = do_trunc ? 11 : 0;

  NdArrayRef inp[2];
  inp[0] = ring_rand(src_field, shape);
  inp[1] = ring_zeros(src_field, shape);
  auto msg = ring_zeros(src_field, shape);

  DISPATCH_ALL_FIELDS(src_field, "setup", [&]() {
    using signedT = std::make_signed<ring2k_t>::type;
    signedT prime = GetPrime<ring2k_t>();
    std::uniform_int_distribution<signedT> uniform(
        -(prime >> PrimeRingExtendProtocol::kHeuristicBound),
        prime >> PrimeRingExtendProtocol::kHeuristicBound);
    std::default_random_engine rdv;

    auto xmsg = NdArrayView<signedT>(msg);
    auto rnd0 = NdArrayView<ring2k_t>(inp[0]);
    auto rnd1 = NdArrayView<ring2k_t>(inp[1]);
    for (int64_t i = 0; i < shape.numel(); ++i) {
      rnd0[i] %= prime;

      xmsg[i] = uniform(rdv);

      rnd1[i] = xmsg[i] > 0 ? xmsg[i] : prime - std::abs(xmsg[i]);
      rnd1[i] = (rnd1[i] + prime - rnd0[i]) % prime;
    }
  });

  NdArrayRef oup[2];
  size_t outwidth;
  utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> ctx) {
    int rank = ctx->Rank();
    auto conn = std::make_shared<Communicator>(ctx);
    auto base = std::make_shared<BasicOTProtocols>(conn);
    PrimeRingExtendProtocol prot(base);
    PrimeRingExtendProtocol::Meta meta;

    meta.dst_ring = dst_field;
    meta.dst_width = SizeOf(src_field) * 8 + 20;
    meta.truncate_nbits = std::nullopt;
    if (do_trunc) {
      meta.truncate_nbits = shft;
    }
    DISPATCH_ALL_FIELDS(src_field, "set_prime",
                        [&]() { meta.prime = GetPrime<ring2k_t>(); });

    [[maybe_unused]] size_t b0 = ctx->GetStats()->sent_bytes;
    oup[rank] = prot.Compute(inp[rank], meta);
    [[maybe_unused]] size_t b1 = ctx->GetStats()->sent_bytes;
    SPDLOG_DEBUG("PrimeExt {} bits to {} bits sent {} bits per",
                 absl::bit_width(meta.prime), meta.dst_width,
                 (b1 - b0) * 8. / shape.numel());

    outwidth = meta.dst_width;
  });

  EXPECT_EQ(oup[0].shape(), oup[1].shape());
  auto got = ring_add(oup[0], oup[1]);
  MaskItInplace(got, outwidth);

  DISPATCH_ALL_FIELDS(src_field, "check", [&]() {
    using S0 = std::make_signed<ring2k_t>::type;
    NdArrayView<S0> expS(msg);
    DISPATCH_ALL_FIELDS(dst_field, "check", [&]() {
      using U1 = std::make_signed<ring2k_t>::type;

      NdArrayView<U1> gotS(got);
      if (do_trunc) {
        for (int64_t i = 0; i < shape.numel(); ++i) {
          ASSERT_NEAR(expS[i] >> shft, ToSignType(gotS[i], outwidth), 1);
        }
      } else {
        for (int64_t i = 0; i < shape.numel(); ++i) {
          ASSERT_EQ(expS[i], ToSignType(gotS[i], outwidth));
        }
      }
    });
  });
}

}  // namespace spu::mpc::cheetah
