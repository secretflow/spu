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

#include "libspu/mpc/cheetah/nonlinear/ring_ext_prot.h"

#include <type_traits>

#include "gtest/gtest.h"

#include "libspu/core/ndarray_ref.h"
#include "libspu/core/parallel_utils.h"
#include "libspu/core/type_util.h"
#include "libspu/mpc/cheetah/ot/basic_ot_prot.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::mpc::cheetah {

class RingExtendProtocolTest : public ::testing::TestWithParam<
                                   std::tuple<FieldType, bool, std::string>> {
 public:
  static constexpr int64_t kBenchSize = 1LL << 10;
  void SetUp() override {}
};

INSTANTIATE_TEST_SUITE_P(
    Cheetah, RingExtendProtocolTest,
    testing::Combine(testing::Values(FieldType::FM32, FieldType::FM64),
                     testing::Values(true, false),
                     testing::Values("Unknown", "Positive", "Negative")),
    [](const testing::TestParamInfo<RingExtendProtocolTest::ParamType> &p) {
      return fmt::format("{}{}MSB{}", std::get<0>(p.param),
                         std::get<1>(p.param) ? "Signed" : "Unsigned",
                         std::get<2>(p.param));
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

TEST_P(RingExtendProtocolTest, Basic) {
  size_t kWorldSize = 2;
  Shape shape = {kBenchSize};

  FieldType src_field = std::get<0>(GetParam());
  FieldType dst_field;
  if (src_field == FM32) {
    dst_field = FM64;
  } else {
    dst_field = FM128;
  }

  bool signed_arith = std::get<1>(GetParam());
  std::string sign_type_s = std::get<2>(GetParam());

  SignType sign_type;
  NdArrayRef inp[2];
  inp[0] = ring_rand(src_field, shape);

  if (sign_type_s == "Unknown") {
    inp[1] = ring_rand(src_field, shape);
    sign_type = SignType::Unknown;
  } else {
    auto msg = ring_rand(src_field, shape);
    DISPATCH_ALL_FIELDS(src_field, "setup", [&]() {
      auto xmsg = NdArrayView<ring2k_t>(msg);
      size_t bw = SizeOf(src_field) * 8;
      if (sign_type_s == "Positive") {
        auto msk = (static_cast<ring2k_t>(1) << (bw - 1)) - 1;
        pforeach(0, xmsg.numel(), [&](int64_t i) { xmsg[i] &= msk; });
        sign_type = SignType::Positive;
      } else {
        auto flag = static_cast<ring2k_t>(1) << (bw - 1);
        pforeach(0, xmsg.numel(), [&](int64_t i) { xmsg[i] |= flag; });
        sign_type = SignType::Negative;
      }
    });

    inp[1] = ring_sub(msg, inp[0]);
  }

  NdArrayRef oup[2];
  utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> ctx) {
    int rank = ctx->Rank();
    auto conn = std::make_shared<Communicator>(ctx);
    auto base = std::make_shared<BasicOTProtocols>(conn);
    RingExtendProtocol prot(base);
    RingExtendProtocol::Meta meta;

    meta.sign = sign_type;
    meta.use_heuristic = false;
    meta.signed_arith = signed_arith;
    meta.src_ring = src_field;
    meta.dst_ring = dst_field;
    meta.src_width = SizeOf(src_field) * 8;
    meta.dst_width = SizeOf(dst_field) * 8;

    [[maybe_unused]] size_t b0 = ctx->GetStats()->sent_bytes;
    oup[rank] = prot.Compute(inp[rank], meta);
    [[maybe_unused]] size_t b1 = ctx->GetStats()->sent_bytes;
    SPDLOG_INFO("Ext {} bits to {} bits sent {} bits per", meta.src_width,
                meta.dst_width, (b1 - b0) * 8. / shape.numel());
  });

  EXPECT_EQ(oup[0].shape(), oup[1].shape());
  auto expected = ring_add(inp[0], inp[1]);
  auto got = ring_add(oup[0], oup[1]);

  DISPATCH_ALL_FIELDS(src_field, "check", [&]() {
    using U0 = std::make_unsigned<ring2k_t>::type;
    using S0 = std::make_signed<ring2k_t>::type;
    NdArrayView<U0> expU(expected);
    NdArrayView<S0> expS(expected);
    DISPATCH_ALL_FIELDS(dst_field, "check", [&]() {
      using U1 = std::make_unsigned<ring2k_t>::type;
      using S1 = std::make_signed<ring2k_t>::type;

      NdArrayView<U1> gotU(got);
      NdArrayView<S1> gotS(got);
      if (signed_arith) {
        for (int64_t i = 0; i < shape.numel(); ++i) {
          ASSERT_EQ(static_cast<S1>(expS[i]), gotS[i]);
        }
      } else {
        for (int64_t i = 0; i < shape.numel(); ++i) {
          ASSERT_EQ(expU[i], gotU[i]);
        }
      }
    });
  });
}

TEST_P(RingExtendProtocolTest, Heuristic) {
  size_t kWorldSize = 2;
  Shape shape = {kBenchSize};

  FieldType src_field = std::get<0>(GetParam());
  FieldType dst_field;
  if (src_field == FM32) {
    dst_field = FM64;
  } else {
    dst_field = FM128;
  }

  bool signed_arith = std::get<1>(GetParam());
  std::string sign_type_s = std::get<2>(GetParam());

  NdArrayRef inp[2];
  inp[0] = ring_rand(src_field, shape);
  if (sign_type_s != "Unknown" or not signed_arith) {
    // 1. When sign type is already known, no need to use heuristic.
    // 2. Heuristic is only for signed arith
    return;
  }

  auto msg = ring_rand(src_field, shape);
  ring_rshift_(msg, RingExtendProtocol::kHeuristicBound);

  DISPATCH_ALL_FIELDS(src_field, "setup", [&]() {
    auto xmsg = NdArrayView<ring2k_t>(msg);
    // some are positive, some are negative
    for (int64_t i = 0; i < shape.numel(); i += 2) {
      xmsg[i] = -xmsg[i];
    }
  });

  inp[1] = ring_sub(msg, inp[0]);

  NdArrayRef oup[2];
  utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> ctx) {
    int rank = ctx->Rank();
    auto conn = std::make_shared<Communicator>(ctx);
    auto base = std::make_shared<BasicOTProtocols>(conn);
    RingExtendProtocol prot(base);
    RingExtendProtocol::Meta meta;

    meta.sign = SignType::Unknown;
    meta.use_heuristic = true;
    meta.signed_arith = signed_arith;
    meta.src_ring = src_field;
    meta.dst_ring = dst_field;
    meta.src_width = SizeOf(src_field) * 8;
    meta.dst_width = SizeOf(dst_field) * 8;

    [[maybe_unused]] size_t b0 = ctx->GetStats()->sent_bytes;
    oup[rank] = prot.Compute(inp[rank], meta);
    [[maybe_unused]] size_t b1 = ctx->GetStats()->sent_bytes;
    SPDLOG_INFO("Heuristic Ext {} bits to {} bits sent {} bits per",
                meta.src_width, meta.dst_width, (b1 - b0) * 8. / shape.numel());
  });

  EXPECT_EQ(oup[0].shape(), oup[1].shape());
  auto expected = ring_add(inp[0], inp[1]);
  auto got = ring_add(oup[0], oup[1]);

  DISPATCH_ALL_FIELDS(src_field, "check", [&]() {
    using U0 = std::make_unsigned<ring2k_t>::type;
    using S0 = std::make_signed<ring2k_t>::type;
    NdArrayView<U0> expU(expected);
    NdArrayView<S0> expS(expected);
    DISPATCH_ALL_FIELDS(dst_field, "check", [&]() {
      using U1 = std::make_unsigned<ring2k_t>::type;
      using S1 = std::make_signed<ring2k_t>::type;

      NdArrayView<U1> gotU(got);
      NdArrayView<S1> gotS(got);
      if (signed_arith) {
        for (int64_t i = 0; i < shape.numel(); ++i) {
          EXPECT_EQ(static_cast<S1>(expS[i]), gotS[i]);
        }
      } else {
        for (int64_t i = 0; i < shape.numel(); ++i) {
          ASSERT_EQ(expU[i], gotU[i]);
        }
      }
    });
  });
}

TEST_P(RingExtendProtocolTest, BasicWithSpecificOutWidth) {
  size_t kWorldSize = 2;
  Shape shape = {kBenchSize};

  FieldType src_field = std::get<0>(GetParam());
  FieldType dst_field;
  if (src_field == FM32) {
    dst_field = FM64;
  } else {
    dst_field = FM128;
  }

  bool signed_arith = std::get<1>(GetParam());
  std::string sign_type_s = std::get<2>(GetParam());

  SignType sign_type;
  NdArrayRef inp[2];
  inp[0] = ring_rand(src_field, shape);

  if (sign_type_s == "Unknown") {
    inp[1] = ring_rand(src_field, shape);
    sign_type = SignType::Unknown;
  } else {
    auto msg = ring_rand(src_field, shape);
    DISPATCH_ALL_FIELDS(src_field, "setup", [&]() {
      auto xmsg = NdArrayView<ring2k_t>(msg);
      size_t bw = SizeOf(src_field) * 8;
      if (sign_type_s == "Positive") {
        auto msk = (static_cast<ring2k_t>(1) << (bw - 1)) - 1;
        pforeach(0, xmsg.numel(), [&](int64_t i) { xmsg[i] &= msk; });
        sign_type = SignType::Positive;
      } else {
        auto flag = static_cast<ring2k_t>(1) << (bw - 1);
        pforeach(0, xmsg.numel(), [&](int64_t i) { xmsg[i] |= flag; });
        sign_type = SignType::Negative;
      }
    });

    inp[1] = ring_sub(msg, inp[0]);
  }

  constexpr size_t kExtWidth = 20;
  size_t src_width;
  size_t dst_width;
  NdArrayRef oup[2];
  utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> ctx) {
    int rank = ctx->Rank();
    auto conn = std::make_shared<Communicator>(ctx);
    auto base = std::make_shared<BasicOTProtocols>(conn);
    RingExtendProtocol prot(base);
    RingExtendProtocol::Meta meta;

    meta.sign = sign_type;
    meta.use_heuristic = false;
    meta.signed_arith = signed_arith;
    meta.src_ring = src_field;
    meta.dst_ring = dst_field;
    // NOTE(lwj): only support field size for the src ring
    meta.src_width = SizeOf(src_field) * 8;
    meta.dst_width = meta.src_width + kExtWidth;

    [[maybe_unused]] size_t b0 = ctx->GetStats()->sent_bytes;
    oup[rank] = prot.Compute(inp[rank], meta);
    [[maybe_unused]] size_t b1 = ctx->GetStats()->sent_bytes;
    SPDLOG_INFO("Ext {} bits to {} bits sent {} bits per", meta.src_width,
                meta.dst_width, (b1 - b0) * 8. / shape.numel());

    src_width = meta.src_width;
    dst_width = meta.dst_width;
  });

  EXPECT_EQ(oup[0].shape(), oup[1].shape());
  auto expected = ring_add(inp[0], inp[1]);
  auto got = ring_add(oup[0], oup[1]);
  MaskItInplace(expected, src_width);
  MaskItInplace(got, dst_width);

  DISPATCH_ALL_FIELDS(src_field, "check", [&]() {
    using U0 = std::make_unsigned<ring2k_t>::type;
    using S0 = std::make_signed<ring2k_t>::type;
    NdArrayView<U0> expU(expected);
    NdArrayView<S0> expS(expected);
    DISPATCH_ALL_FIELDS(dst_field, "check", [&]() {
      using U1 = std::make_unsigned<ring2k_t>::type;
      using S1 = std::make_signed<ring2k_t>::type;

      NdArrayView<U1> gotU(got);
      NdArrayView<S1> gotS(got);
      if (signed_arith) {
        for (int64_t i = 0; i < shape.numel(); ++i) {
          EXPECT_EQ(ToSignType(expS[i], src_width),
                    ToSignType(gotS[i], dst_width));
        }
      } else {
        for (int64_t i = 0; i < shape.numel(); ++i) {
          ASSERT_EQ(expU[i], gotU[i]);
        }
      }
    });
  });
}

TEST_P(RingExtendProtocolTest, HeuristicWithSpecificOutWidth) {
  size_t kWorldSize = 2;
  Shape shape = {kBenchSize};

  FieldType src_field = std::get<0>(GetParam());
  FieldType dst_field;
  if (src_field == FM32) {
    dst_field = FM64;
  } else {
    dst_field = FM128;
  }

  bool signed_arith = std::get<1>(GetParam());
  std::string sign_type_s = std::get<2>(GetParam());

  NdArrayRef inp[2];
  inp[0] = ring_rand(src_field, shape);
  if (sign_type_s != "Unknown" or not signed_arith) {
    // 1. When sign type is already known, no need to use heuristic.
    // 2. Heuristic is only for signed arith
    return;
  }

  auto msg = ring_rand(src_field, shape);
  ring_rshift_(msg, RingExtendProtocol::kHeuristicBound);

  DISPATCH_ALL_FIELDS(src_field, "setup", [&]() {
    auto xmsg = NdArrayView<ring2k_t>(msg);
    // some are positive, some are negative
    for (int64_t i = 0; i < shape.numel(); i += 2) {
      xmsg[i] = -xmsg[i];
    }
  });

  inp[1] = ring_sub(msg, inp[0]);

  NdArrayRef oup[2];
  constexpr size_t kExtWidth = 20;
  size_t src_width;
  size_t dst_width;
  utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> ctx) {
    int rank = ctx->Rank();
    auto conn = std::make_shared<Communicator>(ctx);
    auto base = std::make_shared<BasicOTProtocols>(conn);
    RingExtendProtocol prot(base);
    RingExtendProtocol::Meta meta;

    meta.sign = SignType::Unknown;
    meta.use_heuristic = true;
    meta.signed_arith = signed_arith;
    meta.src_ring = src_field;
    meta.dst_ring = dst_field;
    meta.src_width = SizeOf(src_field) * 8;
    meta.dst_width = meta.src_width + kExtWidth;

    [[maybe_unused]] size_t b0 = ctx->GetStats()->sent_bytes;
    oup[rank] = prot.Compute(inp[rank], meta);
    [[maybe_unused]] size_t b1 = ctx->GetStats()->sent_bytes;
    SPDLOG_INFO("Heuristic Ext {} bits to {} bits sent {} bits per",
                meta.src_width, meta.dst_width, (b1 - b0) * 8. / shape.numel());

    src_width = meta.src_width;
    dst_width = meta.dst_width;
  });

  EXPECT_EQ(oup[0].shape(), oup[1].shape());
  auto expected = ring_add(inp[0], inp[1]);
  auto got = ring_add(oup[0], oup[1]);
  MaskItInplace(expected, src_width);
  MaskItInplace(got, dst_width);

  DISPATCH_ALL_FIELDS(src_field, "check", [&]() {
    using U0 = std::make_unsigned<ring2k_t>::type;
    using S0 = std::make_signed<ring2k_t>::type;
    NdArrayView<U0> expU(expected);
    NdArrayView<S0> expS(expected);
    DISPATCH_ALL_FIELDS(dst_field, "check", [&]() {
      using U1 = std::make_unsigned<ring2k_t>::type;
      using S1 = std::make_signed<ring2k_t>::type;

      NdArrayView<U1> gotU(got);
      NdArrayView<S1> gotS(got);
      if (signed_arith) {
        for (int64_t i = 0; i < shape.numel(); ++i) {
          EXPECT_EQ(ToSignType(expS[i], src_width),
                    ToSignType(gotS[i], dst_width));
        }
      } else {
        for (int64_t i = 0; i < shape.numel(); ++i) {
          ASSERT_EQ(expU[i], gotU[i]);
        }
      }
    });
  });
}

}  // namespace spu::mpc::cheetah
