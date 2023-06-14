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

#include "libspu/mpc/cheetah/nonlinear/truncate_prot.h"

#include <random>

#include "gtest/gtest.h"

#include "libspu/core/xt_helper.h"
#include "libspu/mpc/cheetah/ot/basic_ot_prot.h"
#include "libspu/mpc/semi2k/type.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::mpc::cheetah {

class TruncateProtTest : public ::testing::TestWithParam<
                             std::tuple<FieldType, bool, std::string>> {
  void SetUp() override {}
};

INSTANTIATE_TEST_SUITE_P(
    Cheetah, TruncateProtTest,
    testing::Combine(testing::Values(FieldType::FM32, FieldType::FM64,
                                     FieldType::FM128),
                     testing::Values(true, false),
                     testing::Values("Unknown", "Zero", "One")),
    [](const testing::TestParamInfo<TruncateProtTest::ParamType> &p) {
      return fmt::format("{}{}MSB{}", std::get<0>(p.param),
                         std::get<1>(p.param) ? "Signed" : "Unsigned",
                         std::get<2>(p.param));
    });

template <typename T>
bool SignBit(T x) {
  using uT = typename std::make_unsigned<T>::type;
  return (static_cast<uT>(x) >> (8 * sizeof(T) - 1)) & 1;
}

TEST_P(TruncateProtTest, Basic) {
  size_t kWorldSize = 2;
  size_t n = 1024;
  size_t shift = 13;
  FieldType field = std::get<0>(GetParam());
  bool signed_arith = std::get<1>(GetParam());
  std::string msb = std::get<2>(GetParam());
  TruncateProtocol::MSB_st msb_t;

  ArrayRef inp[2];
  inp[0] = ring_rand(field, n);

  if (msb == "Unknown") {
    inp[1] = ring_rand(field, n);
    msb_t = TruncateProtocol::MSB_st::unknown;
  } else {
    auto msg = ring_rand(field, n);
    DISPATCH_ALL_FIELDS(field, "", [&]() {
      auto xmsg = xt_mutable_adapt<ring2k_t>(msg);
      size_t bw = SizeOf(field) * 8;
      if (msb == "Zero") {
        ring2k_t mask = (static_cast<ring2k_t>(1) << (bw - 1)) - 1;
        xmsg &= mask;
        msb_t = TruncateProtocol::MSB_st::zero;
      } else {
        ring2k_t mask = (static_cast<ring2k_t>(1) << (bw - 1));
        xmsg |= mask;
        msb_t = TruncateProtocol::MSB_st::one;
      }
    });

    inp[1] = ring_sub(msg, inp[0]);
  }

  ArrayRef oup[2];
  utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> ctx) {
    int rank = ctx->Rank();
    auto conn = std::make_shared<Communicator>(ctx);
    auto base = std::make_shared<BasicOTProtocols>(conn);
    TruncateProtocol trunc_prot(base);
    TruncateProtocol::Meta meta;
    meta.msb = msb_t;
    meta.signed_arith = signed_arith;
    meta.shift_bits = shift;
    oup[rank] = trunc_prot.Compute(inp[rank], meta);
  });

  DISPATCH_ALL_FIELDS(field, "", [&]() {
    using signed_t = std::make_signed<ring2k_t>::type;
    using usigned_t = std::make_unsigned<ring2k_t>::type;

    if (signed_arith) {
      auto xout0 = xt_adapt<signed_t>(oup[0]);
      auto xout1 = xt_adapt<signed_t>(oup[1]);
      auto xinp0 = xt_adapt<signed_t>(inp[0]);
      auto xinp1 = xt_adapt<signed_t>(inp[1]);

      for (size_t i = 0; i < n; ++i) {
        signed_t in = xinp0[i] + xinp1[i];
        signed_t expected = in >> shift;
        if (msb_t != TruncateProtocol::MSB_st::unknown) {
          ASSERT_EQ(SignBit<signed_t>(in),
                    msb_t == TruncateProtocol::MSB_st::one);
        }
        signed_t got = xout0[i] + xout1[i];
        EXPECT_NEAR(expected, got, 1);
      }
    } else {
      auto xout0 = xt_adapt<usigned_t>(oup[0]);
      auto xout1 = xt_adapt<usigned_t>(oup[1]);
      auto xinp0 = xt_adapt<usigned_t>(inp[0]);
      auto xinp1 = xt_adapt<usigned_t>(inp[1]);

      for (size_t i = 0; i < n; ++i) {
        usigned_t in = xinp0[i] + xinp1[i];
        usigned_t expected = (in) >> shift;
        if (msb_t != TruncateProtocol::MSB_st::unknown) {
          ASSERT_EQ(SignBit<usigned_t>(in),
                    msb_t == TruncateProtocol::MSB_st::one);
        }
        usigned_t got = xout0[i] + xout1[i];
        ASSERT_NEAR(expected, got, 1);
      }
    }
  });
}

}  // namespace spu::mpc::cheetah
