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

#include "libspu/mpc/aby3/protocol.h"

#include "libspu/mpc/ab_api.h"
#include "libspu/mpc/ab_api_test.h"
#include "libspu/mpc/api.h"
#include "libspu/mpc/api_test.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::mpc::test {
namespace {

RuntimeConfig makeConfig(FieldType field) {
  RuntimeConfig conf;
  conf.protocol = ProtocolKind::ABY3;
  conf.field = field;
  return conf;
}

}  // namespace

INSTANTIATE_TEST_SUITE_P(
    Aby3, ApiTest,
    testing::Combine(testing::Values(makeAby3Protocol),              //
                     testing::Values(makeConfig(FieldType::FM32),    //
                                     makeConfig(FieldType::FM64),    //
                                     makeConfig(FieldType::FM128)),  //
                     testing::Values(3)),                            //
    [](const testing::TestParamInfo<ApiTest::ParamType>& p) {
      return fmt::format("{}x{}", std::get<1>(p.param).field,
                         std::get<2>(p.param));
    });

INSTANTIATE_TEST_SUITE_P(
    Aby3, ArithmeticTest,
    testing::Combine(testing::Values(makeAby3Protocol),              //
                     testing::Values(makeConfig(FieldType::FM32),    //
                                     makeConfig(FieldType::FM64),    //
                                     makeConfig(FieldType::FM128)),  //
                     testing::Values(3)),                            //
    [](const testing::TestParamInfo<ArithmeticTest::ParamType>& p) {
      return fmt::format("{}x{}", std::get<1>(p.param).field,
                         std::get<2>(p.param));
    });

INSTANTIATE_TEST_SUITE_P(
    Aby3, BooleanTest,
    testing::Combine(testing::Values(makeAby3Protocol),              //
                     testing::Values(makeConfig(FieldType::FM32),    //
                                     makeConfig(FieldType::FM64),    //
                                     makeConfig(FieldType::FM128)),  //
                     testing::Values(3)),                            //
    [](const testing::TestParamInfo<BooleanTest::ParamType>& p) {
      return fmt::format("{}x{}", std::get<1>(p.param).field,
                         std::get<2>(p.param));
    });

INSTANTIATE_TEST_SUITE_P(
    Aby3, ConversionTest,
    testing::Combine(testing::Values(makeAby3Protocol),              //
                     testing::Values(makeConfig(FieldType::FM32),    //
                                     makeConfig(FieldType::FM64),    //
                                     makeConfig(FieldType::FM128)),  //
                     testing::Values(3)),                            //
    [](const testing::TestParamInfo<BooleanTest::ParamType>& p) {
      return fmt::format("{}x{}", std::get<1>(p.param).field,
                         std::get<2>(p.param));
    });

namespace {
std::string getSignName(SignType sign) {
  if (sign == SignType::Unknown) {
    return "Unknown";
  } else if (sign == SignType::Positive) {
    return "Positive";
  } else if (sign == SignType::Negative) {
    return "Negative";
  } else {
    SPU_THROW("should not be here.");
  }
}
}  // namespace

// TODO: implement and test general bw truncate
class TruncateTest
    : public ::testing::TestWithParam<std::tuple<FieldType, SignType, size_t>> {
  void SetUp() override {}
};

INSTANTIATE_TEST_SUITE_P(
    Aby3, TruncateTest,
    testing::Combine(testing::Values(FieldType::FM32, FieldType::FM64,
                                     FieldType::FM128),
                     testing::Values(SignType::Unknown, SignType::Positive,
                                     SignType::Negative),
                     testing::Values(18)),
    [](const testing::TestParamInfo<TruncateTest::ParamType>& p) {
      return fmt::format("{}x{}x{}", std::get<0>(p.param),
                         getSignName(std::get<1>(p.param)),
                         std::get<2>(p.param));
    });

TEST_P(TruncateTest, Works) {
  size_t npc = 3;
  int64_t n = 1024;

  auto field = std::get<0>(GetParam());
  auto sign = std::get<1>(GetParam());
  auto shift = std::get<2>(GetParam());

  const auto bw = SizeOf(field) * 8;

  auto pub = ring_rand(field, {n});

  // SPU encoding: must be 00... or 11...
  ring_arshift_(pub, {1});
  const auto ty = makeType<Pub2kTy>(field);
  // adjust sign if necessary
  DISPATCH_ALL_FIELDS(field, [&]() {
    NdArrayView<ring2k_t> _pub(pub);
    if (sign == SignType::Positive) {
      // 0011 ... 1111
      ring2k_t mask = (static_cast<ring2k_t>(1) << (bw - 2)) - 1;
      pforeach(0, n, [&](int64_t i) {  //
        _pub[i] &= mask;
      });
    } else if (sign == SignType::Negative) {
      // 1100 ... 0000
      ring2k_t mask = (static_cast<ring2k_t>(1) << (bw - 2)) |
                      (static_cast<ring2k_t>(1) << (bw - 1));
      pforeach(0, n, [&](int64_t i) {  //
        _pub[i] |= mask;
      });
    }
  });

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lcxt) {
    auto obj = makeAby3Protocol(makeConfig(field), lcxt);

    auto p0 = WrapValue(pub.as(ty));

    auto a0 = p2a(obj.get(), p0);

    auto b0 = lcxt->GetStats()->sent_bytes.load();
    auto s0 = lcxt->GetStats()->sent_actions.load();
    // truncate
    auto a1 = trunc_a(obj.get(), a0, shift, sign);

    auto b1 = lcxt->GetStats()->sent_bytes.load();
    auto s1 = lcxt->GetStats()->sent_actions.load();

    SPDLOG_INFO(
        "Rank {}, sign {}, truncate {} bits shares by {} bits, sent {} "
        "bytes per sample, {} actions total.",
        lcxt->Rank(), getSignName(sign), bw, shift, (b1 - b0) * 1.0 / n,
        s1 - s0);

    // check
    auto r_a = a2p(obj.get(), a1);
    auto r_p = arshift_p(obj.get(), p0, {static_cast<int64_t>(shift)});

    if (lcxt->Rank() == 0) {
      EXPECT_EQ(r_a.shape(), r_p.shape());
      // prob truncation admits 1 bit error at most
      EXPECT_TRUE(ring_all_equal(r_a.data(), r_p.data(), 1));
    }
  });
}

}  // namespace spu::mpc::test
