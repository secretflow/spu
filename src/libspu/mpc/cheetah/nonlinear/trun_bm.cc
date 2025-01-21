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

#include <chrono>

#include "libspu/mpc/cheetah/nonlinear/truncate_prot.h"

#include "gtest/gtest.h"

#include "libspu/mpc/cheetah/ot/basic_ot_prot.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"

#define GetComm(tag) sctx_##tag.get()->getState<Communicator>()->getStats()

namespace spu::mpc::cheetah {

class AlkaidTruncateProtTest : public ::testing::TestWithParam<
                             std::tuple<FieldType, bool, std::string>> {
  void SetUp() override {}
};

INSTANTIATE_TEST_SUITE_P(
    Cheetah, AlkaidTruncateProtTest,
    testing::Combine(testing::Values(FieldType::FM64),
                     testing::Values(true, false),
                     testing::Values("Unknown")),
    [](const testing::TestParamInfo<AlkaidTruncateProtTest::ParamType> &p) {
      return fmt::format("{}{}MSB{}", std::get<0>(p.param),
                         std::get<1>(p.param) ? "Signed" : "Unsigned",
                         std::get<2>(p.param));
    });

template <typename T>
bool SignBit(T x) {
  using uT = typename std::make_unsigned<T>::type;
  return (static_cast<uT>(x) >> (8 * sizeof(T) - 1)) & 1;
}

TEST_P(AlkaidTruncateProtTest, Basic) {
  size_t kWorldSize = 2;
  int64_t n = static_cast<int64_t>(pow(10, 7));
  size_t shift = 18;
  FieldType field = std::get<0>(GetParam());
  bool signed_arith = std::get<1>(GetParam());
  std::string msb = std::get<2>(GetParam());
  SignType sign;

  NdArrayRef inp[2];
  inp[0] = ring_rand(field, {n});

  if (msb == "Unknown") {
    inp[1] = ring_rand(field, {n});
    sign = SignType::Unknown;
  } else {
    return;
  }

  NdArrayRef oup[2];
  utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> ctx) {
    int rank = ctx->Rank();
    auto conn = std::make_shared<Communicator>(ctx);
    auto base = std::make_shared<BasicOTProtocols>(
        conn, CheetahOtKind::YACL_Softspoken);
    TruncateProtocol trunc_prot(base);
    TruncateProtocol::Meta meta;
    meta.sign = sign;
    meta.signed_arith = signed_arith;
    meta.shift_bits = shift;
    meta.use_heuristic = false;

    [[maybe_unused]] auto b0s = ctx->GetStats()->sent_bytes.load();
    [[maybe_unused]] auto s0s = ctx->GetStats()->sent_actions.load();
    [[maybe_unused]] auto b0r = ctx->GetStats()->recv_bytes.load();
    [[maybe_unused]] auto s0r = ctx->GetStats()->recv_actions.load();

    auto start_time = std::chrono::high_resolution_clock::now();

    oup[rank] = trunc_prot.Compute(inp[rank], meta);

    auto end_time = std::chrono::high_resolution_clock::now();

    [[maybe_unused]] auto b1s = ctx->GetStats()->sent_bytes.load();
    [[maybe_unused]] auto s1s = ctx->GetStats()->sent_actions.load();
    [[maybe_unused]] auto b1r = ctx->GetStats()->recv_bytes.load();
    [[maybe_unused]] auto s1r = ctx->GetStats()->recv_actions.load();

    SPDLOG_DEBUG("Truncate {} bits share by {} bits {} bits each #sent {}",
                 SizeOf(field) * 8, meta.shift_bits,
                 (b1s - b0s) * 8. / inp[0].numel(), (s1s - s0s));

    if (rank == 0)
    std::cout << "Party " << rank
              << " truncate a tensor of length " << n << " with " << SizeOf(field) * 8 << " bits by "
              << meta.shift_bits << " bits " 
              << "sending " << (b1s - b0s) / 1024. << " KB in " << (s1s - s0s) << " rounds and "
              << "receiving " << (b1r - b0r) / 1024. << " KB in " << (s1r - s0r) << " rounds. " 
              << "\nTime: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << " ms."
              << std::endl;
  });

  EXPECT_EQ(oup[0].shape(), oup[1].shape());

  DISPATCH_ALL_FIELDS(field, [&]() {
    using signed_t = std::make_signed<ring2k_t>::type;
    using usigned_t = std::make_unsigned<ring2k_t>::type;

    if (signed_arith) {
      auto xout0 = NdArrayView<signed_t>(oup[0]);
      auto xout1 = NdArrayView<signed_t>(oup[1]);
      auto xinp0 = absl::MakeSpan(&inp[0].at<signed_t>(0), inp[0].numel());
      auto xinp1 = absl::MakeSpan(&inp[1].at<signed_t>(0), inp[1].numel());

      for (int64_t i = 0; i < n; ++i) {
        signed_t in = xinp0[i] + xinp1[i];
        signed_t expected = in >> shift;
        if (sign != SignType::Unknown) {
          ASSERT_EQ(SignBit<signed_t>(in), sign == SignType::Negative);
        }
        signed_t got = xout0[i] + xout1[i];
        EXPECT_NEAR(expected, got, 1);
      }
    } else {
      auto xout0 = NdArrayView<usigned_t>(oup[0]);
      auto xout1 = NdArrayView<usigned_t>(oup[1]);
      auto xinp0 = absl::MakeSpan(&inp[0].at<usigned_t>(0), inp[0].numel());
      auto xinp1 = absl::MakeSpan(&inp[1].at<usigned_t>(0), inp[1].numel());

      for (int64_t i = 0; i < n; ++i) {
        usigned_t in = xinp0[i] + xinp1[i];
        usigned_t expected = (in) >> shift;
        if (sign != SignType::Unknown) {
          ASSERT_EQ(SignBit<usigned_t>(in), sign == SignType::Negative);
        }
        usigned_t got = xout0[i] + xout1[i];
        ASSERT_NEAR(expected, got, 1);
      }
    }
  });
}

}  // namespace spu::mpc::cheetah
