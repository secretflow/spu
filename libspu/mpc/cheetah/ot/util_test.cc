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

#include "libspu/mpc/cheetah/ot/util.h"

#include <random>

#include "gtest/gtest.h"

#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::cheetah::test {

class UtilTest : public ::testing::TestWithParam<FieldType> {
  void SetUp() override {}
};

INSTANTIATE_TEST_SUITE_P(
    Cheetah, UtilTest,
    testing::Values(FieldType::FM32, FieldType::FM64, FieldType::FM128),
    [](const testing::TestParamInfo<UtilTest::ParamType> &p) {
      return fmt::format("{}", p.param);
    });

TEST_P(UtilTest, ZipArray) {
  const size_t n = 20;
  const auto field = GetParam();
  const size_t elsze = SizeOf(field);

  auto unzip = ring_zeros(field, n);

  DISPATCH_ALL_FIELDS(field, "", [&]() {
    for (size_t bw : {1, 2, 4, 7, 15, 16}) {
      size_t pack_load = elsze * 8 / bw;
      auto zip = ring_zeros(field, (n + pack_load - 1) / pack_load);
      auto array = ring_rand(field, n);
      auto inp = xt_mutable_adapt<ring2k_t>(array);
      auto mask = makeBitsMask<ring2k_t>(bw);
      inp &= mask;

      auto _zip = xt_mutable_adapt<ring2k_t>(zip);
      auto _unzip = xt_mutable_adapt<ring2k_t>(unzip);
      size_t zip_sze = ZipArray<ring2k_t>({inp.data(), inp.size()}, bw,
                                          {_zip.data(), _zip.size()});

      UnzipArray<ring2k_t>({_zip.data(), zip_sze}, bw,
                           {_unzip.data(), _unzip.size()});

      for (size_t i = 0; i < n; ++i) {
        EXPECT_EQ(inp[i], _unzip[i]);
      }
    }
  });
}

TEST_P(UtilTest, PackU8Array) {
  const size_t num_bytes = 223;
  const auto field = GetParam();
  const size_t elsze = SizeOf(field);

  std::uniform_int_distribution<uint8_t> uniform(0, -1);
  std::default_random_engine rdv;
  std::vector<uint8_t> u8array(num_bytes);
  std::generate_n(u8array.data(), u8array.size(),
                  [&]() { return uniform(rdv); });

  auto packed = ring_zeros(field, (num_bytes + elsze - 1) / elsze);

  DISPATCH_ALL_FIELDS(field, "", [&]() {
    auto xp = xt_mutable_adapt<ring2k_t>(packed);
    PackU8Array<ring2k_t>(absl::MakeSpan(u8array), {xp.data(), xp.size()});
    std::vector<uint8_t> _u8(num_bytes, -1);
    UnpackU8Array<ring2k_t>({xp.data(), xp.size()}, absl::MakeSpan(_u8));

    EXPECT_TRUE(std::memcmp(_u8.data(), u8array.data(), num_bytes) == 0);
  });
}

}  // namespace spu::mpc::cheetah::test
