// Copyright 2023 Ant Group Co., Ltd.
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

#include "libspu/kernel/hal/ring.h"

#include "gtest/gtest.h"
#include "xtensor/xarray.hpp"

#include "libspu/kernel/test_util.h"

namespace spu::kernel::hal {

TEST(RingTest, _bit_parity) {
  auto ctx = test::makeSPUContext();
  std::vector<std::pair<uint64_t, int>> para = {
      {0x0505, 0},
      {0x10505, 1},
      {0x050500000505, 0},
      {0x1505050505050505, 1},
  };
  for (auto [init, exp] : para) {
    auto input = _make_p(&ctx, init, {3});

    auto output = _bit_parity(&ctx, input, 64);

    output.setDtype(DT_U64);
    auto p_ret = hal::dump_public_as<int64_t>(&ctx, output);
    xt::xarray<int64_t> expected{exp, exp, exp};
    EXPECT_EQ(p_ret, expected);
  }
}

TEST(RingTest, _popcount) {
  auto ctx = test::makeSPUContext();
  std::vector<std::pair<uint64_t, int>> para = {
      {0x0505, 4},
      {0x10505, 5},
      {0x05050505, 8},
      {0x0500000000000005, 4},
  };
  for (auto [init, exp] : para) {
    auto input = _make_p(&ctx, init, {3});

    auto output = _popcount(&ctx, input, 64);

    output.setDtype(DT_U64);
    auto p_ret = hal::dump_public_as<int64_t>(&ctx, output);
    xt::xarray<int64_t> expected{exp, exp, exp};
    EXPECT_EQ(p_ret, expected);
  }
}
}  // namespace spu::kernel::hal
