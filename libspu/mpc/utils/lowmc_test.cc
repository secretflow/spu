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

#include "libspu/mpc/utils/lowmc.h"

#include "gtest/gtest.h"
#include "yacl/utils/elapsed_timer.h"

#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc {

TEST(LowMC, List) {
  uint128_t seed = 107;
  uint128_t key = 11;
  int64_t d = 20;  // data complexity
  int64_t n = 100;
  const Shape shape = {n, n};

  // 64-bits block
  {
    FieldType field = FM64;

    yacl::ElapsedTimer pack_timer;
    auto cipher = LowMC(field, seed, d, 128, true);
    double init_time = pack_timer.CountMs();

    cipher.set_key(key);

    auto values = ring_rand(field, shape);

    auto c = cipher.encrypt(values);

    auto p = cipher.decrypt(c);

    SPDLOG_INFO("{} blocks, {}-bits block, fill random {} ms", shape.numel(),
                64, init_time);

    EXPECT_TRUE(ring_all_equal(values, p));
  }

  // 128-bits block
  {
    FieldType field = FM128;

    yacl::ElapsedTimer pack_timer;
    auto cipher = LowMC(field, seed, d, 128, true);
    double init_time = pack_timer.CountMs();

    cipher.set_key(key);

    auto values = ring_rand(field, shape);

    auto c = cipher.encrypt(values);

    auto p = cipher.decrypt(c);

    SPDLOG_INFO("{} blocks, {}-bits block, fill random {} ms", shape.numel(),
                128, init_time);

    EXPECT_TRUE(ring_all_equal(values, p));
  }
}

}  // namespace spu::mpc
