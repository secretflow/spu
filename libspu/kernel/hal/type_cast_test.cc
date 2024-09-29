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

#include "libspu/kernel/hal/type_cast.h"

#include <cstdint>

#include "gtest/gtest.h"
#include "xtensor/xio.hpp"

#include "libspu/kernel/hal/constants.h"
#include "libspu/kernel/test_util.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::kernel::hal {
namespace {

TEST(TypeCastTest, boolean) {
  mpc::utils::simulate(
      3, [&](const std::shared_ptr<yacl::link::Context> &lctx) {
        SPUContext sctx = test::makeSPUContext(REF2K, 64, lctx);
        MemRef pa = constant(&sctx, true);
        MemRef sa = seal(&sctx, pa);
        EXPECT_TRUE(sa.isSecret());
      });
}

}  // namespace
}  // namespace spu::kernel::hal
