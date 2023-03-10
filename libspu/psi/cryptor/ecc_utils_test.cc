// Copyright 2023 Ant Group Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "libspu/psi/cryptor/ecc_utils.h"

#include "gtest/gtest.h"

namespace spu::psi::test {

TEST(EcPointStTest, HashToCurveWorks) {
  EcGroupSt curve(NID_sm2);

  for (int i = 0; i < 100; ++i) {
    EcPointSt point = EcPointSt::CreateEcPointByHashToCurve(
        fmt::format("id{}", i).c_str(), curve);

    BigNumSt x, y;
    ASSERT_EQ(EC_POINT_get_affine_coordinates(curve.get(), point.get(), x.get(),
                                              y.get(), nullptr),
              1);
    char *str = BN_bn2hex(x.get());
    fmt::print("\"id{}\" -> point_x:{}\n", i, str);
    OPENSSL_free(str);
  }
}

}  // namespace spu::psi::test
