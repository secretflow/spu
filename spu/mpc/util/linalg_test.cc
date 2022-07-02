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

#include "spu/mpc/util/linalg.h"

#include <vector>

#include "gtest/gtest.h"

namespace spu::mpc::linalg {

TEST(LinalgTest, MatMulBasic) {
  std::vector<float> A = {1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15};  // 4x3
  std::vector<float> B = {1, 2, 3, 4, 5, 6};                         // 3x2
  std::vector<float> C(8);                                           // 4x2

  matmul(4, 2, 3, A.data(), 3, 1, B.data(), 2, 1, C.data(), 2, 1);

  std::vector<float> expected = {22.f, 28.f,  58.f,  76.f,
                                 94.f, 124.f, 130.f, 172.f};

  EXPECT_EQ(C, expected);
}

TEST(LinalgTest, MatMulStrides) {
  std::vector<float> A = {1,  100, 2,  100, 3,  100,   //
                          5,  100, 6,  100, 7,  100,   //
                          9,  100, 10, 100, 11, 100,   //
                          13, 100, 14, 100, 15, 100};  // 4x3

  std::vector<float> B = {1, 100, 2, 100,   //
                          3, 100, 4, 100,   //
                          5, 100, 6, 100};  // 3x2
  std::vector<float> C(8);                  // 4x2

  matmul(4, 2, 3, A.data(), 6, 2, B.data(), 4, 2, C.data(), 2, 1);

  std::vector<float> expected = {22.f, 28.f,  58.f,  76.f,
                                 94.f, 124.f, 130.f, 172.f};

  EXPECT_EQ(C, expected);
}

}  // namespace spu::mpc::linalg
