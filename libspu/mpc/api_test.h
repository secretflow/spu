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

#include <functional>

#include "gtest/gtest.h"
#include "yacl/link/link.h"

#include "libspu/mpc/api_test_params.h"

#include "libspu/spu.pb.h"

namespace spu::mpc::test {

// This test fixture defines the standard test cases for the compute interface.
//
// Protocol implementers should instantiate this test when a new protocol is
// added. see
// [here](https://google.github.io/googletest/advanced.html#creating-value-parameterized-abstract-tests)
// for more details.

class ApiTest : public ::testing::TestWithParam<OpTestParams> {};

}  // namespace spu::mpc::test
