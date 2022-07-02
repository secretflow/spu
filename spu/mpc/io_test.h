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

#include "spu/mpc/io_interface.h"

namespace spu::mpc {

using CreateIoFn =
    std::function<std::unique_ptr<IoInterface>(FieldType field, size_t npc)>;

// This test fixture defines the standard test cases for the io interface.
//
// Protocol implementers should instantiate this test when a new protocol is
// added. see
// [here](https://google.github.io/googletest/advanced.html#creating-value-parameterized-abstract-tests)
// for more details.
class IoTest : public ::testing::TestWithParam<
                   std::tuple<CreateIoFn, size_t, FieldType>> {};

}  // namespace spu::mpc
