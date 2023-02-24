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

#include "libspu/psi/utils/emp_io_adapter.h"

#include <future>
#include <thread>

#include "gtest/gtest.h"
#include "yacl/link/test_util.h"

namespace spu {

TEST(EmpIoAdapterTest, Test) {
  const int kWorldSize = 2;
  auto contexts = yacl::link::test::SetupWorld(kWorldSize);

  std::future<void> player1 = std::async([&] {
    char msg[100];
    EmpIoAdapter io(contexts[0]);
    io.send_data("hello", 5);
    io.send_data(" world", 6);

    io.recv_data(msg, 7);

    std::cout << "player1 receive: " << msg << std::endl;
  });

  std::future<void> player2 = std::async([&] {
    char msg[100];
    EmpIoAdapter io(contexts[1]);
    io.recv_data(msg, 11);

    io.send_data("goodbye", 7);

    std::cout << "player2 receive: " << msg << std::endl;
  });

  player1.get();
  player2.get();
}

TEST(EmpIoAdapterTest, TestPartial) {
  const int kWorldSize = 2;
  auto contexts = yacl::link::test::SetupWorld(kWorldSize);

  std::future<void> player1 = std::async([&] {
    uint64_t a[4] = {0x284, 0xf3a, 0x97e4, 0x8fa};
    EmpIoAdapter io(contexts[0]);
    io.send_data_partial(a, 4, 12);

    std::cout << "player1 sends: " << a[0] << ", " << a[1] << ", "
              << (a[2] & ((1 << 12) - 1)) << ", " << a[3] << std::endl;
  });

  std::future<void> player2 = std::async([&] {
    uint64_t a[4];
    EmpIoAdapter io(contexts[1]);
    io.recv_data_partial(a, 4, 12);
    std::cout << "player2 receives: " << a[0] << ", " << a[1] << ", " << a[2]
              << ", " << a[3] << std::endl;
  });

  player1.get();
  player2.get();
}

}  // namespace spu
