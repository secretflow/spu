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

#include "spu/psi/core/throttle_control_link.h"

#include <future>
#include <iostream>

#include "gtest/gtest.h"
#include "yasl/base/exception.h"
#include "yasl/link/test_util.h"

namespace spu::psi {

struct TestParams {
  size_t batch_size;
  size_t buf_size;
};

class ThrottleControlTest : public ::testing::TestWithParam<TestParams> {};

TEST_P(ThrottleControlTest, Works) {
  const int kWorldSize = 2;
  auto contexts = yasl::link::test::SetupWorld(kWorldSize);

  auto param = GetParam();
  std::string send_data(param.buf_size, '\0');

  std::future<void> f_send = std::async([&] {
    ThrottleControlSender throttle_control_sender(
        contexts[0], contexts[0]->Spawn(), param.batch_size);

    throttle_control_sender.StartRecvThread();

    for (size_t i = 0; i < param.batch_size; i++) {
      throttle_control_sender.SendAsync(
          send_data, fmt::format("KKRT_PSI:sendBatch:{}", i));
    }
    throttle_control_sender.WaitRecvThread();
  });

  std::future<void> f_recv = std::async([&] {
    ThrottleControlReceiver throttle_control_receiver(contexts[1],
                                                      contexts[1]->Spawn());
    for (size_t i = 0; i < param.batch_size; i++) {
      auto buf = throttle_control_receiver.RecvMsgWithSendCounter();
      std::string recv_data(buf.data<char>(), buf.size());
      EXPECT_EQ(recv_data.size(), param.buf_size);
    }
  });

  f_send.get();
  f_recv.get();
}

INSTANTIATE_TEST_SUITE_P(Works_Instances, ThrottleControlTest,
                         testing::Values(TestParams{1024, 5120},     //
                                         TestParams{1024, 10240},    //
                                         TestParams{5120, 5120},     //
                                         TestParams{5120, 10240},    //
                                         TestParams{10240, 5120},    //
                                         TestParams{10240, 10240},   //
                                         TestParams{102400, 5120},   //
                                         TestParams{102400, 10240})  //
);

}  // namespace spu::psi