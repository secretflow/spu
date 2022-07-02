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

#include <iostream>

#include "spdlog/spdlog.h"

#include "spu/psi/core/utils.h"

namespace spu::psi {

void ThrottleControlSender::SendAsync(yasl::ByteContainerView value,
                                      std::string_view tag) {
  link_send_->SendAsync(link_send_->NextRank(), value, tag);
  batch_count_ += 1;

  std::unique_lock<std::mutex> lock(window_mutex_);
  auto now = std::chrono::system_clock::now();

  YASL_ENFORCE(
      window_cv_.wait_until(
          lock, now + std::chrono::milliseconds(window_throttle_timeout_ms_),
          [&]() {
            return batch_count_ - finished_batch_count_ <= window_size_;
          }),
      "Timeout when waiting for the finished batch to catch up, "
      "batch_count={}, finished_batch_count={}",
      batch_count_, finished_batch_count_);
}

void ThrottleControlSender::RecvCounterResponse() {
  while (finished_batch_count_ < batch_num_) {
    const auto tag =
        fmt::format("BatchSend:RecvResponse:{}", finished_batch_count_);

    size_t batch_count =
        utils::DeserializeSize(link_recv_->Recv(link_recv_->NextRank(), tag));

    std::unique_lock<std::mutex> lock(window_mutex_);
    finished_batch_count_ = batch_count;

    window_cv_.notify_one();
  }
}

void ThrottleControlSender::StartRecvThread() {
  // recv counter response
  recv_thread_ = std::async([&] { return RecvCounterResponse(); });
}

void ThrottleControlSender::WaitRecvThread() { recv_thread_.get(); }

ThrottleControlSender::~ThrottleControlSender() = default;

yasl::Buffer ThrottleControlReceiver::RecvMsgWithSendCounter() {
  auto recv_buf = link_recv_msg_->Recv(
      link_recv_msg_->NextRank(),
      fmt::format("KKRT:PSI:ThrottleControlReceiver recv batch_count:{}",
                  batch_count_));
  batch_count_++;

  link_send_counter_->SendAsync(
      link_send_counter_->NextRank(), utils::SerializeSize(batch_count_),
      fmt::format("KKRT_PSI:ThrottleControlReceiver send reponse {}",
                  batch_count_));
  return recv_buf;
}

}  // namespace spu::psi