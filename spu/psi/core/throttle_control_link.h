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

#pragma once

#include <future>
#include <numeric>

#include "yasl/base/exception.h"
#include "yasl/link/context.h"
#include "yasl/link/link.h"
#include "yasl/utils/serialize.h"

namespace spu::psi {

//
// KKRT(> 2^20 data size) in low bound case(10Mbps) will throw
// yasl::NetworkException
// class ThrottleControlSender and ThrottleControlReceiver
//  add ThrottleControl avoid exception
//  sender send msg --->
//                      recviver recv message
//                  <---         send counter
//  sender check windows, package number in queue exceed windows_size wait
//  because link should used spawn in multi thread
//  ThrottleControlSender and ThrottleControlReceiver should used in single
//  thread
//

class ThrottleControlSender {
 public:
  ThrottleControlSender(const std::shared_ptr<yasl::link::Context>& link_send,
                        const std::shared_ptr<yasl::link::Context>& link_recv,
                        size_t batch_num, size_t window_size = 32)
      : link_send_(link_send),
        link_recv_(link_recv),
        batch_num_(batch_num),
        window_size_(window_size) {}

  ~ThrottleControlSender();

  void SendAsync(yasl::ByteContainerView value, std::string_view tag);

  void StartRecvThread();
  void WaitRecvThread();

  void SetWindowThrottleTimeout(size_t window_throttle_timeout_ms) {
    this->window_throttle_timeout_ms_ = window_throttle_timeout_ms;
  }

  void SetWindowSize(size_t window_size) { this->window_size_ = window_size; }

  size_t sent_count() const { return batch_count_; }
  size_t ack_count() const { return finished_batch_count_; }

 private:
  // link_send_ used to send real messsage.
  std::shared_ptr<yasl::link::Context> link_send_;
  // link_recv_ used to receive peer's received count number.
  std::shared_ptr<yasl::link::Context> link_recv_;
  std::mutex window_mutex_;
  std::condition_variable window_cv_;
  // received count by peer
  size_t finished_batch_count_ = 0;
  // send count
  size_t batch_count_ = 0;
  // total batch number
  size_t batch_num_ = 0;

  size_t window_throttle_timeout_ms_ = 60 * 1000;
  //
  // window_size_ = 450 is base on test result on 10M bandwidth
  //   brpc default socket_max_unwritten_bytes value is 64M, reference
  //   brpc/socket.cpp:85:DEFINE_int64(socket_max_unwritten_bytes, 64 * 1024 *
  //   1024,
  // 1. correction send batch size is 52352B,
  //   set window_size_ < 64M /52352 = 1281, can avoid overcrowded exception
  // 2. 454 < window_size_< 1281,
  //    still has NetworkException, Reached timeout=20000ms,
  //    exception reason is batch in send queue reached timeout
  // 3. window_size = 450 < 454 test ok on 10M bandwidth
  //
  size_t window_size_ = 450;

  std::future<void> recv_thread_;

  void RecvCounterResponse();
};

class ThrottleControlReceiver {
 public:
  ThrottleControlReceiver(
      const std::shared_ptr<yasl::link::Context>& link_recv_msg,
      const std::shared_ptr<yasl::link::Context>& link_send_counter)
      : link_recv_msg_(link_recv_msg), link_send_counter_(link_send_counter) {}

  yasl::Buffer RecvMsgWithSendCounter();

 private:
  // link_recv_msg_ used to receive real messsage.
  std::shared_ptr<yasl::link::Context> link_recv_msg_;
  // link_send_counter_ used to send received message count.
  std::shared_ptr<yasl::link::Context> link_send_counter_;
  // received message count.
  size_t batch_count_ = 0;
};

}  // namespace spu::psi