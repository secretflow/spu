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

#include "emp-tool/io/io_channel.h"
#include "yasl/base/buffer.h"
#include "yasl/link/link.h"

namespace spu {

class CheetahIo : public emp::IOChannel<CheetahIo> {
 public:
  std::shared_ptr<yasl::link::Context> ctx_;

  const static uint64_t SEND_BUFFER_SIZE = 1024 * 1024;
  uint32_t send_op_;
  uint32_t recv_op_;

  std::vector<uint8_t> send_buffer_;
  uint64_t send_buffer_used_;

  yasl::Buffer recv_buffer_;
  uint64_t recv_buffer_used_;

  explicit CheetahIo(std::shared_ptr<yasl::link::Context> ctx);

  ~CheetahIo();

  void flush();

  void fill_recv();

  void send_data_internal(const void* data, int len);

  void recv_data_internal(void* data, int len);

  template <typename T>
  void send_data_partial(const T* data, int len, int bitlength);

  template <typename T>
  void recv_data_partial(T* data, int len, int bitlength);
};

}  // namespace spu
