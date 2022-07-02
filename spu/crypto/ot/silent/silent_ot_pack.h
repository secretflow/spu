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

#include "cheetah_io_channel.h"
#include "silent_ot.h"

#define PRE_OT_DATA_REG_SEND_FILE_ALICE "pre_ot_data_reg_send_alice"
#define PRE_OT_DATA_REG_SEND_FILE_BOB "pre_ot_data_reg_send_bob"
#define PRE_OT_DATA_REG_RECV_FILE_ALICE "pre_ot_data_reg_recv_alice"
#define PRE_OT_DATA_REG_RECV_FILE_BOB "pre_ot_data_reg_recv_bob"

#define KKOT_TYPES 8

namespace spu {
using IO = CheetahIo;

class SilentOTPack {
 public:
  int party_;
  std::unique_ptr<IO> io_ = nullptr;
  std::array<IO*, 1> ios_;
  std::unique_ptr<SilentOT> silent_ot_ = nullptr;
  std::unique_ptr<SilentOT> silent_ot_reversed_ = nullptr;

  std::array<std::unique_ptr<SilentOTN>, KKOT_TYPES> kkot_;

  SilentOTPack(int party, std::unique_ptr<IO> io);
  ~SilentOTPack() = default;
};

}  // namespace spu
