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

#include "silent_ot_pack.h"

#include <memory>
#include <utility>

namespace spu {
SilentOTPack::SilentOTPack(int party, std::unique_ptr<IO> io) {
  party_ = party;
  io_ = std::move(io);
  ios_[0] = io_.get();
  silent_ot_ = std::make_unique<SilentOT>(party, 1, ios_.data(), false, true,
                                          party == emp::ALICE
                                              ? PRE_OT_DATA_REG_SEND_FILE_ALICE
                                              : PRE_OT_DATA_REG_RECV_FILE_BOB,
                                          false);
  silent_ot_reversed_ = std::make_unique<SilentOT>(
      3 - party, 1, ios_.data(), false, true,
      party == emp::ALICE ? PRE_OT_DATA_REG_RECV_FILE_ALICE
                          : PRE_OT_DATA_REG_SEND_FILE_BOB,
      false);

  for (int i = 0; i < KKOT_TYPES; i++) {
    kkot_[i] = std::make_unique<SilentOTN>(silent_ot_.get(), 1 << (i + 1));
  }
}

}  // namespace spu
