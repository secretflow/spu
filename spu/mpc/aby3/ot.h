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

#include "yasl/link/link.h"

#include "spu/core/array_ref.h"
#include "spu/core/type_util.h"
#include "spu/mpc/common/prg_state.h"
#include "spu/mpc/util/communicator.h"

namespace spu::mpc::aby3 {

// Referrence:
// 5.4.1 Semi-honest Security
// https://eprint.iacr.org/2018/403.pdf
//
// Latency: 1
class Ot3 {
 public:
  struct RoleRanks {
    size_t sender;
    size_t receiver;
    size_t helper;
  };

 protected:
  // public information of this blocking block.
  const FieldType field_;  // each msg has SizeOf(field_) bits
  const int64_t numel_;    // total num of msgs
  const RoleRanks roles_;

  // state information
  Communicator* const comm_;
  PrgState* const prg_state_;

  //
  std::pair<ArrayRef, ArrayRef> genMasks();

 public:
  explicit Ot3(FieldType field, int64_t numel, const RoleRanks& roles,
               Communicator* comm, PrgState* prg_state);

  void send(ArrayRef m0, ArrayRef m1);

  ArrayRef recv(const std::vector<uint8_t>& choices);

  void help(const std::vector<uint8_t>& choices);
};

}  // namespace spu::mpc::aby3
