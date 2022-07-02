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

#include "spu/mpc/aby3/ot.h"

#include "spu/mpc/util/ring_ops.h"

namespace spu::mpc::aby3 {
namespace {

// test a in front of b
inline bool inFrontOf(size_t a, size_t b) { return (a + 1) % 3 == b; }

}  // namespace

Ot3::Ot3(FieldType field, int64_t numel, const RoleRanks& roles,
         Communicator* comm, PrgState* prg_state)
    : field_(field),
      numel_(numel),
      roles_(roles),
      comm_(comm),
      prg_state_(prg_state) {}

std::pair<ArrayRef, ArrayRef> Ot3::genMasks() {
  ArrayRef w0, w1, _;

  if (comm_->getRank() == roles_.sender) {
    if (inFrontOf(roles_.sender, roles_.helper)) {
      std::tie(_, w0) = prg_state_->genPrssPair(field_, numel_, true, false);
      std::tie(_, w1) = prg_state_->genPrssPair(field_, numel_, true, false);
    } else {
      YASL_ENFORCE(inFrontOf(roles_.helper, roles_.sender));
      std::tie(w0, _) = prg_state_->genPrssPair(field_, numel_, false, true);
      std::tie(w1, _) = prg_state_->genPrssPair(field_, numel_, false, true);
    }
  } else if (comm_->getRank() == roles_.helper) {
    if (inFrontOf(roles_.sender, roles_.helper)) {
      std::tie(w0, _) = prg_state_->genPrssPair(field_, numel_, false, true);
      std::tie(w1, _) = prg_state_->genPrssPair(field_, numel_, false, true);
    } else {
      YASL_ENFORCE(inFrontOf(roles_.helper, roles_.sender));
      std::tie(_, w0) = prg_state_->genPrssPair(field_, numel_, true, false);
      std::tie(_, w1) = prg_state_->genPrssPair(field_, numel_, true, false);
    }
  } else {
    YASL_ENFORCE(comm_->getRank() == roles_.receiver);
    prg_state_->genPrssPair(field_, numel_, true, true);
    prg_state_->genPrssPair(field_, numel_, true, true);
  }

  return {w0, w1};
}

void Ot3::send(ArrayRef m0, ArrayRef m1) {
  // sanity check.
  YASL_ENFORCE(comm_->getRank() == roles_.sender);
  YASL_ENFORCE(m0.numel() == numel_);
  YASL_ENFORCE(m1.numel() == numel_);

  // generate masks
  ArrayRef w0, w1;
  std::tie(w0, w1) = genMasks();
  YASL_ENFORCE(w0.numel() == numel_);
  YASL_ENFORCE(w1.numel() == numel_);

  // mask the values
  auto masked_m0 = ring_xor(m0, w0);
  auto masked_m1 = ring_xor(m1, w1);

  comm_->sendAsync(roles_.receiver, masked_m0, "m0");
  comm_->sendAsync(roles_.receiver, masked_m1, "m1");
}

ArrayRef Ot3::recv(const std::vector<uint8_t>& choices) {
  // sanity check.
  YASL_ENFORCE(comm_->getRank() == roles_.receiver);
  YASL_ENFORCE(choices.size() == static_cast<size_t>(numel_));

  // consume prss to keep state synced.
  genMasks();

  const auto ty = makeType<RingTy>(field_);

  // get masked messages from sender.
  auto m0 = comm_->recv(roles_.sender, ty, "m0");
  auto m1 = comm_->recv(roles_.sender, ty, "m1");
  // get chosen masks
  auto wc = comm_->recv(roles_.helper, ty, "wc");

  YASL_ENFORCE(m0.numel() == static_cast<int64_t>(choices.size()));

  // reconstruct mc
  ring_xor_(m0, wc);
  ring_xor_(m1, wc);
  ArrayRef mc = ring_select(choices, m0, m1);

  return mc;
}

void Ot3::help(const std::vector<uint8_t>& choices) {
  // sanity check.
  YASL_ENFORCE(comm_->getRank() == roles_.helper);
  YASL_ENFORCE(choices.size() == static_cast<size_t>(numel_));

  // generate masks, same as sender
  ArrayRef w0;
  ArrayRef w1;
  std::tie(w0, w1) = genMasks();
  YASL_ENFORCE(w0.numel() == numel_);
  YASL_ENFORCE(w1.numel() == numel_);

  // gen chosen masks
  ArrayRef wc = ring_select(choices, w0, w1);

  // send to receiver
  comm_->sendAsync(roles_.receiver, wc, "wc");
}

}  // namespace spu::mpc::aby3
