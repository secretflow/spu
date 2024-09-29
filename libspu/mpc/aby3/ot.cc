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

#include "libspu/mpc/aby3/ot.h"

#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::aby3 {
namespace {

// test a in front of b
inline bool inFrontOf(size_t a, size_t b) { return (a + 1) % 3 == b; }

inline SemanticType GetSemanticType(int64_t field) {
  switch (field) {
    case 32:
      return SE_I32;
    case 64:
      return SE_I64;
    case 128:
      return SE_I128;
  }
  return SE_INVALID;
}

}  // namespace

Ot3::Ot3(size_t field, absl::Span<const int64_t> shape, const RoleRanks& roles,
         Communicator* comm, PrgState* prg_state, bool reentrancy)
    : field_(field),
      shape_(shape.begin(), shape.end()),
      roles_(roles),
      comm_(comm),
      prg_state_(prg_state),
      reentrancy_(reentrancy) {
  if (!reentrancy_) {
    masks_ = genMasks();
  }
}

std::pair<MemRef, MemRef> Ot3::genMasks() {
  const Type type = makeType<RingTy>(GetSemanticType(field_), field_);
  const size_t size = shape_.numel() * type.size();

  MemRef w0(type, shape_);
  MemRef w1(type, shape_);

  if (comm_->getRank() == roles_.sender) {
    if (inFrontOf(roles_.sender, roles_.helper)) {
      prg_state_->fillPrssPair(nullptr, w0.data(), size);
      prg_state_->fillPrssPair(nullptr, w1.data(), size);
    } else {
      SPU_ENFORCE(inFrontOf(roles_.helper, roles_.sender));
      prg_state_->fillPrssPair(w0.data(), nullptr, size);
      prg_state_->fillPrssPair(w1.data(), nullptr, size);
    }
  } else if (comm_->getRank() == roles_.helper) {
    if (inFrontOf(roles_.sender, roles_.helper)) {
      prg_state_->fillPrssPair(w0.data(), nullptr, size);
      prg_state_->fillPrssPair(w1.data(), nullptr, size);
    } else {
      SPU_ENFORCE(inFrontOf(roles_.helper, roles_.sender));
      prg_state_->fillPrssPair(nullptr, w0.data(), size);
      prg_state_->fillPrssPair(nullptr, w1.data(), size);
    }
  } else {
    SPU_ENFORCE(comm_->getRank() == roles_.receiver);
  }

  return {w0, w1};
}

void Ot3::send(const MemRef& m0, const MemRef& m1) {
  // sanity check.
  SPU_ENFORCE(comm_->getRank() == roles_.sender);
  SPU_ENFORCE(m0.shape() == shape_);
  SPU_ENFORCE(m1.shape() == shape_);

  // generate masks
  MemRef w0;
  MemRef w1;
  if (!reentrancy_) {
    SPU_ENFORCE(masks_.has_value(), "this OT instance can only use once.");
    std::tie(w0, w1) = masks_.value();
    masks_.reset();
  } else {
    std::tie(w0, w1) = genMasks();
  }
  SPU_ENFORCE(w0.shape() == shape_);
  SPU_ENFORCE(w1.shape() == shape_);

  // mask the values
  auto masked_m0 = ring_xor(m0, w0);
  auto masked_m1 = ring_xor(m1, w1);

  comm_->sendAsync(roles_.receiver, masked_m0, "m0");
  comm_->sendAsync(roles_.receiver, masked_m1, "m1");
}

MemRef Ot3::recv(const std::vector<uint8_t>& choices,
                 SemanticType semantic_type) {
  // sanity check.
  SPU_ENFORCE(comm_->getRank() == roles_.receiver);
  SPU_ENFORCE(choices.size() == static_cast<size_t>(shape_.numel()));

  const auto ty = makeType<RingTy>(semantic_type, field_);

  if (!reentrancy_) {
    SPU_ENFORCE(masks_.has_value(), "this OT instance can only use once.");
    masks_.reset();
  } else {
    genMasks();
  }

  // get masked messages from sender.
  auto m0 = comm_->recv(roles_.sender, ty, "m0");
  auto m1 = comm_->recv(roles_.sender, ty, "m1");

  auto mc = ring_select(choices, m0, m1);
  // get chosen masks
  auto wc = comm_->recv(roles_.helper, ty, "wc");

  SPU_ENFORCE(m0.numel() == static_cast<int64_t>(choices.size()));

  // reconstruct mc
  ring_xor_(mc, wc);

  return mc;
}

void Ot3::help(const std::vector<uint8_t>& choices) {
  // sanity check.
  SPU_ENFORCE(comm_->getRank() == roles_.helper);
  SPU_ENFORCE(choices.size() == static_cast<size_t>(shape_.numel()));

  // generate masks, same as sender
  MemRef w0;
  MemRef w1;
  if (!reentrancy_) {
    SPU_ENFORCE(masks_.has_value(), "this OT instance can only use once.");
    std::tie(w0, w1) = masks_.value();
    masks_.reset();
  } else {
    std::tie(w0, w1) = genMasks();
  }
  SPU_ENFORCE(w0.shape() == shape_);
  SPU_ENFORCE(w1.shape() == shape_);

  // gen chosen masks
  auto wc = ring_select(choices, w0, w1);

  // send to receiver
  comm_->sendAsync(roles_.receiver, wc, "wc");
}

}  // namespace spu::mpc::aby3
