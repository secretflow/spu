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

#include <cstdint>
#include <numeric>
#include <string>
#include <type_traits>
#include <utility>

#include "yasl/base/buffer.h"
#include "yasl/base/exception.h"
#include "yasl/link/link.h"

#include "spu/mpc/object.h"

// This module defines the protocol comm pattern used for all
// protocols.

namespace spu::mpc {

enum class ReduceOp {
  INVALID = 0,
  ADD = 1,
  XOR = 2,
};

// yasl::link does not make assumption on data types, (it works on buffer),
// which means it's hard to write algorithms which depends on data arithmetics
// like reduce/allreduce.
//
// In mpc module, we have concrete data type definition, so we can fill this
// gap.
class Communicator : public State {
 public:
  static constexpr char kBindName[] = "Communicator";

  struct Stats {
    //
    size_t latency = 0;

    // Number of communication in bytes.
    //
    // For collective MPI algorithms only.
    // TODO(jint) add formal definition for asymmetric algorithms.
    size_t comm = 0;

    Stats operator-(const Stats& rhs) const {
      return {latency - rhs.latency, comm - rhs.comm};
    }
  };

  mutable Stats stats_;

  const std::shared_ptr<yasl::link::Context> lctx_;

  explicit Communicator(std::shared_ptr<yasl::link::Context> lctx)
      : lctx_(std::move(lctx)) {}

  Stats getStats() const { return stats_; }

  // only use when you're 100% sure what you are doing
  void addCommStatsManually(size_t latency, size_t comm) {
    stats_.latency += latency;
    stats_.comm += comm;
  }

  size_t getWorldSize() const { return lctx_->WorldSize(); }

  size_t getRank() const { return lctx_->Rank(); }

  ArrayRef allReduce(ReduceOp op, const ArrayRef& in, std::string_view tag);

  ArrayRef reduce(ReduceOp op, const ArrayRef& in, size_t root,
                  std::string_view tag);

  ArrayRef rotate(const ArrayRef& in, std::string_view tag);

  void sendAsync(size_t dst_rank, const ArrayRef& in, std::string_view tag);

  ArrayRef recv(size_t src_rank, Type eltype, std::string_view tag);

  template <typename T>
  std::vector<T> rotate(absl::Span<T const> in, std::string_view tag);

  template <typename T>
  void sendAsync(size_t dst_rank, absl::Span<T const> in, std::string_view tag);

  template <typename T>
  std::vector<T> recv(size_t src_rank, std::string_view tag);
};

template <typename T>
std::vector<T> Communicator::rotate(absl::Span<T const> in,
                                    std::string_view tag) {
  yasl::ByteContainerView bv(reinterpret_cast<uint8_t const*>(in.data()),
                             sizeof(T) * in.size());
  lctx_->SendAsync(lctx_->PrevRank(), bv, tag);
  auto buf = lctx_->Recv(lctx_->NextRank(), tag);

  stats_.latency += 1;
  stats_.comm += in.size() * sizeof(T);

  YASL_ENFORCE(buf.size() == static_cast<int64_t>(sizeof(T) * in.size()));
  return std::vector<T>(buf.data<T>(), buf.data<T>() + in.size());
}

template <typename T>
void Communicator::sendAsync(size_t dst_rank, absl::Span<T const> in,
                             std::string_view tag) {
  yasl::ByteContainerView bv(reinterpret_cast<uint8_t const*>(in.data()),
                             sizeof(T) * in.size());
  lctx_->SendAsync(dst_rank, bv, tag);
}

template <typename T>
std::vector<T> Communicator::recv(size_t src_rank, std::string_view tag) {
  auto buf = lctx_->Recv(src_rank, tag);
  YASL_ENFORCE(buf.size() % sizeof(T) == 0);
  auto numel = buf.size() / sizeof(T);
  // TODO: use a container which memory could be stealed.
  return std::vector<T>(buf.data<T>(), buf.data<T>() + numel);
}

}  // namespace spu::mpc
