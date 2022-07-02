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

#include "spu/mpc/util/communicator.h"

#include "spu/mpc/util/ring_ops.h"

namespace spu::mpc {
namespace {

// When ArrayRef is transferred via network, it's suppused to be compact.
constexpr int64_t kStride = 1;
constexpr int64_t kOffset = 0;

std::shared_ptr<yasl::Buffer> stealBuffer(yasl::Buffer&& buf) {
  return std::make_shared<yasl::Buffer>(std::move(buf));
}

}  // namespace

ArrayRef Communicator::allReduce(ReduceOp op, const ArrayRef& in,
                                 std::string_view tag) {
  const auto buf = in.getOrCreateCompactBuf();

  std::vector<yasl::Buffer> all_str = yasl::link::AllGather(lctx_, *buf, tag);

  YASL_ENFORCE(all_str.size() == getWorldSize());
  ArrayRef res = in.clone();
  for (size_t idx = 0; idx < all_str.size(); idx++) {
    if (idx == getRank()) {
      continue;
    }

    auto arr = ArrayRef(stealBuffer(std::move(all_str[idx])), in.eltype(),
                        in.numel(), kStride, kOffset);
    if (op == ReduceOp::ADD) {
      ring_add_(res, arr);
    } else if (op == ReduceOp::XOR) {
      ring_xor_(res, arr);
    } else {
      YASL_THROW("unsupported reduce op={}", static_cast<int>(op));
    }
  }

  stats_.latency += 1;
  stats_.comm += buf->size() * (lctx_->WorldSize() - 1);

  return res;
}

ArrayRef Communicator::reduce(ReduceOp op, const ArrayRef& in, size_t root,
                              std::string_view tag) {
  const auto buf = in.getOrCreateCompactBuf();

  std::vector<yasl::Buffer> all_str =
      yasl::link::Gather(lctx_, *buf, root, tag);

  YASL_ENFORCE(all_str.size() == getWorldSize());
  ArrayRef res = in.clone();
  for (size_t idx = 0; idx < all_str.size(); idx++) {
    if (idx == getRank()) {
      continue;
    }

    auto arr = ArrayRef(stealBuffer(std::move(all_str[idx])), in.eltype(),
                        in.numel(), kStride, kOffset);
    if (op == ReduceOp::ADD) {
      ring_add_(res, arr);
    } else if (op == ReduceOp::XOR) {
      ring_xor_(res, arr);
    } else {
      YASL_THROW("unsupported reduce op={}", static_cast<int>(op));
    }
  }

  stats_.latency += 1;
  stats_.comm += buf->size();

  return res;
}

ArrayRef Communicator::rotate(const ArrayRef& in, std::string_view tag) {
  const auto buf = in.getOrCreateCompactBuf();

  // NOTE: need to ensure link->SendAsync is a secure P2P channel
  lctx_->SendAsync(lctx_->PrevRank(), *buf, tag);

  auto res_buf = lctx_->Recv(lctx_->NextRank(), tag);

  stats_.latency += 1;
  stats_.comm += buf->size();

  return ArrayRef(stealBuffer(std::move(res_buf)), in.eltype(), in.numel(),
                  kStride, kOffset);
}

void Communicator::sendAsync(size_t dst_rank, const ArrayRef& in,
                             std::string_view tag) {
  const auto buf = in.getOrCreateCompactBuf();

  // NOTE: need to ensure link->SendAsync is a secure P2P channel
  lctx_->SendAsync(dst_rank, *buf, tag);
}

ArrayRef Communicator::recv(size_t src_rank, Type eltype,
                            std::string_view tag) {
  auto buf = lctx_->Recv(src_rank, tag);

  auto numel = buf.size() / eltype.size();
  return ArrayRef(stealBuffer(std::move(buf)), eltype, numel, kStride, kOffset);
}

}  // namespace spu::mpc
