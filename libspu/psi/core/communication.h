// Copyright 2022 Ant Group Co., Ltd.
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

#include <memory>
#include <string>

#include "yacl/base/buffer.h"
#include "yacl/link/link.h"

#include "libspu/psi/utils/serializable.pb.h"

namespace spu::psi {

// I prefer 4096.
inline constexpr size_t kEcdhPsiBatchSize = 4096;

// Ecc256 requires 32 bytes.
inline constexpr size_t kKeySize = 32;
inline constexpr size_t kHashSize = kKeySize;

// The final comparison bytes.
// Hongcheng suggested that 90 bits would be enough. Here we give 96 bits.
//
// The least significant bits(LSB) of g^{ab} are globally indistinguishable from
// a random bit-string, Reference:
// Optimal Randomness Extraction from a Diffie-Hellman Element
// EUROCRYPT 2009 https://link.springer.com/chapter/10.1007/978-3-642-01001-9_33
//
inline constexpr size_t kFinalCompareBytes = 12;

enum class PsiRoleType {
  Sender,
  Receiver,
};

struct PsiDataBatch {
  // current batch item num
  uint32_t item_num = 0;

  // Pack all items in a single `std::string` to save bandwidth.
  std::string flatten_bytes;

  // Metadata.
  // Batch index. Start from 0.
  int32_t batch_index = 0;
  bool is_last_batch = false;

  // The type hint for each message.
  //
  // "enc": the first stage ciphertext
  // "dual.enc": the second stage ciphertext
  std::string type;

  // deprecated. use IcPsiBatchSerializer instead.
  yacl::Buffer Serialize() const {
    proto::PsiDataBatchProto proto;
    proto.set_item_num(item_num);
    proto.set_flatten_bytes(flatten_bytes);
    proto.set_is_last_batch(is_last_batch);

    yacl::Buffer buf(proto.ByteSizeLong());
    proto.SerializeToArray(buf.data(), buf.size());
    return buf;
  }

  // deprecated. use IcPsiBatchSerializer instead.
  static PsiDataBatch Deserialize(const yacl::Buffer& buf) {
    proto::PsiDataBatchProto proto;
    proto.ParseFromArray(buf.data(), buf.size());

    PsiDataBatch batch;
    batch.item_num = proto.item_num();
    batch.flatten_bytes = proto.flatten_bytes();
    batch.is_last_batch = proto.is_last_batch();

    return batch;
  }
};

// Serialize data using an interconnection standard protocol
class IcPsiBatchSerializer {
 public:
  // support interconnection standard.
  static yacl::Buffer Serialize(PsiDataBatch&& batch);
  static PsiDataBatch Deserialize(yacl::ByteContainerView buf);
};

std::shared_ptr<yacl::link::Context> CreateP2PLinkCtx(
    const std::string& id_prefix,
    const std::shared_ptr<yacl::link::Context>& link_ctx, size_t peer_rank);
}  // namespace spu::psi
