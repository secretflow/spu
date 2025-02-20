// Copyright 2024 Ant Group Co., Ltd.
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

#include "libspu/mpc/swift/commitment.h"

#include "spdlog/spdlog.h"
#include "yacl/crypto/hash/blake3.h"
#include "yacl/crypto/hash/hash_utils.h"
#include "yacl/crypto/rand/rand.h"

#include "libspu/core/prelude.h"

namespace spu::mpc {
std::string commit(size_t send_player, absl::string_view msg,
                   absl::string_view r, size_t hash_len,
                   yacl::crypto::HashAlgorithm hash_type) {
  std::unique_ptr<yacl::crypto::HashInterface> hash_algo;
  switch (hash_type) {
    case yacl::crypto::HashAlgorithm::BLAKE3:
      hash_algo = std::make_unique<yacl::crypto::Blake3Hash>();
      break;
    default:
      SPU_THROW("Unsupported hash algo in commitment scheme");
  }

  hash_algo->Update(std::to_string(send_player));
  hash_algo->Update(msg);
  hash_algo->Update(r);
  std::vector<uint8_t> hash = hash_algo->CumulativeHash();
  SPU_ENFORCE(hash_len <= hash.size());

  std::string hash_str(reinterpret_cast<char*>(hash.data()), hash_len);

  return hash_str;
}

}  // namespace spu::mpc
