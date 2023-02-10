// Copyright 2023 Ant Group Co., Ltd.
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

#include "libspu/mpc/spdz2k/commitment.h"

#include "spdlog/spdlog.h"
#include "yacl/crypto/base/hash/blake3.h"
#include "yacl/crypto/base/hash/hash_utils.h"
#include "yacl/crypto/utils/rand.h"

namespace spu::mpc {
// TODO: Maybe we need a better commit scheme
std::string commit(size_t send_player, absl::string_view msg,
                   absl::string_view r, size_t hash_len,
                   yacl::crypto::HashAlgorithm hash_type) {
  std::unique_ptr<yacl::crypto::HashInterface> hash_algo;
  switch (hash_type) {
    case yacl::crypto::HashAlgorithm::BLAKE3:
      hash_algo = std::make_unique<yacl::crypto::Blake3Hash>();
      break;
    default:
      YACL_THROW("Unsupported hash algo in commitment scheme");
  }

  hash_algo->Update(std::to_string(send_player));
  hash_algo->Update(msg);
  hash_algo->Update(r);
  std::vector<uint8_t> hash = hash_algo->CumulativeHash();
  YACL_ENFORCE(hash_len <= hash.size());

  std::string hash_str(reinterpret_cast<char*>(hash.data()), hash_len);

  return hash_str;
}

bool commit_and_open(const std::shared_ptr<yacl::link::Context>& lctx,
                     const std::string& z_str,
                     std::vector<std::string>* z_strs) {
  bool res = true;
  size_t send_player = lctx->Rank();
  uint128_t rs = yacl::crypto::RandSeed();
  std::string rs_str(reinterpret_cast<char*>(&rs), sizeof(rs));
  // 1. commit and send
  auto cmt = commit(send_player, z_str, rs_str);
  auto all_cmts = yacl::link::AllGather(
      lctx, yacl::ByteContainerView(cmt.data(), cmt.size()),
      "COMMITMENT::COMMIT");

  // 2. open commit
  std::string open_str = z_str + rs_str;
  auto all_opens = yacl::link::AllGather(
      lctx, yacl::ByteContainerView(open_str.data(), open_str.size()),
      "COMMITMENT::OPEN");

  // 3. check consistency
  YACL_ENFORCE(z_strs != nullptr);
  for (size_t i = 0; i < lctx->WorldSize(); ++i) {
    if (i == lctx->Rank()) {
      z_strs->emplace_back(z_str);
      continue;
    }
    auto _open = std::string_view(all_opens[i]);
    auto _z = _open.substr(0, z_str.size());
    auto _rs = _open.substr(z_str.size(), rs_str.size());

    auto ref_cmt = commit(i, _z, _rs);
    if (ref_cmt != std::string_view(all_cmts[i])) {
      res = false;
      SPDLOG_INFO("commit check fail for rank {}", i);
    }

    z_strs->emplace_back(_z);
  }

  return res;
}

}  // namespace spu::mpc
