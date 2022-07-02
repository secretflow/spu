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

#include "spu/psi/cryptor/sodium_curve25519_cryptor.h"

extern "C" {
#include "sodium.h"
}

#include <iostream>

#include "yasl/crypto/hash_util.h"
#include "yasl/utils/parallel.h"

namespace spu {

void SodiumCurve25519Cryptor::EccMask(absl::Span<const char> batch_points,
                                      absl::Span<char> dest_points) const {
  YASL_ENFORCE(batch_points.size() % kEccKeySize == 0);

  using Item = std::array<unsigned char, kEccKeySize>;
  static_assert(sizeof(Item) == kEccKeySize);

  auto mask_functor = [this](const Item& in, Item& out) {
    YASL_ENFORCE(out.size() == kEccKeySize);
    YASL_ENFORCE(in.size() == kEccKeySize);

    YASL_ENFORCE(0 == crypto_scalarmult_curve25519(
                          out.data(), this->private_key_, in.data()));
  };

  absl::Span<const Item> input(
      reinterpret_cast<const Item*>(batch_points.data()),
      batch_points.size() / sizeof(Item));
  absl::Span<Item> output(reinterpret_cast<Item*>(dest_points.data()),
                          dest_points.size() / sizeof(Item));

  yasl::parallel_for(0, input.size(), 1, [&](int64_t begin, int64_t end) {
    for (int64_t idx = begin; idx < end; ++idx) {
      mask_functor(input[idx], output[idx]);
    }
  });
}

std::vector<uint8_t> SodiumCurve25519Cryptor::KeyExchange(
      const std::shared_ptr<yasl::link::Context> &link_ctx){
  std::array<uint8_t, kEccKeySize> self_public_key;

  crypto_scalarmult_curve25519_base(self_public_key.data(), this->private_key_);

  yasl::Buffer self_pubkey_buf(self_public_key.data(), self_public_key.size());

  link_ctx->SendAsync(link_ctx->NextRank(), self_pubkey_buf,
                      fmt::format("send rank-{} public key", link_ctx->Rank()));

  yasl::Buffer peer_pubkey_buf = link_ctx->Recv(
      link_ctx->NextRank(),
      fmt::format("recv rank-{} public key", link_ctx->NextRank()));

  std::vector<uint8_t> dh_key(kEccKeySize);
  YASL_ENFORCE(0 == crypto_scalarmult_curve25519(dh_key.data(), this->private_key_,
                   (const unsigned char*)peer_pubkey_buf.data()));

  std::vector<uint8_t> shared_key = yasl::crypto::Blake3(dh_key);
  return shared_key;
}

}  // namespace spu