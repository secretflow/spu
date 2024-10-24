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

#include "libspu/psi/cryptor/sodium_curve25519_cryptor.h"

extern "C" {
#include "sodium.h"
}

#include <iostream>

#include "yacl/crypto/base/hash/hash_utils.h"
#include "yacl/utils/parallel.h"

namespace spu::psi {

void SodiumCurve25519Cryptor::EccMask(absl::Span<const char> batch_points,
                                      absl::Span<char> dest_points) const {
  SPU_ENFORCE(batch_points.size() % kEccKeySize == 0);

  using Item = std::array<unsigned char, kEccKeySize>;
  static_assert(sizeof(Item) == kEccKeySize);

  auto mask_functor = [this](const Item& in, Item& out) {
    SPU_ENFORCE(out.size() == kEccKeySize);
    SPU_ENFORCE(in.size() == kEccKeySize);

    SPU_ENFORCE(0 == crypto_scalarmult_curve25519(
                         out.data(), this->private_key_, in.data()));
  };

  absl::Span<const Item> input(
      reinterpret_cast<const Item*>(batch_points.data()),
      batch_points.size() / sizeof(Item));
  absl::Span<Item> output(reinterpret_cast<Item*>(dest_points.data()),
                          dest_points.size() / sizeof(Item));

  yacl::parallel_for(0, input.size(), 1, [&](int64_t begin, int64_t end) {
    for (int64_t idx = begin; idx < end; ++idx) {
      mask_functor(input[idx], output[idx]);
    }
  });
}

std::vector<uint8_t> SodiumCurve25519Cryptor::KeyExchange(
    const std::shared_ptr<yacl::link::Context>& link_ctx) {
  std::array<uint8_t, kEccKeySize> self_public_key;
  crypto_scalarmult_curve25519_base(self_public_key.data(), this->private_key_);
  yacl::Buffer self_pubkey_buf(self_public_key.data(), self_public_key.size());
  link_ctx->SendAsyncThrottled(
      link_ctx->NextRank(), self_pubkey_buf,
      fmt::format("send rank-{} public key", link_ctx->Rank()));
  yacl::Buffer peer_pubkey_buf = link_ctx->Recv(
      link_ctx->NextRank(),
      fmt::format("recv rank-{} public key", link_ctx->NextRank()));
  std::vector<uint8_t> dh_key(kEccKeySize);
  SPU_ENFORCE(0 == crypto_scalarmult_curve25519(
                       dh_key.data(), this->private_key_,
                       (const unsigned char*)peer_pubkey_buf.data()));
  const auto shared_key = yacl::crypto::Blake3(dh_key);
  return {shared_key.begin(), shared_key.end()};
}

}  // namespace spu::psi