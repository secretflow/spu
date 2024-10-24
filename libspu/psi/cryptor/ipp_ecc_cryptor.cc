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

#include "libspu/psi/cryptor/ipp_ecc_cryptor.h"

#include <array>

#include "crypto_mb/x25519.h"
#include "yacl/utils/parallel.h"

namespace spu::psi {

void IppEccCryptor::EccMask(absl::Span<const char> batch_points,
                            absl::Span<char> dest_points) const {
  SPU_ENFORCE(batch_points.size() % kEccKeySize == 0);

  using Item = std::array<unsigned char, kEccKeySize>;
  static_assert(sizeof(Item) == kEccKeySize);

  std::array<const int8u *, 8> ptr_sk;
  std::fill(ptr_sk.begin(), ptr_sk.end(),
            static_cast<const int8u *>(&private_key_[0]));

  int8u key_data[8][32];  // Junk buffer

  auto mask_functor = [&ptr_sk, &key_data](absl::Span<const Item> in,
                                           absl::Span<Item> out) {
    size_t current_batch_size = in.size();

    std::array<const int8u *, 8> ptr_pk;
    std::array<int8u *, 8> ptr_key;

    for (size_t i = 0; i < 8; i++) {
      if (i < current_batch_size) {
        ptr_pk[i] = static_cast<const int8u *>(in[i].data());
        ptr_key[i] = static_cast<int8u *>(out[i].data());
      } else {
        ptr_pk[i] = static_cast<const int8u *>(in[0].data());
        ptr_key[i] = static_cast<int8u *>(key_data[i]);
      }
    }
    mbx_status status =
        mbx_x25519_mb8(ptr_key.data(), ptr_sk.data(), ptr_pk.data());
    SPU_ENFORCE(status == 0, "ippc mbx_x25519_mb8 Error: ", status);
  };

  absl::Span<const Item> input(
      reinterpret_cast<const Item *>(batch_points.data()),
      batch_points.size() / sizeof(Item));
  absl::Span<Item> output(reinterpret_cast<Item *>(dest_points.data()),
                          dest_points.size() / sizeof(Item));

  yacl::parallel_for(0, input.size(), 8, [&](int64_t begin, int64_t end) {
    for (int64_t idx = begin; idx < end; idx += 8) {
      int64_t current_batch_size =
          std::min(static_cast<int64_t>(8), end - begin);
      mask_functor(input.subspan(idx, current_batch_size),
                   output.subspan(idx, current_batch_size));
    }
  });
}

}  // namespace spu::psi
