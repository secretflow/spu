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

#include "libspu/psi/cryptor/ecc_cryptor.h"

#include <vector>

#include "yacl/crypto/base/hash/hash_utils.h"
#include "yacl/utils/parallel.h"

namespace spu::psi {

namespace {
std::string CreateFlattenEccBuffer(const std::vector<std::string>& items,
                                   size_t item_size,
                                   size_t chosen_size = kEccKeySize) {
  std::string ret;
  ret.reserve(items.size() * item_size);
  size_t size = std::min<size_t>(chosen_size, item_size);
  for (const auto& item : items) {
    SPU_ENFORCE(item.size() == item_size, "item.size:{}, item_size:{}",
                item.size(), item_size);
    ret.append(item.data(), size);
  }
  return ret;
}

std::string CreateFlattenEccBuffer(const std::vector<absl::string_view>& items,
                                   size_t item_size,
                                   size_t chosen_size = kEccKeySize) {
  std::string ret;
  ret.reserve(items.size() * item_size);
  size_t size = std::min<size_t>(chosen_size, item_size);
  for (const auto& item : items) {
    SPU_ENFORCE(item.size() == item_size, "item.size:{}, item_size:{}",
                item.size(), item_size);
    ret.append(item.data(), size);
  }
  return ret;
}

std::vector<std::string> CreateItemsFromFlattenEccBuffer(
    std::string_view buf, size_t item_size = kEccKeySize) {
  SPU_ENFORCE(buf.size() % item_size == 0);
  size_t num_item = buf.size() / item_size;
  std::vector<std::string> ret;
  ret.reserve(num_item);
  for (size_t i = 0; i < num_item; i++) {
    ret.emplace_back(buf.data() + i * item_size, item_size);
  }
  return ret;
}

}  // namespace

std::vector<uint8_t> IEccCryptor::HashToCurve(
    absl::Span<const char> input) const {
  auto d = yacl::crypto::Sha256(input);
  return {d.begin(), d.end()};
}

std::vector<std::string> Mask(const std::shared_ptr<IEccCryptor>& cryptor,
                              const std::vector<std::string>& items) {
  std::string batch_points = CreateFlattenEccBuffer(
      items, cryptor->GetMaskLength(), cryptor->GetMaskLength());
  std::string out_points(batch_points.size(), '\0');
  cryptor->EccMask(batch_points, absl::MakeSpan(out_points));
  return CreateItemsFromFlattenEccBuffer(out_points, cryptor->GetMaskLength());
}

std::vector<std::string> Mask(const std::shared_ptr<IEccCryptor>& cryptor,
                              const std::vector<absl::string_view>& items) {
  std::string batch_points = CreateFlattenEccBuffer(
      items, cryptor->GetMaskLength(), cryptor->GetMaskLength());
  std::string out_points(batch_points.size(), '\0');
  cryptor->EccMask(batch_points, absl::MakeSpan(out_points));
  return CreateItemsFromFlattenEccBuffer(out_points, cryptor->GetMaskLength());
}

std::string HashInput(const std::shared_ptr<IEccCryptor>& cryptor,
                      const std::string& item) {
  auto sha_bytes = cryptor->HashToCurve(item);
  std::string ret(sha_bytes.size(), '\0');
  std::memcpy(ret.data(), sha_bytes.data(), sha_bytes.size());
  return ret;
}

std::vector<std::string> HashInputs(const std::shared_ptr<IEccCryptor>& cryptor,
                                    const std::vector<std::string>& items) {
  std::vector<std::string> ret(items.size());
  yacl::parallel_for(0, items.size(), 1, [&](int64_t begin, int64_t end) {
    for (int64_t idx = begin; idx < end; ++idx) {
      ret[idx] = HashInput(cryptor, items[idx]);
    }
  });
  return ret;
}

}  // namespace spu::psi
