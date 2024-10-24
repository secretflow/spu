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

#include "libspu/pir/seal_pir_utils.h"

#include "spdlog/spdlog.h"

#include "libspu/core/prelude.h"

namespace spu::pir {

std::vector<uint8_t> MemoryDbElementProvider::ReadElement(size_t index) {
  SPU_ENFORCE(index < items_.size());

  std::vector<uint8_t> element(element_size_);

  std::memcpy(element.data(), &items_[index], element_size_);
  return element;
}

std::vector<uint8_t> MemoryDbElementProvider::ReadElement(size_t index,
                                                          size_t size) {
  SPU_ENFORCE((index + size) <= items_.size());

  std::vector<uint8_t> element(size);

  std::memcpy(element.data(), &items_[index], size);
  return element;
}

void MemoryDbPlaintextStore::SetSubDbNumber(size_t sub_db_num) {
  db_vec_.resize(sub_db_num);
}

void MemoryDbPlaintextStore::SavePlaintext(const seal::Plaintext& plaintext,
                                           size_t sub_db_index) {
  db_vec_[sub_db_index].push_back(plaintext);
}

void MemoryDbPlaintextStore::SavePlaintexts(
    const std::vector<seal::Plaintext>& plaintexts, size_t sub_db_index) {
  for (const auto& plaintext : plaintexts) {
    db_vec_[sub_db_index].push_back(plaintext);
  }
}

std::vector<seal::Plaintext> MemoryDbPlaintextStore::ReadPlaintexts(
    size_t sub_db_index) {
  return std::move(db_vec_[sub_db_index]);
}

}  // namespace spu::pir
