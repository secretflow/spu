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

#include <string>
#include <utility>
#include <vector>

#include "seal/seal.h"

namespace spu::pir {
// Interface which read db data.
class IDbElementProvider {
 public:
  virtual ~IDbElementProvider() = default;

  // Read at  `index` item and return data. An empty returned vector
  // is treated as the end of stream.
  virtual std::vector<uint8_t> ReadElement(size_t index) = 0;
  virtual std::vector<uint8_t> ReadElement(size_t index, size_t size) = 0;

  virtual size_t GetDbSize() = 0;
  virtual size_t GetDbByteSize() = 0;
};

// Interface which read batch of db data.
class IDbPlaintextStore {
 public:
  virtual ~IDbPlaintextStore() = default;

  virtual void SetSubDbNumber(size_t sub_db_num) = 0;

  virtual void SavePlaintext(const seal::Plaintext& plaintext,
                             size_t sub_db_index) = 0;

  virtual void SavePlaintexts(const std::vector<seal::Plaintext>& plaintext,
                              size_t sub_db_index) = 0;
  virtual std::vector<seal::Plaintext> ReadPlaintexts(size_t sub_db_index) = 0;
};

class MemoryDbElementProvider : public IDbElementProvider {
 public:
  explicit MemoryDbElementProvider(const std::vector<uint8_t>& items,
                                   size_t element_size)
      : items_(std::move(items)), element_size_(element_size) {}

  std::vector<uint8_t> ReadElement(size_t index) override;
  std::vector<uint8_t> ReadElement(size_t index, size_t size) override;

  size_t GetDbSize() override { return items_.size() / element_size_; }
  size_t GetDbByteSize() override { return items_.size(); }

  const std::vector<uint8_t>& items() const;

 private:
  const std::vector<uint8_t> items_;
  size_t element_size_;
};

class MemoryDbPlaintextStore : public IDbPlaintextStore {
 public:
  virtual ~MemoryDbPlaintextStore() = default;

  void SetSubDbNumber(size_t sub_db_num) override;

  void SavePlaintext(const seal::Plaintext& plaintext,
                     size_t sub_db_index) override;

  void SavePlaintexts(const std::vector<seal::Plaintext>& plaintexts,
                      size_t sub_db_index) override;
  std::vector<seal::Plaintext> ReadPlaintexts(size_t sub_db_index) override;

 private:
  std::vector<std::vector<seal::Plaintext>> db_vec_;
};

}  // namespace spu::pir
