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

#include <map>
#include <memory>
#include <vector>

#include "spdlog/spdlog.h"
#include "yacl/crypto/base/symmetric_crypto.h"
#include "yacl/utils/parallel.h"

#include "libspu/core/prelude.h"
#include "libspu/pir/seal_pir.h"
#include "libspu/psi/core/cuckoo_index.h"

namespace spu::pir {

struct MultiQueryOptions {
  spu::pir::SealPirOptions seal_options;
  size_t batch_number = 0;
  size_t cuckoo_hash_number = 3;
};

struct HashItem {
  uint8_t seed[32];
  size_t index;
};

struct MultiQueryItem {
  size_t db_index;
  uint128_t item_hash;
  size_t bin_item_index;
};

class MultiQuery {
 public:
  MultiQuery(const MultiQueryOptions &query_options,
             const psi::CuckooIndex::Options &cuckoo_params,
             yacl::ByteContainerView seed)
      : query_options_(query_options), cuckoo_params_(cuckoo_params) {
    SPU_ENFORCE(seed.size() <= oracle_seed_.size());

    std::memcpy(oracle_seed_.data(), seed.data(), seed.size());
    uint128_t crypto_key, crypto_iv = 0;

    std::memcpy(&crypto_key, seed.data(), sizeof(uint128_t));

    crypto_ = std::make_unique<yacl::crypto::SymmetricCrypto>(
        yacl::crypto::SymmetricCrypto::CryptoType::AES128_ECB, crypto_key,
        crypto_iv);
  }

  size_t GetMaxBinItemSize() const { return max_bin_item_size_; }

 protected:
  uint128_t HashItemIndex(size_t index) {
    uint128_t plaintext = yacl::MakeUint128(0, index);

    // aes(x) xor x
    return crypto_->Encrypt(plaintext) ^ plaintext;
  }

  MultiQueryOptions query_options_;
  spu::psi::CuckooIndex::Options cuckoo_params_;
  std::array<uint8_t, 32> oracle_seed_;

  size_t max_bin_item_size_ = 0;
  std::unique_ptr<yacl::crypto::SymmetricCrypto> crypto_;
};

class MultiQueryServer : public MultiQuery {
 public:
  MultiQueryServer(const MultiQueryOptions &options,
                   const psi::CuckooIndex::Options &cuckoo_params,
                   yacl::ByteContainerView seed)
      : MultiQuery(options, cuckoo_params, seed) {
    simple_hash_.resize(cuckoo_params_.NumBins());

    GenerateSimpleHash();

    spu::pir::SealPirOptions pir_options{
        query_options_.seal_options.poly_modulus_degree, max_bin_item_size_,
        query_options_.seal_options.element_size};

    for (size_t idx = 0; idx < cuckoo_params.NumBins(); ++idx) {
      std::shared_ptr<IDbPlaintextStore> plaintext_store =
          std::make_shared<MemoryDbPlaintextStore>();
      pir_server_.push_back(
          std::make_unique<SealPirServer>(pir_options, plaintext_store));
    }
  }

  void GenerateSimpleHash();

  std::vector<size_t> &GetBin(size_t index) { return simple_hash_[index]; }

  size_t GetBinNum() const { return simple_hash_.size(); }

  void SetDatabase(yacl::ByteContainerView db_bytes);

  void SetGaloisKeys(const seal::GaloisKeys &galkey) {
    for (size_t idx = 0; idx < pir_server_.size(); ++idx) {
      pir_server_[idx]->SetGaloisKeys(galkey);
    }
  }

  void RecvGaloisKeys(const std::shared_ptr<yacl::link::Context> &link_ctx);

  void DoMultiPirAnswer(const std::shared_ptr<yacl::link::Context> &link_ctx);

 private:
  std::vector<std::vector<size_t>> simple_hash_;

  std::vector<std::unique_ptr<SealPirServer>> pir_server_;
};

class MultiQueryClient : public MultiQuery {
 public:
  MultiQueryClient(const MultiQueryOptions &options,
                   const psi::CuckooIndex::Options &cuckoo_params,
                   yacl::ByteContainerView seed)
      : MultiQuery(options, cuckoo_params, seed) {
    GenerateSimpleHashMap();

    // seal pir client
    SealPirOptions client_options{
        query_options_.seal_options.poly_modulus_degree, max_bin_item_size_,
        query_options_.seal_options.element_size};

    pir_client_ = std::make_unique<SealPirClient>(client_options);
  }

  std::vector<MultiQueryItem> GenerateBatchQueryIndex(
      const std::vector<size_t> &multi_query_index);

  seal::GaloisKeys GenerateGaloisKeys() {
    return pir_client_->GenerateGaloisKeys();
  }

  void SendGaloisKeys(const std::shared_ptr<yacl::link::Context> &link_ctx);

  std::vector<std::vector<uint8_t>> DoMultiPirQuery(
      const std::shared_ptr<yacl::link::Context> &link_ctx,
      const std::vector<size_t> &multi_query_index);

 private:
  void GenerateSimpleHashMap();

  std::vector<std::map<uint128_t, size_t>> simple_hash_map_;

  std::unique_ptr<SealPirClient> pir_client_;
};

}  // namespace spu::pir
