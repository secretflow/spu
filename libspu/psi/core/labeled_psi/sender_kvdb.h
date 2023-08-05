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

// code reference https://github.com/microsoft/APSI/sender/sender_db.h
// Licensed under the MIT license.

#pragma once

// STD
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

// GSL
#include "gsl/span"

// APSI
#include "apsi/bin_bundle.h"
#include "apsi/crypto_context.h"
#include "apsi/item.h"
#include "apsi/psi_params.h"
#include "yacl/base/byte_container_view.h"
#include "yacl/io/kv/leveldb_kvstore.h"
#include "yacl/io/kv/memory_kvstore.h"

#include "libspu/psi/core/ecdh_oprf/basic_ecdh_oprf.h"
#include "libspu/psi/core/labeled_psi/sender_db.h"
#include "libspu/psi/utils/batch_provider.h"

// SEAL
#include "seal/plaintext.h"
#include "seal/util/locks.h"
#include "spdlog/spdlog.h"

namespace spu::psi {

/**
A SenderDB maintains an in-memory representation of the sender's set of items
and labels (in labeled mode). This data is not simply copied into the SenderDB
data structures, but also preprocessed heavily to allow for faster online
computation time. Since inserting a large number of new items into a SenderDB
can take time, it is not recommended to recreate the SenderDB when the
database changes a little bit. Instead, the class supports fast update and
deletion operations that should be preferred: SenderDB::InsertOrAssign and
SenderDB::remove.

The SenderDB constructor allows the label byte count to be specified;
unlabeled mode is activated by setting the label byte count to zero. It is
possible to optionally specify the size of the nonce used in encrypting the
labels, but this is best left to its default value unless the user is
absolutely sure of what they are doing.

The SenderDB requires substantially more memory than the raw data would. Part
of that memory can automatically be compressed when it is not in use; this
feature is enabled by default, and can be disabled when constructing the
SenderDB. The downside of in-memory compression is a performance reduction
from decompressing parts of the data when they are used, and recompressing
them if they are updated.
*/
class SenderKvDB : public ISenderDB {
 public:
  /**
  Creates a new SenderDB.
  */
  SenderKvDB(const apsi::PSIParams &params, yacl::ByteContainerView oprf_key,
             std::string_view kv_store_path = "",
             std::size_t label_byte_count = 0,
             std::size_t nonce_byte_count = 16, bool compressed = true);

  /**
  Clears the database. Every item and label will be removed. The OPRF key is
  unchanged.
  */
  void clear();

  /**
  Strips the SenderDB of all information not needed for serving a query.
  Returns a copy of the OPRF key and clears it from the SenderDB.
  */
  void strip();

  /**
   * @brief Insert data from BatchProvider
   *
   * @param batch_provider
   */
  void InsertOrAssign(const std::shared_ptr<IBatchProvider> &batch_provider,
                      size_t batch_size);

  /**
  Clears the database and inserts the given data. This function can be used
  only on a labeled SenderDB instance.
  */
  void SetData(const std::vector<std::string> &keys,
               const std::vector<std::string> &labels) {
    std::shared_ptr<IBatchProvider> batch_provider =
        std::make_shared<MemoryBatchProvider>(keys, labels);

    SetData(batch_provider);
  }

  /**
  Clears the database and inserts the given data. This function can be used
  only on an unlabeled SenderDB instance.
  */
  void SetData(const std::vector<std::string> &data) {
    std::shared_ptr<IBatchProvider> batch_provider =
        std::make_shared<MemoryBatchProvider>(data);

    SetData(batch_provider);
  }

  void SetData(const std::shared_ptr<IBatchProvider> &batch_provider,
               size_t batch_size = 500000) override {
    clear();
    InsertOrAssign(batch_provider, batch_size);
  }

  /**
  Returns the bundle at the given bundle index.
  */
  std::shared_ptr<apsi::sender::BinBundle> GetBinBundleAt(
      std::uint32_t bundle_idx, size_t cache_idx) override;

  /**
  Returns the total number of bin bundles at a specific bundle index.
  */
  std::size_t GetBinBundleCount(std::uint32_t bundle_idx) const override;

  /**
  Returns the total number of bin bundles.
  */
  std::size_t GetBinBundleCount() const override;

 private:
  void ClearInternal();

  void GenerateCaches();

  std::string kv_store_path_;
  std::shared_ptr<yacl::io::KVStore> meta_info_store_;

  std::vector<std::shared_ptr<yacl::io::IndexStore>> bundles_store_;
  std::vector<size_t> bundles_store_idx_;
};  // class SenderDB

}  // namespace spu::psi
