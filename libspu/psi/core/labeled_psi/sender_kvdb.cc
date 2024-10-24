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

// code reference https://github.com/microsoft/APSI/sender/sender_db.cpp
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

// we are using our own OPRF, the reason is we wanna make the oprf
// switchable between secp256k1, sm2 or other types

// STD
#include <algorithm>
#include <chrono>
#include <future>
#include <iterator>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <string>

// APSI
#include "apsi/psi_params.h"
#include "apsi/thread_pool_mgr.h"
#include "apsi/util/db_encoding.h"
#include "apsi/util/label_encryptor.h"
#include "apsi/util/utils.h"
#include "spdlog/spdlog.h"

#include "libspu/core/prelude.h"
#include "libspu/psi/core/ecdh_oprf/ecdh_oprf_selector.h"
#include "libspu/psi/core/labeled_psi/sender_kvdb.h"
#include "libspu/psi/core/labeled_psi/serialize.h"
#include "libspu/psi/utils/utils.h"

// Kuku
#include "kuku/locfunc.h"

// SEAL
#include "absl/strings/escaping.h"
#include "seal/util/common.h"
#include "seal/util/streambuf.h"
#include "yacl/crypto/utils/rand.h"
#include "yacl/utils/parallel.h"

namespace spu::psi {

namespace {

using DurationMillis = std::chrono::duration<double, std::milli>;

std::vector<std::pair<apsi::util::AlgItem, size_t>> PreprocessUnlabeledData(
    const apsi::HashedItem &hashed_item, const apsi::PSIParams &params) {
  // Some variables we'll need
  size_t bins_per_item = params.item_params().felts_per_item;
  size_t item_bit_count = params.item_bit_count();

  // Set up Kuku hash functions
  auto hash_funcs = labeled_psi::HashFunctions(params);

  std::vector<std::pair<apsi::util::AlgItem, size_t>> data_with_indices;

  // Serialize the data into field elements
  apsi::util::AlgItem alg_item = algebraize_item(
      hashed_item, item_bit_count, params.seal_params().plain_modulus());

  // Get the cuckoo table locations for this item and add to data_with_indices
  for (auto location : labeled_psi::AllLocations(hash_funcs, hashed_item)) {
    // The current hash value is an index into a table of Items. In reality
    // our BinBundles are tables of bins, which contain chunks of items. How
    // many chunks? bins_per_item many chunks
    size_t bin_idx = location * bins_per_item;

    // Store the data along with its index
    data_with_indices.emplace_back(std::make_pair(alg_item, bin_idx));
  }

  return data_with_indices;
}

std::vector<std::pair<apsi::util::AlgItemLabel, size_t>> PreprocessLabeledData(
    const std::pair<apsi::HashedItem, apsi::EncryptedLabel> &item_label_pair,
    const apsi::PSIParams &params,
    const std::vector<kuku::LocFunc> &hash_funcs) {
  SPDLOG_DEBUG("Start preprocessing {} labeled items", distance(begin, end));

  // Some variables we'll need
  size_t bins_per_item = params.item_params().felts_per_item;
  size_t item_bit_count = params.item_bit_count();

  std::vector<std::pair<apsi::util::AlgItemLabel, size_t>> data_with_indices;

  // Serialize the data into field elements
  const apsi::HashedItem &item = item_label_pair.first;
  const apsi::EncryptedLabel &label = item_label_pair.second;
  apsi::util::AlgItemLabel alg_item_label = algebraize_item_label(
      item, label, item_bit_count, params.seal_params().plain_modulus());

  std::set<size_t> loc_set;

  // Get the cuckoo table locations for this item and add to data_with_indices
  for (auto location : spu::psi::labeled_psi::AllLocations(hash_funcs, item)) {
    // The current hash value is an index into a table of Items. In reality
    // our BinBundles are tables of bins, which contain chunks of items. How
    // many chunks? bins_per_item many chunks
    if (loc_set.find(location) == loc_set.end()) {
      size_t bin_idx = location * bins_per_item;

      // Store the data along with its index
      data_with_indices.emplace_back(alg_item_label, bin_idx);
      loc_set.insert(location);
    }
  }

  return data_with_indices;
}

/**
Inserts the given items and corresponding labels into bin_bundles at their
respective cuckoo indices. It will only insert the data with bundle index in the
half-open range range indicated by work_range. If inserting into a BinBundle
would make the number of items in a bin larger than max_bin_size, this function
will create and insert a new BinBundle. If overwrite is set, this will overwrite
the labels if it finds an AlgItemLabel that matches the input perfectly.
*/
template <typename T>
void InsertOrAssignWorker(
    const std::vector<std::pair<T, size_t>> &data_with_indices,
    std::vector<std::shared_ptr<yacl::io::IndexStore>> *bundles_store,
    std::vector<size_t> *bundles_store_idx,
    const apsi::CryptoContext &crypto_context, uint32_t bundle_index,
    uint32_t bins_per_bundle, size_t label_size, size_t max_bin_size,
    size_t ps_low_degree, bool overwrite, bool compressed) {
  STOPWATCH(sender_stopwatch, "insert_or_assign_worker");
  SPDLOG_DEBUG(
      "Insert-or-Assign worker for bundle index {}; mode of operation: {}",
      bundle_index, overwrite ? "overwriting existing" : "inserting new");

  // Create the bundle set at the given bundle index
  std::vector<apsi::sender::BinBundle> bundle_set;

  // Iteratively insert each item-label pair at the given cuckoo index
  for (auto &data_with_idx : data_with_indices) {
    const T &data = data_with_idx.first;

    // Get the bundle index
    size_t cuckoo_idx = data_with_idx.second;
    size_t bin_idx;
    size_t bundle_idx;
    std::tie(bin_idx, bundle_idx) =
        labeled_psi::UnpackCuckooIdx(cuckoo_idx, bins_per_bundle);

    // If the bundle_idx isn't in the prescribed range, don't try to insert this
    // data
    if (bundle_idx != bundle_index) {
      // Dealing with this bundle index is not our job
      continue;
    }

    // Try to insert or overwrite these field elements in an existing BinBundle
    // at this bundle index. Keep track of whether or not we succeed.
    bool written = false;

    for (auto bundle_it = bundle_set.rbegin(); bundle_it != bundle_set.rend();
         bundle_it++) {
      // If we're supposed to overwrite, try to overwrite. One of these
      // BinBundles has to have the data we're trying to overwrite.
      if (overwrite) {
        // If we successfully overwrote, we're done with this bundle
        written = bundle_it->try_multi_overwrite(data, bin_idx);
        if (written) {
          break;
        }
      }

      // Do a dry-run insertion and see if the new largest bin size in the range
      // exceeds the limit
      int32_t new_largest_bin_size =
          bundle_it->multi_insert_dry_run(data, bin_idx);

      // Check if inserting would violate the max bin size constraint
      if (new_largest_bin_size > 0 &&
          seal::util::safe_cast<size_t>(new_largest_bin_size) < max_bin_size) {
        // All good
        bundle_it->multi_insert_for_real(data, bin_idx);
        written = true;
        break;
      }
    }

    // We tried to overwrite an item that doesn't exist. This should never
    // happen
    if (overwrite && !written) {
      SPDLOG_ERROR(
          "Insert-or-Assign worker: "
          "failed to overwrite item at bundle index {} because the item was "
          "not found",
          bundle_idx);
      SPU_THROW("tried to overwrite non-existent item");
    }

    // If we had conflicts everywhere when trying to insert, then we need to
    // make a new BinBundle and insert the data there
    if (!written) {
      // Make a fresh BinBundle and insert
      apsi::sender::BinBundle new_bin_bundle(
          crypto_context, label_size, max_bin_size, ps_low_degree,
          bins_per_bundle, compressed, false);
      int res = new_bin_bundle.multi_insert_for_real(data, bin_idx);

      // If even that failed, I don't know what could've happened
      if (res < 0) {
        SPDLOG_ERROR(
            "Insert-or-Assign worker: "
            "failed to insert item into a new BinBundle at bundle index {}",
            bundle_idx);
        SPU_THROW("failed to insert item into a new BinBundle");
      }

      // Push a new BinBundle to the set of BinBundles at this bundle index
      bundle_set.push_back(std::move(new_bin_bundle));
    }
  }

  const auto db_save_start = std::chrono::system_clock::now();

  for (auto &bundle : bundle_set) {
    size_t store_idx = (*bundles_store_idx)[bundle_index]++;

    // Generate the BinBundle caches
    // bundle.regen_cache();
    bundle.strip();

    std::stringstream stream;
    bundle.save(stream, store_idx);

    (*bundles_store)[bundle_index]->Put(store_idx, stream.str());
  }

  const auto db_save_end = std::chrono::system_clock::now();
  const DurationMillis db_save_duration = db_save_end - db_save_start;
  SPDLOG_INFO("*** step leveldb put duration:{}", db_save_duration.count());

  SPDLOG_DEBUG("Insert-or-Assign worker: finished processing bundle index {}",
               bundle_index);
}

void InsertOrAssignWorker(
    const std::shared_ptr<yacl::io::IndexStore> &indices_store,
    size_t indices_count,
    std::vector<std::shared_ptr<yacl::io::IndexStore>> *bundles_store,
    std::vector<size_t> *bundles_store_idx, bool is_labeled,
    const apsi::CryptoContext &crypto_context, uint32_t bundle_index,
    uint32_t bins_per_bundle, size_t label_size, size_t max_bin_size,

    size_t ps_low_degree, bool overwrite, bool compressed) {
  STOPWATCH(sender_stopwatch, "insert_or_assign_worker");

  SPDLOG_DEBUG(
      "Insert-or-Assign worker for bundle index {}; mode of operation: {}",
      bundle_index, overwrite ? "overwriting existing" : "inserting new");

  // Create the bundle set at the given bundle index
  std::vector<apsi::sender::BinBundle> bundle_set;

  // Iteratively insert each item-label pair at the given cuckoo index
  for (size_t i = 0; i < indices_count; ++i) {
    yacl::Buffer value;
    bool get_status = indices_store->Get(i, &value);
    SPU_ENFORCE(get_status, "get_status:{}", get_status);

    size_t cuckoo_idx;

    std::pair<apsi::util::AlgItemLabel, size_t> datalabel_with_idx;
    std::pair<apsi::util::AlgItem, size_t> data_with_idx;

    if (is_labeled) {
      datalabel_with_idx = DeserializeDataLabelWithIndices(std::string_view(
          reinterpret_cast<char *>(value.data()), value.size()));

      cuckoo_idx = datalabel_with_idx.second;
    } else {
      data_with_idx = DeserializeDataWithIndices(std::string_view(
          reinterpret_cast<char *>(value.data()), value.size()));
      cuckoo_idx = data_with_idx.second;
    }

    // const apsi::util::AlgItem &data = data_with_idx.first;

    // Get the bundle index
    size_t bin_idx, bundle_idx;
    std::tie(bin_idx, bundle_idx) =
        labeled_psi::UnpackCuckooIdx(cuckoo_idx, bins_per_bundle);

    // If the bundle_idx isn't in the prescribed range, don't try to insert
    // this data
    if (bundle_idx != bundle_index) {
      // Dealing with this bundle index is not our job
      continue;
    }

    // Try to insert or overwrite these field elements in an existing
    // BinBundle at this bundle index. Keep track of whether or not we
    // succeed.
    bool written = false;

    for (auto bundle_it = bundle_set.rbegin(); bundle_it != bundle_set.rend();
         bundle_it++) {
      // If we're supposed to overwrite, try to overwrite. One of these
      // BinBundles has to have the data we're trying to overwrite.
      if (overwrite) {
        // If we successfully overwrote, we're done with this bundle
        // written = bundle_it->try_multi_overwrite(data, bin_idx);
        if (is_labeled) {
          written =
              bundle_it->try_multi_overwrite(datalabel_with_idx.first, bin_idx);

        } else {
          written =
              bundle_it->try_multi_overwrite(data_with_idx.first, bin_idx);
        }

        if (written) {
          break;
        }
      }

      // Do a dry-run insertion and see if the new largest bin size in the
      // range exceeds the limit
      // int32_t new_largest_bin_size = bundle_it->multi_insert_dry_run(data,
      // bin_idx);
      int32_t new_largest_bin_size;
      if (is_labeled) {
        new_largest_bin_size =
            bundle_it->multi_insert_dry_run(datalabel_with_idx.first, bin_idx);

      } else {
        new_largest_bin_size =
            bundle_it->multi_insert_dry_run(data_with_idx.first, bin_idx);
      }

      // Check if inserting would violate the max bin size constraint
      if (new_largest_bin_size > 0 &&
          seal::util::safe_cast<size_t>(new_largest_bin_size) < max_bin_size) {
        // All good
        // bundle_it->multi_insert_for_real(data, bin_idx);
        if (is_labeled) {
          bundle_it->multi_insert_for_real(datalabel_with_idx.first, bin_idx);
        } else {
          bundle_it->multi_insert_for_real(data_with_idx.first, bin_idx);
        }
        written = true;
        break;
      }
    }

    // We tried to overwrite an item that doesn't exist. This should never
    // happen
    if (overwrite && !written) {
      SPDLOG_ERROR(
          "Insert-or-Assign worker: "
          "failed to overwrite item at bundle index {} because the item was "
          "not found",
          bundle_idx);
      SPU_THROW("tried to overwrite non-existent item");
    }

    // If we had conflicts everywhere when trying to insert, then we need to
    // make a new BinBundle and insert the data there
    if (!written) {
      // Make a fresh BinBundle and insert
      apsi::sender::BinBundle new_bin_bundle(
          crypto_context, label_size, max_bin_size, ps_low_degree,
          bins_per_bundle, compressed, false);

      // int res = new_bin_bundle.multi_insert_for_real(data, bin_idx);
      int res;
      if (is_labeled) {
        res = new_bin_bundle.multi_insert_for_real(datalabel_with_idx.first,
                                                   bin_idx);

      } else {
        res =
            new_bin_bundle.multi_insert_for_real(data_with_idx.first, bin_idx);
      }

      // If even that failed, I don't know what could've happened
      if (res < 0) {
        SPDLOG_ERROR(
            "Insert-or-Assign worker: "
            "failed to insert item into a new BinBundle at bundle index {}",
            bundle_idx);
        SPU_THROW("failed to insert item into a new BinBundle");
      }

      // Push a new BinBundle to the set of BinBundles at this bundle index
      bundle_set.push_back(std::move(new_bin_bundle));
    }
  }

  const auto db_save_start = std::chrono::system_clock::now();

  for (auto &bundle : bundle_set) {
    size_t store_idx = (*bundles_store_idx)[bundle_index]++;

    SPDLOG_INFO(
        "Polynomial Interpolate and HE Plaintext Encode, bundle_indx:{}, "
        "store_idx:{}",
        bundle_index, store_idx);

    // Generate the BinBundle caches
    // bundle.regen_cache();
    bundle.strip();

    std::stringstream stream;
    bundle.save(stream, store_idx);

    (*bundles_store)[bundle_index]->Put(store_idx, stream.str());
  }
  const auto db_save_end = std::chrono::system_clock::now();
  const DurationMillis db_save_duration = db_save_end - db_save_start;
  SPDLOG_INFO("*** step leveldb put duration:{}", db_save_duration.count());

  SPDLOG_DEBUG("Insert-or-Assign worker: finished processing bundle index {}",
               bundle_index);
}

/**
Takes algebraized data to be inserted, splits it up, and distributes it so
that thread_count many threads can all insert in parallel. If overwrite is
set, this will overwrite the labels if it finds an AlgItemLabel that matches
the input perfectly.
*/
template <typename T>
void DispatchInsertOrAssign(
    const std::vector<std::pair<T, size_t>> &data_with_indices,
    std::vector<std::shared_ptr<yacl::io::IndexStore>> *bundles_store,
    std::vector<size_t> *bundles_store_idx,
    const apsi::CryptoContext &crypto_context, uint32_t bins_per_bundle,
    size_t label_size, uint32_t max_bin_size, uint32_t ps_low_degree,
    bool overwrite, bool compressed) {
  apsi::ThreadPoolMgr tpm;

  // Collect the bundle indices and partition them into thread_count many
  // partitions. By some uniformity assumption, the number of things to insert
  // per partition should be roughly the same. Note that the contents of
  // bundle_indices is always sorted (increasing order).
  std::set<size_t> bundle_indices_set;
  for (auto &data_with_idx : data_with_indices) {
    size_t cuckoo_idx = data_with_idx.second;
    size_t bin_idx;
    size_t bundle_idx;
    std::tie(bin_idx, bundle_idx) =
        labeled_psi::UnpackCuckooIdx(cuckoo_idx, bins_per_bundle);
    bundle_indices_set.insert(bundle_idx);
  }

  // Copy the set of indices into a vector and sort so each thread processes a
  // range of indices
  std::vector<size_t> bundle_indices;
  bundle_indices.reserve(bundle_indices_set.size());
  std::copy(bundle_indices_set.begin(), bundle_indices_set.end(),
            std::back_inserter(bundle_indices));
  std::sort(bundle_indices.begin(), bundle_indices.end());

  // Run the threads on the partitions
  std::vector<std::future<void>> futures(bundle_indices.size());
  SPDLOG_INFO("Launching {} insert-or-assign worker tasks",
              bundle_indices.size());
  size_t future_idx = 0;
  for (auto &bundle_idx : bundle_indices) {
    futures[future_idx++] = tpm.thread_pool().enqueue([&, bundle_idx]() {
      InsertOrAssignWorker(data_with_indices, bundles_store, bundles_store_idx,
                           crypto_context, static_cast<uint32_t>(bundle_idx),
                           bins_per_bundle, label_size, max_bin_size,
                           ps_low_degree, overwrite, compressed);
    });
  }

  // Wait for the tasks to finish
  for (auto &f : futures) {
    f.get();
  }

  SPDLOG_INFO("Finished insert-or-assign worker tasks");
}

void DispatchInsertOrAssign(
    const std::shared_ptr<yacl::io::IndexStore> &indices_store,
    size_t indices_count, const std::set<size_t> &bundle_indices_set,
    std::vector<std::shared_ptr<yacl::io::IndexStore>> *bundles_store,
    std::vector<size_t> *bundles_store_idx, bool is_labeled,
    const apsi::CryptoContext &crypto_context, uint32_t bins_per_bundle,
    size_t label_size, uint32_t max_bin_size, uint32_t ps_low_degree,
    bool overwrite, bool compressed) {
  apsi::ThreadPoolMgr tpm;

  std::vector<size_t> bundle_indices;
  bundle_indices.reserve(bundle_indices_set.size());
  std::copy(bundle_indices_set.begin(), bundle_indices_set.end(),
            std::back_inserter(bundle_indices));
  std::sort(bundle_indices.begin(), bundle_indices.end());

  // Run the threads on the partitions
  std::vector<std::future<void>> futures(bundle_indices.size());
  SPDLOG_INFO("Launching {} insert-or-assign worker tasks",
              bundle_indices.size());
  size_t future_idx = 0;
  for (auto &bundle_idx : bundle_indices) {
    futures[future_idx++] = tpm.thread_pool().enqueue([&, bundle_idx]() {
      InsertOrAssignWorker(indices_store, indices_count, bundles_store,
                           bundles_store_idx, is_labeled, crypto_context,
                           static_cast<uint32_t>(bundle_idx), bins_per_bundle,
                           label_size, max_bin_size, ps_low_degree, overwrite,
                           compressed);
    });
  }

  // Wait for the tasks to finish
  for (auto &f : futures) {
    f.get();
  }

  SPDLOG_INFO("Finished insert-or-assign worker tasks");
}

constexpr char kMetaInfoStoreName[] = "db_meta_info";
constexpr char kServerDataCount[] = "server_data_count";

constexpr char kMemoryStoreFlag[] = "::memory";

}  // namespace

SenderKvDB::SenderKvDB(const apsi::PSIParams &params,
                       yacl::ByteContainerView oprf_key,
                       std::string_view kv_store_path, size_t label_byte_count,
                       size_t nonce_byte_count, bool compressed)
    : ISenderDB(params, oprf_key, label_byte_count, nonce_byte_count,
                compressed),
      kv_store_path_(kv_store_path) {
  // Reset the SenderDB data structures
  clear();

  bundles_store_.resize(params_.bundle_idx_count());
  bundles_store_idx_.resize(params_.bundle_idx_count());

  if (kv_store_path == kMemoryStoreFlag) {
    meta_info_store_ = std::make_shared<yacl::io::MemoryKVStore>();

    for (size_t i = 0; i < bundles_store_.size(); ++i) {
      std::shared_ptr<yacl::io::KVStore> kv_store =
          std::make_shared<yacl::io::MemoryKVStore>();

      bundles_store_[i] = std::make_shared<yacl::io::IndexStore>(kv_store);
    }
  } else {
    std::string meta_store_name =
        fmt::format("{}/{}", kv_store_path, kMetaInfoStoreName);

    meta_info_store_ =
        std::make_shared<yacl::io::LeveldbKVStore>(false, meta_store_name);

    for (size_t i = 0; i < bundles_store_.size(); ++i) {
      std::string bundle_store_name =
          fmt::format("{}/bundle_{}", kv_store_path_, i);

      std::shared_ptr<yacl::io::KVStore> kv_store =
          std::make_shared<yacl::io::LeveldbKVStore>(false, bundle_store_name);

      bundles_store_[i] = std::make_shared<yacl::io::IndexStore>(kv_store);
    }
  }

  try {
    yacl::Buffer temp_value;

    meta_info_store_->Get(kServerDataCount, &temp_value);
    item_count_ = std::stoul(std::string(std::string_view(
        reinterpret_cast<char *>(temp_value.data()), temp_value.size())));

    for (size_t i = 0; i < bundles_store_idx_.size(); ++i) {
      bundles_store_idx_[i] = bundles_store_[i]->Count();
    }

  } catch (const std::exception &e) {
    SPDLOG_INFO("key item_count no value");
  }
}

size_t SenderKvDB::GetBinBundleCount(uint32_t bundle_idx) const {
  // Lock the database for reading
  auto lock = GetReaderLock();

  // return bin_bundles_.at(seal::util::safe_cast<size_t>(bundle_idx)).size();
  return bundles_store_idx_[seal::util::safe_cast<size_t>(bundle_idx)];
}

size_t SenderKvDB::GetBinBundleCount() const {
  // Lock the database for reading
  auto lock = GetReaderLock();

  // Compute the total number of BinBundles
  // return std::accumulate(bin_bundles_.cbegin(), bin_bundles_.cend(),
  // static_cast<size_t>(0),
  //                       [&](auto &a, auto &b) { return a + b.size(); });
  return std::accumulate(bundles_store_idx_.cbegin(), bundles_store_idx_.cend(),
                         static_cast<size_t>(0),
                         [&](auto &a, auto &b) { return a + b; });
}

void SenderKvDB::ClearInternal() {
  item_count_ = 0;

  // Reset the stripped_ flag
  stripped_ = false;
  // TODO(changjun): delete kv store
}

void SenderKvDB::clear() {
  // Lock the database for writing
  auto lock = GetWriterLock();

  ClearInternal();
}

void SenderKvDB::GenerateCaches() {
  STOPWATCH(sender_stopwatch, "SenderDB::GenerateCaches");
  SPDLOG_INFO("Start generating bin bundle caches");

  SPDLOG_INFO("Finished generating bin bundle caches");
}

std::shared_ptr<apsi::sender::BinBundle> SenderKvDB::GetBinBundleAt(
    uint32_t bundle_idx, size_t cache_idx) {
  yacl::Buffer value;

  bool get_status = bundles_store_[bundle_idx]->Get(cache_idx, &value);

  SPU_ENFORCE(get_status);

  size_t label_size = labeled_psi::ComputeLabelSize(
      nonce_byte_count_ + label_byte_count_, params_);

  uint32_t bins_per_bundle = params_.bins_per_bundle();
  uint32_t max_bin_size = params_.table_params().max_items_per_bin;
  uint32_t ps_low_degree = params_.query_params().ps_low_degree;

  bool compressed = false;

  std::shared_ptr<apsi::sender::BinBundle> load_bin_bundle =
      std::make_shared<apsi::sender::BinBundle>(
          crypto_context_, label_size, max_bin_size, ps_low_degree,
          bins_per_bundle, compressed, false);

  gsl::span<unsigned char> value_span = {
      reinterpret_cast<unsigned char *>(value.data()),
      gsl::narrow_cast<gsl::span<unsigned char>::size_type>(value.size())};
  std::pair<std::uint32_t, std::size_t> load_ret =
      load_bin_bundle->load(value_span);

  SPU_ENFORCE(load_ret.first == cache_idx);

  // check cache is valid
  if (load_bin_bundle->cache_invalid()) {
    load_bin_bundle->regen_cache();
  }

  return load_bin_bundle;
}

void SenderKvDB::strip() {
  // Lock the database for writing
  auto lock = GetWriterLock();

  stripped_ = true;

  memset(oprf_key_.data(), 0, oprf_key_.size());

  SPDLOG_INFO("SenderDB has been stripped");
}

void SenderKvDB::InsertOrAssign(
    const std::shared_ptr<IBatchProvider> &batch_provider, size_t batch_size) {
  size_t batch_count = 0;
  size_t indices_count = 0;

  std::shared_ptr<yacl::io::KVStore> kv_store;

  if (kv_store_path_ == kMemoryStoreFlag) {
    kv_store = std::make_shared<yacl::io::MemoryKVStore>();
  } else {
    kv_store = std::make_shared<yacl::io::LeveldbKVStore>(true);
  }

  std::shared_ptr<yacl::io::IndexStore> items_oprf_store =
      std::make_shared<yacl::io::IndexStore>(kv_store);

  std::set<size_t> bundle_indices_set;

  // Dispatch the insertion
  uint32_t bins_per_bundle = params_.bins_per_bundle();
  uint32_t max_bin_size = params_.table_params().max_items_per_bin;
  uint32_t ps_low_degree = params_.query_params().ps_low_degree;

  while (true) {
    std::vector<std::string> batch_items;
    std::vector<std::string> batch_labels;

    if (IsLabeled()) {
      std::tie(batch_items, batch_labels) =
          batch_provider->ReadNextBatchWithLabel(batch_size);
    } else {
      batch_items = batch_provider->ReadNextBatch(batch_size);
    }

    if (batch_items.empty()) {
      SPDLOG_INFO(
          "OPRF FullEvaluate and EncryptLabel Last batch count: "
          "{}, item_count:{}",
          batch_count, item_count_);
      break;
    }
    SPDLOG_INFO("OPRF FullEvaluate and EncryptLabel batch_count: {}",
                batch_count);

    std::vector<std::string> oprf_out = oprf_server_->FullEvaluate(batch_items);

    std::vector<std::vector<std::pair<apsi::util::AlgItemLabel, size_t>>>
        data_with_indices_vec;

    if (IsLabeled()) {
      data_with_indices_vec.resize(oprf_out.size());
      size_t key_offset_pos = sizeof(apsi::Item::value_type);

      // Set up Kuku hash functions
      auto hash_funcs = labeled_psi::HashFunctions(params_);

      yacl::parallel_for(
          0, oprf_out.size(), 1, [&](int64_t begin, int64_t end) {
            for (int64_t idx = begin; idx < end; ++idx) {
              apsi::Item::value_type value{};
              std::memcpy(value.data(), &oprf_out[idx][0], value.size());

              apsi::HashedItem hashed_item(value);

              apsi::LabelKey key;
              std::memcpy(key.data(), &oprf_out[idx][key_offset_pos],
                          apsi::label_key_byte_count);

              apsi::Label label_with_padding =
                  PaddingData(batch_labels[idx], label_byte_count_);

              apsi::EncryptedLabel encrypted_label = apsi::util::encrypt_label(
                  label_with_padding, key, label_byte_count_,
                  nonce_byte_count_);

              std::pair<apsi::HashedItem, apsi::EncryptedLabel>
                  item_label_pair =
                      std::make_pair(hashed_item, encrypted_label);

              data_with_indices_vec[idx] =
                  PreprocessLabeledData(item_label_pair, params_, hash_funcs);
            }
          });

      for (size_t i = 0; i < oprf_out.size(); ++i) {
        for (size_t j = 0; j < data_with_indices_vec[i].size(); ++j) {
          std::string indices_buffer =
              SerializeDataLabelWithIndices(data_with_indices_vec[i][j]);

          items_oprf_store->Put(indices_count + j, indices_buffer);

          size_t cuckoo_idx = data_with_indices_vec[i][j].second;
          size_t bin_idx, bundle_idx;
          std::tie(bin_idx, bundle_idx) =
              labeled_psi::UnpackCuckooIdx(cuckoo_idx, bins_per_bundle);
          bundle_indices_set.insert(bundle_idx);
        }

        indices_count += data_with_indices_vec[i].size();
      }

    } else {
      for (size_t i = 0; i < oprf_out.size(); ++i) {
        //
        apsi::Item::value_type value{};
        std::memcpy(value.data(), &oprf_out[i][0], value.size());

        apsi::HashedItem hashed_item(value);

        std::vector<std::pair<apsi::util::AlgItem, size_t>> data_with_indices =
            PreprocessUnlabeledData(hashed_item, params_);

        for (size_t j = 0; j < data_with_indices.size(); ++j) {
          std::string indices_buffer =
              SerializeDataWithIndices(data_with_indices[j]);

          items_oprf_store->Put(indices_count + j, indices_buffer);

          size_t cuckoo_idx = data_with_indices[j].second;

          size_t bin_idx, bundle_idx;
          std::tie(bin_idx, bundle_idx) =
              labeled_psi::UnpackCuckooIdx(cuckoo_idx, bins_per_bundle);

          bundle_indices_set.insert(bundle_idx);
        }
        indices_count += data_with_indices.size();
      }
    }
    item_count_ += batch_items.size();

    batch_count++;
  }
  meta_info_store_->Put(kServerDataCount, std::to_string(item_count_));

  size_t label_size = 0;
  if (IsLabeled()) {
    label_size = labeled_psi::ComputeLabelSize(
        nonce_byte_count_ + label_byte_count_, params_);
  }

  DispatchInsertOrAssign(
      items_oprf_store, indices_count, bundle_indices_set, &bundles_store_,
      &bundles_store_idx_, IsLabeled(), crypto_context_, bins_per_bundle,
      label_size,                         /* label size */
      max_bin_size, ps_low_degree, false, /* don't overwrite items */
      compressed_);

  SPDLOG_INFO("Finished inserting {} items in SenderDB", item_count_);
}

}  // namespace spu::psi
