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
#include "libspu/psi/core/labeled_psi/sender_db.h"
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

/**
Creates and returns the vector of hash functions similarly to how Kuku 2.x sets
them internally.
*/
std::vector<kuku::LocFunc> HashFunctions(const apsi::PSIParams &params) {
  std::vector<kuku::LocFunc> result;
  for (uint32_t i = 0; i < params.table_params().hash_func_count; i++) {
    result.emplace_back(params.table_params().table_size,
                        kuku::make_item(i, 0));
  }

  return result;
}

/**
Computes all cuckoo hash table locations for a given item.
*/
std::unordered_set<kuku::location_type> AllLocations(
    const std::vector<kuku::LocFunc> &hash_funcs,
    const apsi::HashedItem &item) {
  std::unordered_set<kuku::location_type> result;
  for (const auto &hf : hash_funcs) {
    result.emplace(hf(item.get_as<kuku::item_type>().front()));
  }

  return result;
}

/**
Compute the label size in multiples of item-size chunks.
*/
size_t ComputeLabelSize(size_t label_byte_count,
                        const apsi::PSIParams &params) {
  return (label_byte_count * 8 + params.item_bit_count() - 1) /
         params.item_bit_count();
}

/**
Unpacks a cuckoo idx into its bin and bundle indices
*/
std::pair<size_t, size_t> UnpackCuckooIdx(size_t cuckoo_idx,
                                          size_t bins_per_bundle) {
  // Recall that bin indices are relative to the bundle index. That is, the
  // first bin index of a bundle at bundle index 5 is 0. A cuckoo index is
  // similar, except it is not relative to the bundle index. It just keeps
  // counting past bundle boundaries. So in order to get the bin index from the
  // cuckoo index, just compute cuckoo_idx (mod bins_per_bundle).
  size_t bin_idx = cuckoo_idx % bins_per_bundle;

  // Compute which bundle index this cuckoo index belongs to
  size_t bundle_idx = (cuckoo_idx - bin_idx) / bins_per_bundle;

  return {bin_idx, bundle_idx};
}

/**
Converts each given Item-Label pair in between the given iterators into its
algebraic form, i.e., a sequence of felt-felt pairs. Also computes each Item's
cuckoo index.
*/
std::vector<std::pair<apsi::util::AlgItemLabel, size_t>> PreprocessLabeledData(
    const std::vector<std::pair<apsi::HashedItem,
                                apsi::EncryptedLabel>>::const_iterator begin,
    const std::vector<
        std::pair<apsi::HashedItem, apsi::EncryptedLabel>>::const_iterator end,
    const apsi::PSIParams &params) {
  STOPWATCH(sender_stopwatch, "preprocess_labeled_data");
  SPDLOG_DEBUG("Start preprocessing {} labeled items",
               std::distance(begin, end));

  // Some variables we'll need
  size_t bins_per_item = params.item_params().felts_per_item;
  size_t item_bit_count = params.item_bit_count();

  // Set up Kuku hash functions
  auto hash_funcs = HashFunctions(params);

  // Calculate the cuckoo indices for each item. Store every pair of
  // (item-label, cuckoo_idx) in a vector. Later, we're gonna sort this vector
  // by cuckoo_idx and use the result to parallelize the work of inserting the
  // items into BinBundles.
  std::vector<std::pair<apsi::util::AlgItemLabel, size_t>> data_with_indices;
  for (auto it = begin; it != end; it++) {
    const std::pair<apsi::HashedItem, apsi::EncryptedLabel> &item_label_pair =
        *it;

    // Serialize the data into field elements
    const apsi::HashedItem &item = item_label_pair.first;
    const apsi::EncryptedLabel &label = item_label_pair.second;
    apsi::util::AlgItemLabel alg_item_label = algebraize_item_label(
        item, label, item_bit_count, params.seal_params().plain_modulus());

    // Get the cuckoo table locations for this item and add to data_with_indices
    for (auto location : AllLocations(hash_funcs, item)) {
      // The current hash value is an index into a table of Items. In reality
      // our BinBundles are tables of bins, which contain chunks of items. How
      // many chunks? bins_per_item many chunks
      size_t bin_idx = location * bins_per_item;

      // Store the data along with its index
      data_with_indices.emplace_back(alg_item_label, bin_idx);
    }
  }

  SPDLOG_DEBUG("Finished preprocessing {} labeled items",
               std::distance(begin, end));

  return data_with_indices;
}

/**
Converts each given Item into its algebraic form, i.e., a sequence of
felt-monostate pairs. Also computes each Item's cuckoo index.
*/
std::vector<std::pair<apsi::util::AlgItem, size_t>> PreprocessUnlabeledData(
    const std::vector<apsi::HashedItem>::const_iterator begin,
    const std::vector<apsi::HashedItem>::const_iterator end,
    const apsi::PSIParams &params) {
  STOPWATCH(sender_stopwatch, "preprocess_unlabeled_data");
  SPDLOG_DEBUG("Start preprocessing {} unlabeled items",
               std::distance(begin, end));

  // Some variables we'll need
  size_t bins_per_item = params.item_params().felts_per_item;
  size_t item_bit_count = params.item_bit_count();

  // Set up Kuku hash functions
  auto hash_funcs = HashFunctions(params);

  // Calculate the cuckoo indices for each item. Store every pair of
  // (item-label, cuckoo_idx) in a vector. Later, we're gonna sort this vector
  // by cuckoo_idx and use the result to parallelize the work of inserting the
  // items into BinBundles.
  std::vector<std::pair<apsi::util::AlgItem, size_t>> data_with_indices;
  for (auto it = begin; it != end; it++) {
    const apsi::HashedItem &item = *it;

    // Serialize the data into field elements
    apsi::util::AlgItem alg_item = algebraize_item(
        item, item_bit_count, params.seal_params().plain_modulus());

    // Get the cuckoo table locations for this item and add to data_with_indices
    for (auto location : AllLocations(hash_funcs, item)) {
      // The current hash value is an index into a table of Items. In reality
      // our BinBundles are tables of bins, which contain chunks of items. How
      // many chunks? bins_per_item many chunks
      size_t bin_idx = location * bins_per_item;

      // Store the data along with its index
      data_with_indices.emplace_back(alg_item, bin_idx);
    }
  }

  SPDLOG_DEBUG("Finished preprocessing {} unlabeled items",
               std::distance(begin, end));

  return data_with_indices;
}

std::vector<std::pair<apsi::util::AlgItemLabel, size_t>> PreprocessLabeledData(
    const std::pair<apsi::HashedItem, apsi::EncryptedLabel> &item_label_pair,
    const apsi::PSIParams &params) {
  SPDLOG_DEBUG("Start preprocessing {} labeled items", distance(begin, end));

  // Some variables we'll need
  size_t bins_per_item = params.item_params().felts_per_item;
  size_t item_bit_count = params.item_bit_count();

  // Set up Kuku hash functions
  auto hash_funcs = HashFunctions(params);

  std::vector<std::pair<apsi::util::AlgItemLabel, size_t>> data_with_indices;

  // Serialize the data into field elements
  const apsi::HashedItem &item = item_label_pair.first;
  const apsi::EncryptedLabel &label = item_label_pair.second;
  apsi::util::AlgItemLabel alg_item_label = algebraize_item_label(
      item, label, item_bit_count, params.seal_params().plain_modulus());

  // Get the cuckoo table locations for this item and add to data_with_indices
  for (auto location : AllLocations(hash_funcs, item)) {
    // The current hash value is an index into a table of Items. In reality
    // our BinBundles are tables of bins, which contain chunks of items. How
    // many chunks? bins_per_item many chunks
    size_t bin_idx = location * bins_per_item;

    // Store the data along with its index
    data_with_indices.push_back(std::make_pair(alg_item_label, bin_idx));
  }

  return data_with_indices;
}

std::vector<std::pair<apsi::util::AlgItem, size_t>> PreprocessUnlabeledData(
    const apsi::HashedItem &hashed_item, const apsi::PSIParams &params) {
  // Some variables we'll need
  size_t bins_per_item = params.item_params().felts_per_item;
  size_t item_bit_count = params.item_bit_count();

  // Set up Kuku hash functions
  auto hash_funcs = HashFunctions(params);

  std::vector<std::pair<apsi::util::AlgItem, size_t>> data_with_indices;

  // Serialize the data into field elements
  apsi::util::AlgItem alg_item = algebraize_item(
      hashed_item, item_bit_count, params.seal_params().plain_modulus());

  // Get the cuckoo table locations for this item and add to data_with_indices
  for (auto location : AllLocations(hash_funcs, hashed_item)) {
    // The current hash value is an index into a table of Items. In reality
    // our BinBundles are tables of bins, which contain chunks of items. How
    // many chunks? bins_per_item many chunks
    size_t bin_idx = location * bins_per_item;

    // Store the data along with its index
    data_with_indices.emplace_back(std::make_pair(alg_item, bin_idx));
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
        UnpackCuckooIdx(cuckoo_idx, bins_per_bundle);

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
    std::stringstream stream;

    // Generate the BinBundle caches
    // bundle.regen_cache();
    bundle.strip();

    size_t store_idx = (*bundles_store_idx)[bundle_index]++;
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
        UnpackCuckooIdx(cuckoo_idx, bins_per_bundle);

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
    std::stringstream stream;

    // Generate the BinBundle caches
    // bundle.regen_cache();
    bundle.strip();

    size_t store_idx = (*bundles_store_idx)[bundle_index]++;
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
        UnpackCuckooIdx(cuckoo_idx, bins_per_bundle);
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

constexpr char kMetaInfoStoreName[] = "meta_info";
constexpr char kServerDataCount[] = "server_data_count";

constexpr char kMemoryStoreFlag[] = "::memory";

}  // namespace

SenderDB::SenderDB(const apsi::PSIParams &params,
                   std::string_view kv_store_path, size_t label_byte_count,
                   size_t nonce_byte_count, bool compressed)
    : params_(params),
      crypto_context_(params_),
      label_byte_count_(label_byte_count),
      nonce_byte_count_(label_byte_count_ != 0 ? nonce_byte_count : 0),
      item_count_(0),
      compressed_(compressed),
      kv_store_path_(kv_store_path) {
  // The labels cannot be more than 1 KB.
  if (label_byte_count_ > 1024) {
    SPDLOG_ERROR("Requested label byte count {} exceeds the maximum (1024)",
                 label_byte_count_);

    SPU_THROW("label_byte_count is too large");
  }

  if (nonce_byte_count_ > apsi::max_nonce_byte_count) {
    SPDLOG_ERROR("Request nonce byte count {} exceeds the maximum ({}) ",
                 nonce_byte_count_, apsi::max_nonce_byte_count);
    SPU_THROW("nonce_byte_count is too large");
  }

  // If the nonce byte count is less than max_nonce_byte_count, print a warning;
  // this is a labeled SenderDB but may not be safe to use for arbitrary label
  // changes.
  if ((label_byte_count_ != 0) &&
      nonce_byte_count_ < apsi::max_nonce_byte_count) {
    SPDLOG_WARN(
        "You have instantiated a labeled SenderDB instance with a nonce byte "
        "count {} , which is less than the safe default value {} . Updating "
        "labels for existing items in the SenderDB or removing and reinserting "
        "items with different labels may leak information about the labels.",
        nonce_byte_count_, apsi::max_nonce_byte_count);
  }

  // Set the evaluator. This will be used for BatchedPlaintextPolyn::eval.
  crypto_context_.set_evaluator();

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

SenderDB::SenderDB(const apsi::PSIParams &params,
                   yacl::ByteContainerView oprf_key,
                   std::string_view kv_store_path, size_t label_byte_count,
                   size_t nonce_byte_count, bool compressed)
    : SenderDB(params, kv_store_path, label_byte_count, nonce_byte_count,
               compressed) {
  oprf_key_.resize(oprf_key.size());
  std::memcpy(oprf_key_.data(), oprf_key.data(), oprf_key.size());

  oprf_server_ =
      CreateEcdhOprfServer(oprf_key, OprfType::Basic, CurveType::CURVE_FOURQ);
  oprf_server_->SetCompareLength(kEccKeySize);
}

SenderDB::SenderDB(SenderDB &&source) noexcept
    : params_(source.params_),
      crypto_context_(source.crypto_context_),
      label_byte_count_(source.label_byte_count_),
      nonce_byte_count_(source.nonce_byte_count_),
      item_count_(source.item_count_),
      compressed_(source.compressed_),
      stripped_(source.stripped_) {
  // Lock the source before moving stuff over
  auto lock = source.GetWriterLock();

  hashed_items_ = std::move(source.hashed_items_);
  // bin_bundles_ = std::move(source.bin_bundles_);

  std::vector<uint8_t> oprf_key = source.GetOprfKey();
  oprf_key_.resize(oprf_key.size());
  std::memcpy(oprf_key_.data(), oprf_key.data(), oprf_key.size());

  oprf_server_ =
      CreateEcdhOprfServer(oprf_key, OprfType::Basic, CurveType::CURVE_FOURQ);

  // Reset the source data structures
  source.ClearInternal();
}

SenderDB &SenderDB::operator=(SenderDB &&source) noexcept {
  // Do nothing if moving to self
  if (&source == this) {
    return *this;
  }

  // Lock the current SenderDB
  auto this_lock = GetWriterLock();

  params_ = source.params_;
  crypto_context_ = source.crypto_context_;
  label_byte_count_ = source.label_byte_count_;
  nonce_byte_count_ = source.nonce_byte_count_;
  item_count_ = source.item_count_;
  compressed_ = source.compressed_;
  stripped_ = source.stripped_;

  // Lock the source before moving stuff over
  auto source_lock = source.GetWriterLock();

  hashed_items_ = std::move(source.hashed_items_);
  // bin_bundles_ = std::move(source.bin_bundles_);

  std::vector<uint8_t> oprf_key = source.GetOprfKey();
  oprf_key_.resize(oprf_key.size());
  std::memcpy(oprf_key_.data(), oprf_key.data(), oprf_key.size());

  oprf_server_ =
      CreateEcdhOprfServer(oprf_key, OprfType::Basic, CurveType::CURVE_FOURQ);

  // Reset the source data structures
  source.ClearInternal();

  return *this;
}

size_t SenderDB::GetBinBundleCount(uint32_t bundle_idx) const {
  // Lock the database for reading
  auto lock = GetReaderLock();

  // return bin_bundles_.at(seal::util::safe_cast<size_t>(bundle_idx)).size();
  return bundles_store_idx_[seal::util::safe_cast<size_t>(bundle_idx)];
}

size_t SenderDB::GetBinBundleCount() const {
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

double SenderDB::GetPackingRate() const {
  // Lock the database for reading
  auto lock = GetReaderLock();

  uint64_t item_count = seal::util::mul_safe(
      static_cast<uint64_t>(GetItemCount()),
      static_cast<uint64_t>(params_.table_params().hash_func_count));
  uint64_t max_item_count = seal::util::mul_safe(
      static_cast<uint64_t>(GetBinBundleCount()),
      static_cast<uint64_t>(params_.items_per_bundle()),
      static_cast<uint64_t>(params_.table_params().max_items_per_bin));

  return max_item_count != 0 ? static_cast<double>(item_count) /
                                   static_cast<double>(max_item_count)
                             : 0.0;
}

void SenderDB::ClearInternal() {
  // Assume the SenderDB is already locked for writing

  // Clear the set of inserted items
  hashed_items_.clear();
  item_count_ = 0;

  // Reset the stripped_ flag
  stripped_ = false;
}

void SenderDB::clear() {
  if (!hashed_items_.empty()) {
    SPDLOG_INFO("Removing {} items pairs from SenderDB", hashed_items_.size());
  }

  // Lock the database for writing
  auto lock = GetWriterLock();

  ClearInternal();
}

void SenderDB::GenerateCaches() {
  STOPWATCH(sender_stopwatch, "SenderDB::GenerateCaches");
  SPDLOG_INFO("Start generating bin bundle caches");

  SPDLOG_INFO("Finished generating bin bundle caches");
}

std::shared_ptr<apsi::sender::BinBundle> SenderDB::GetCacheAt(
    uint32_t bundle_idx, size_t cache_idx) {
  yacl::Buffer value;

  bool get_status = bundles_store_[bundle_idx]->Get(cache_idx, &value);

  SPU_ENFORCE(get_status);

  size_t label_size =
      ComputeLabelSize(nonce_byte_count_ + label_byte_count_, params_);

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

void SenderDB::strip() {
  // Lock the database for writing
  auto lock = GetWriterLock();

  stripped_ = true;

  memset(oprf_key_.data(), 0, oprf_key_.size());
  hashed_items_.clear();

  apsi::ThreadPoolMgr tpm;

  SPDLOG_INFO("SenderDB has been stripped");
}

std::vector<uint8_t> SenderDB::GetOprfKey() const {
  if (stripped_) {
    SPDLOG_ERROR("Cannot return the OPRF key from a stripped SenderDB");
    SPU_THROW("failed to return OPRF key");
  }
  return oprf_key_;
}

void SenderDB::InsertOrAssign(
    const std::vector<std::pair<apsi::Item, apsi::Label>> &data) {
  if (stripped_) {
    SPDLOG_ERROR("Cannot insert data to a stripped SenderDB");
    SPU_THROW("failed to insert data");
  }
  if (!IsLabeled()) {
    SPDLOG_ERROR(
        "Attempted to insert labeled data but this is an unlabeled SenderDB");
    SPU_THROW("failed to insert data");
  }

  SPDLOG_INFO("Start inserting {} items in SenderDB", data.size());

  // First compute the hashes for the input data
  // auto hashed_data = OPRFSender::ComputeHashes(
  //    data, oprf_key_, label_byte_count_, nonce_byte_count_);
  std::vector<std::string> data_str(data.size());
  for (size_t i = 0; i < data.size(); ++i) {
    std::string temp_str(data[i].first.value().size(), '\0');

    std::memcpy(temp_str.data(), data[i].first.value().data(),
                data[i].first.value().size());
    data_str[i] = temp_str;
  }
  std::vector<std::string> oprf_out = oprf_server_->FullEvaluate(data_str);
  std::vector<std::pair<apsi::HashedItem, apsi::EncryptedLabel>> hashed_data;
  for (size_t i = 0; i < oprf_out.size(); ++i) {
    apsi::HashedItem hashed_item;
    std::memcpy(hashed_item.value().data(), oprf_out[i].data(),
                hashed_item.value().size());

    apsi::LabelKey key;
    std::memcpy(key.data(), &oprf_out[i][hashed_item.value().size()],
                key.size());

    apsi::EncryptedLabel encrypted_label = apsi::util::encrypt_label(
        data[i].second, key, label_byte_count_, nonce_byte_count_);

    hashed_data.emplace_back(hashed_item, encrypted_label);
  }

  // Lock the database for writing
  auto lock = GetWriterLock();

  // We need to know which items are new and which are old, since we have to
  // tell dispatch_insert_or_assign when to have an overwrite-on-collision
  // versus add-binbundle-on-collision policy.
  auto new_data_end = std::remove_if(
      hashed_data.begin(), hashed_data.end(), [&](const auto &item_label_pair) {
        bool found =
            hashed_items_.find(item_label_pair.first) != hashed_items_.end();
        if (!found) {
          // Add to hashed_items_ already at this point!
          hashed_items_.insert(item_label_pair.first);
          item_count_++;
        }

        // Remove those that were found
        return found;
      });

  // Dispatch the insertion, first for the new data, then for the data we're
  // gonna overwrite
  uint32_t bins_per_bundle = params_.bins_per_bundle();
  uint32_t max_bin_size = params_.table_params().max_items_per_bin;
  uint32_t ps_low_degree = params_.query_params().ps_low_degree;

  // Compute the label size; this ceil(effective_label_bit_count /
  // item_bit_count)
  size_t label_size =
      ComputeLabelSize(nonce_byte_count_ + label_byte_count_, params_);

  auto new_item_count = std::distance(hashed_data.begin(), new_data_end);
  auto existing_item_count = std::distance(new_data_end, hashed_data.end());

  if (existing_item_count != 0) {
    SPDLOG_INFO("Found {} existing items to replace in SenderDB",
                existing_item_count);

    // Break the data into field element representation. Also compute the
    // items' cuckoo indices.
    std::vector<std::pair<apsi::util::AlgItemLabel, size_t>> data_with_indices =
        PreprocessLabeledData(new_data_end, hashed_data.end(), params_);

    DispatchInsertOrAssign(data_with_indices, &bundles_store_,
                           &bundles_store_idx_, crypto_context_,
                           bins_per_bundle, label_size, max_bin_size,
                           ps_low_degree, true, /* overwrite items */
                           compressed_);

    // Release memory that is no longer needed
    hashed_data.erase(new_data_end, hashed_data.end());
  }

  if (new_item_count != 0) {
    SPDLOG_INFO("Found {} new items to insert in SenderDB", new_item_count);

    // Process and add the new data. Break the data into field element
    // representation. Also compute the items' cuckoo indices.
    std::vector<std::pair<apsi::util::AlgItemLabel, size_t>> data_with_indices =
        PreprocessLabeledData(hashed_data.begin(), hashed_data.end(), params_);

    DispatchInsertOrAssign(data_with_indices, &bundles_store_,
                           &bundles_store_idx_, crypto_context_,
                           bins_per_bundle, label_size, max_bin_size,
                           ps_low_degree, false, /* don't overwrite items */
                           compressed_);
  }

  SPDLOG_INFO("Finished inserting {} items in SenderDB", data.size());
}

void SenderDB::InsertOrAssign(const std::vector<apsi::Item> &data) {
  if (stripped_) {
    SPDLOG_ERROR("Cannot insert data to a stripped SenderDB");
    SPU_THROW("failed to insert data");
  }
  if (IsLabeled()) {
    SPDLOG_ERROR(
        "Attempted to insert unlabeled data but this is a labeled SenderDB");
    SPU_THROW("failed to insert data");
  }

  STOPWATCH(sender_stopwatch, "SenderDB::insert_or_assign (unlabeled)");
  SPDLOG_INFO("Start inserting {} items in SenderDB", data.size());

  // First compute the hashes for the input data
  // auto hashed_data = OPRFSender::ComputeHashes(data, oprf_key_);
  std::vector<std::string> data_str(data.size());
  for (size_t i = 0; i < data.size(); ++i) {
    std::string item_str(data[i].value().size(), '\0');
    std::memcpy(item_str.data(), data[i].value().data(),
                data[i].value().size());
    data_str[i] = item_str;
  }
  std::vector<std::string> oprf_out = oprf_server_->FullEvaluate(data_str);
  std::vector<apsi::HashedItem> hashed_data;
  for (const auto &out : oprf_out) {
    apsi::Item::value_type value{};
    std::memcpy(value.data(), out.data(), value.size());

    hashed_data.emplace_back(value);
  }

  // Lock the database for writing
  auto lock = GetWriterLock();

  // We are not going to insert items that already appear in the database.
  auto new_data_end = std::remove_if(
      hashed_data.begin(), hashed_data.end(), [&](const auto &item) {
        bool found = hashed_items_.find(item) != hashed_items_.end();
        if (!found) {
          // Add to hashed_items_ already at this point!
          hashed_items_.insert(item);
          item_count_++;
        }

        // Remove those that were found
        return found;
      });

  // Erase the previously existing items from hashed_data; in unlabeled case
  // there is nothing to do
  hashed_data.erase(new_data_end, hashed_data.end());

  SPDLOG_INFO("Found {} new items to insert in SenderDB", hashed_data.size());

  // Break the new data down into its field element representation. Also
  // compute the items' cuckoo indices.
  std::vector<std::pair<apsi::util::AlgItem, size_t>> data_with_indices =
      PreprocessUnlabeledData(hashed_data.begin(), hashed_data.end(), params_);

  // Dispatch the insertion
  uint32_t bins_per_bundle = params_.bins_per_bundle();
  uint32_t max_bin_size = params_.table_params().max_items_per_bin;
  uint32_t ps_low_degree = params_.query_params().ps_low_degree;

  DispatchInsertOrAssign(
      data_with_indices, &bundles_store_, &bundles_store_idx_, crypto_context_,
      bins_per_bundle, 0,                 /* label size */
      max_bin_size, ps_low_degree, false, /* don't overwrite items */
      compressed_);

  // Generate the BinBundle caches
  GenerateCaches();

  SPDLOG_INFO("Finished inserting {} items in SenderDB", data.size());
}

void SenderDB::InsertOrAssign(
    const std::shared_ptr<IBatchProvider> &batch_provider, size_t batch_size) {
  [[maybe_unused]] size_t batch_count = 0;
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
      break;
    }

    std::vector<std::string> oprf_out = oprf_server_->FullEvaluate(batch_items);

    std::vector<std::vector<std::pair<apsi::util::AlgItemLabel, size_t>>>
        data_with_indices_vec;

    if (IsLabeled()) {
      data_with_indices_vec.resize(oprf_out.size());
      size_t key_offset_pos = sizeof(apsi::Item::value_type);

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
                  PreprocessLabeledData(item_label_pair, params_);
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
              UnpackCuckooIdx(cuckoo_idx, bins_per_bundle);
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
              UnpackCuckooIdx(cuckoo_idx, bins_per_bundle);

          bundle_indices_set.insert(bundle_idx);
        }
        indices_count += data_with_indices.size();
      }
    }
    item_count_ += batch_items.size();

    batch_count++;
  }

  size_t label_size = 0;
  if (IsLabeled()) {
    label_size =
        ComputeLabelSize(nonce_byte_count_ + label_byte_count_, params_);
  }

  DispatchInsertOrAssign(
      items_oprf_store, indices_count, bundle_indices_set, &bundles_store_,
      &bundles_store_idx_, IsLabeled(), crypto_context_, bins_per_bundle,
      label_size,                         /* label size */
      max_bin_size, ps_low_degree, false, /* don't overwrite items */
      compressed_);

  SPDLOG_INFO("Finished inserting {} items in SenderDB", indices_count);
}

bool SenderDB::HasItem(const apsi::Item &item) const {
  if (stripped_) {
    SPDLOG_ERROR(
        "Cannot retrieve the presence of an item from a stripped SenderDB");
    SPU_THROW("failed to retrieve the presence of item");
  }

  // First compute the hash for the input item
  // auto hashed_item = OPRFSender::ComputeHashes({&item, 1}, oprf_key_)[0];
  std::string item_str;
  item_str.reserve(item.value().size());
  std::memcpy(item_str.data(), item.value().data(), item.value().size());
  std::string oprf_out = oprf_server_->FullEvaluate(item_str);
  apsi::HashedItem hashed_item;
  std::memcpy(hashed_item.value().data(), oprf_out.data(),
              hashed_item.value().size());

  // Lock the database for reading
  auto lock = GetReaderLock();

  return hashed_items_.find(hashed_item) != hashed_items_.end();
}

}  // namespace spu::psi
