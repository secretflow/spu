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

#include "libspu/psi/core/ecdh_oprf/ecdh_oprf_selector.h"
#include "libspu/psi/core/labeled_psi/sender_db.h"

// Kuku
#include "kuku/locfunc.h"

// SEAL
#include "absl/strings/escaping.h"
#include "seal/util/common.h"
#include "seal/util/streambuf.h"

namespace spu::psi {

namespace {
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
  for (auto &hf : hash_funcs) {
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
  SPDLOG_DEBUG("Start preprocessing {} labeled items", distance(begin, end));

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
      data_with_indices.push_back(std::make_pair(alg_item_label, bin_idx));
    }
  }

  SPDLOG_DEBUG("Finished preprocessing {} labeled items", distance(begin, end));

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
  SPDLOG_DEBUG("Start preprocessing {} unlabeled items", distance(begin, end));

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
      data_with_indices.emplace_back(std::make_pair(alg_item, bin_idx));
    }
  }

  SPDLOG_DEBUG("Finished preprocessing {} unlabeled items",
               distance(begin, end));

  return data_with_indices;
}

/**
Converts given Item into its algebraic form, i.e., a sequence of felt-monostate
pairs. Also computes the Item's cuckoo index.
*/
std::vector<std::pair<apsi::util::AlgItem, size_t>> PreprocessUnlabeledData(
    const apsi::HashedItem &item, const apsi::PSIParams &params) {
  std::vector<apsi::HashedItem> item_singleton{item};
  return PreprocessUnlabeledData(item_singleton.begin(), item_singleton.end(),
                                 params);
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
    std::vector<std::vector<apsi::sender::BinBundle>> *bin_bundles,
    const apsi::CryptoContext &crypto_context, uint32_t bundle_index,
    uint32_t bins_per_bundle, size_t label_size, size_t max_bin_size,
    size_t ps_low_degree, bool overwrite, bool compressed) {
  STOPWATCH(sender_stopwatch, "insert_or_assign_worker");
  SPDLOG_DEBUG(
      "Insert-or-Assign worker for bundle index {}; mode of operation: {}",
      bundle_index, overwrite ? "overwriting existing" : "inserting new");

  // Iteratively insert each item-label pair at the given cuckoo index
  for (auto &data_with_idx : data_with_indices) {
    const T &data = data_with_idx.first;

    // Get the bundle index
    size_t cuckoo_idx = data_with_idx.second;
    size_t bin_idx, bundle_idx;
    std::tie(bin_idx, bundle_idx) =
        UnpackCuckooIdx(cuckoo_idx, bins_per_bundle);

    // If the bundle_idx isn't in the prescribed range, don't try to insert this
    // data
    if (bundle_idx != bundle_index) {
      // Dealing with this bundle index is not our job
      continue;
    }

    // Get the bundle set at the given bundle index
    std::vector<apsi::sender::BinBundle> &bundle_set =
        (*bin_bundles)[bundle_idx];

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
      YACL_THROW("tried to overwrite non-existent item");
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
        YACL_THROW("failed to insert item into a new BinBundle");
      }

      // Push a new BinBundle to the set of BinBundles at this bundle index
      bundle_set.push_back(std::move(new_bin_bundle));
    }
  }

  SPDLOG_DEBUG("Insert-or-Assign worker: finished processing bundle index {}",
               bundle_index);
}

/**
Takes algebraized data to be inserted, splits it up, and distributes it so that
thread_count many threads can all insert in parallel. If overwrite is set, this
will overwrite the labels if it finds an AlgItemLabel that matches the input
perfectly.
*/
template <typename T>
void DispatchInsertOrAssign(
    const std::vector<std::pair<T, size_t>> &data_with_indices,
    std::vector<std::vector<apsi::sender::BinBundle>> *bin_bundles,
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
    size_t bin_idx, bundle_idx;
    std::tie(bin_idx, bundle_idx) =
        UnpackCuckooIdx(cuckoo_idx, bins_per_bundle);
    bundle_indices_set.insert(bundle_idx);
  }

  // Copy the set of indices into a vector and sort so each thread processes a
  // range of indices
  std::vector<size_t> bundle_indices;
  bundle_indices.reserve(bundle_indices_set.size());
  copy(bundle_indices_set.begin(), bundle_indices_set.end(),
       back_inserter(bundle_indices));
  std::sort(bundle_indices.begin(), bundle_indices.end());

  // Run the threads on the partitions
  std::vector<std::future<void>> futures(bundle_indices.size());
  SPDLOG_INFO("Launching {} insert-or-assign worker tasks",
              bundle_indices.size());
  size_t future_idx = 0;
  for (auto &bundle_idx : bundle_indices) {
    futures[future_idx++] = tpm.thread_pool().enqueue([&, bundle_idx]() {
      InsertOrAssignWorker(data_with_indices, bin_bundles, crypto_context,
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

/**
Removes the given items and corresponding labels from bin_bundles at their
respective cuckoo indices.
*/
void RemoveWorker(
    const std::vector<std::pair<apsi::util::AlgItem, size_t>>
        &data_with_indices,
    std::vector<std::vector<apsi::sender::BinBundle>> *bin_bundles,
    uint32_t bundle_index, uint32_t bins_per_bundle) {
  STOPWATCH(sender_stopwatch, "remove_worker");
  SPDLOG_INFO("Remove worker [{}]", bundle_index);

  // Iteratively remove each item-label pair at the given cuckoo index
  for (auto &data_with_idx : data_with_indices) {
    // Get the bundle index
    size_t cuckoo_idx = data_with_idx.second;
    size_t bin_idx, bundle_idx;
    std::tie(bin_idx, bundle_idx) =
        UnpackCuckooIdx(cuckoo_idx, bins_per_bundle);

    // If the bundle_idx isn't in the prescribed range, don't try to remove this
    // data
    if (bundle_idx != bundle_index) {
      // Dealing with this bundle index is not our job
      continue;
    }

    // Get the bundle set at the given bundle index
    std::vector<apsi::sender::BinBundle> &bundle_set =
        (*bin_bundles)[bundle_idx];

    // Try to remove these field elements from an existing BinBundle at this
    // bundle index. Keep track of whether or not we succeed.
    bool removed = false;
    for (apsi::sender::BinBundle &bundle : bundle_set) {
      // If we successfully removed, we're done with this bundle
      removed = bundle.try_multi_remove(data_with_idx.first, bin_idx);
      if (removed) {
        break;
      }
    }

    // We may have produced some empty BinBundles so just remove them all
    auto rem_it = std::remove_if(bundle_set.begin(), bundle_set.end(),
                                 [](auto &bundle) { return bundle.empty(); });
    bundle_set.erase(rem_it, bundle_set.end());

    // We tried to remove an item that doesn't exist. This should never happen
    if (!removed) {
      SPDLOG_ERROR(
          "Remove worker: "
          "failed to remove item at bundle index {} because the item was not "
          "found",
          bundle_idx);
      YACL_THROW("failed to remove item");
    }
  }

  SPDLOG_INFO("Remove worker: finished processing bundle index {}",
              bundle_index);
}

/**
Takes algebraized data to be removed, splits it up, and distributes it so that
thread_count many threads can all remove in parallel.
*/
void DispatchRemove(
    const std::vector<std::pair<apsi::util::AlgItem, size_t>>
        &data_with_indices,
    std::vector<std::vector<apsi::sender::BinBundle>> *bin_bundles,
    uint32_t bins_per_bundle) {
  apsi::ThreadPoolMgr tpm;

  // Collect the bundle indices and partition them into thread_count many
  // partitions. By some uniformity assumption, the number of things to remove
  // per partition should be roughly the same. Note that the contents of
  // bundle_indices is always sorted (increasing order).
  std::set<size_t> bundle_indices_set;
  for (auto &data_with_idx : data_with_indices) {
    size_t cuckoo_idx = data_with_idx.second;
    size_t bin_idx, bundle_idx;
    std::tie(bin_idx, bundle_idx) =
        UnpackCuckooIdx(cuckoo_idx, bins_per_bundle);
    bundle_indices_set.insert(bundle_idx);
  }

  // Copy the set of indices into a vector and sort so each thread processes a
  // range of indices
  std::vector<size_t> bundle_indices;
  bundle_indices.reserve(bundle_indices_set.size());
  copy(bundle_indices_set.begin(), bundle_indices_set.end(),
       back_inserter(bundle_indices));
  sort(bundle_indices.begin(), bundle_indices.end());

  // Run the threads on the partitions
  std::vector<std::future<void>> futures(bundle_indices.size());
  SPDLOG_INFO("Launching {} remove worker tasks", bundle_indices.size());
  size_t future_idx = 0;
  for (auto &bundle_idx : bundle_indices) {
    futures[future_idx++] = tpm.thread_pool().enqueue([&]() {
      RemoveWorker(data_with_indices, bin_bundles,
                   static_cast<uint32_t>(bundle_idx), bins_per_bundle);
    });
  }

  // Wait for the tasks to finish
  for (auto &f : futures) {
    f.get();
  }
}

/**
Returns a set of DB cache references corresponding to the bundles in the given
set
*/
std::vector<std::reference_wrapper<const apsi::sender::BinBundleCache>>
CollectCaches(std::vector<apsi::sender::BinBundle> *bin_bundles) {
  std::vector<std::reference_wrapper<const apsi::sender::BinBundleCache>>
      result;
  for (const auto &bundle : (*bin_bundles)) {
    result.emplace_back(std::cref(bundle.get_cache()));
  }

  return result;
}
}  // namespace

SenderDB::SenderDB(const apsi::PSIParams &params, size_t label_byte_count,
                   size_t nonce_byte_count, bool compressed)
    : params_(params),
      crypto_context_(params_),
      label_byte_count_(label_byte_count),
      nonce_byte_count_(label_byte_count_ ? nonce_byte_count : 0),
      item_count_(0),
      compressed_(compressed) {
  // The labels cannot be more than 1 KB.
  if (label_byte_count_ > 1024) {
    SPDLOG_ERROR("Requested label byte count {} exceeds the maximum (1024)",
                 label_byte_count_);

    YACL_THROW("label_byte_count is too large");
  }

  if (nonce_byte_count_ > apsi::max_nonce_byte_count) {
    SPDLOG_ERROR("Request nonce byte count {} exceeds the maximum ({}) ",
                 nonce_byte_count_, apsi::max_nonce_byte_count);
    YACL_THROW("nonce_byte_count is too large");
  }

  // If the nonce byte count is less than max_nonce_byte_count, print a warning;
  // this is a labeled SenderDB but may not be safe to use for arbitrary label
  // changes.
  if (label_byte_count_ && nonce_byte_count_ < apsi::max_nonce_byte_count) {
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
}

SenderDB::SenderDB(const apsi::PSIParams &params,
                   yacl::ByteContainerView oprf_key, size_t label_byte_count,
                   size_t nonce_byte_count, bool compressed)
    : SenderDB(params, label_byte_count, nonce_byte_count, compressed) {
  oprf_key_.resize(oprf_key.size());
  std::memcpy(oprf_key_.data(), oprf_key.data(), oprf_key.size());

  oprf_server_ =
      CreateEcdhOprfServer(oprf_key, OprfType::Basic, CurveType::CURVE_FOURQ);
  oprf_server_->SetCompareLength(kEccKeySize);
}

SenderDB::SenderDB(SenderDB &&source)
    : params_(source.params_),
      crypto_context_(source.crypto_context_),
      label_byte_count_(source.label_byte_count_),
      nonce_byte_count_(source.nonce_byte_count_),
      item_count_(source.item_count_),
      compressed_(source.compressed_),
      stripped_(source.stripped_) {
  // Lock the source before moving stuff over
  auto lock = source.GetWriterLock();

  hashed_items_ = move(source.hashed_items_);
  bin_bundles_ = move(source.bin_bundles_);

  std::vector<uint8_t> oprf_key = source.GetOprfKey();
  oprf_key_.resize(oprf_key.size());
  std::memcpy(oprf_key_.data(), oprf_key.data(), oprf_key.size());

  oprf_server_ =
      CreateEcdhOprfServer(oprf_key, OprfType::Basic, CurveType::CURVE_FOURQ);

  // Reset the source data structures
  source.ClearInternal();
}

SenderDB &SenderDB::operator=(SenderDB &&source) {
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

  hashed_items_ = move(source.hashed_items_);
  bin_bundles_ = move(source.bin_bundles_);

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

  return bin_bundles_.at(seal::util::safe_cast<size_t>(bundle_idx)).size();
}

size_t SenderDB::GetBinBundleCount() const {
  // Lock the database for reading
  auto lock = GetReaderLock();

  // Compute the total number of BinBundles
  return std::accumulate(bin_bundles_.cbegin(), bin_bundles_.cend(), size_t(0),
                         [&](auto &a, auto &b) { return a + b.size(); });
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

  return max_item_count ? static_cast<double>(item_count) /
                              static_cast<double>(max_item_count)
                        : 0.0;
}

void SenderDB::ClearInternal() {
  // Assume the SenderDB is already locked for writing

  // Clear the set of inserted items
  hashed_items_.clear();
  item_count_ = 0;

  // Clear the BinBundles
  bin_bundles_.clear();
  bin_bundles_.resize(params_.bundle_idx_count());

  // Reset the stripped_ flag
  stripped_ = false;
}

void SenderDB::clear() {
  if (hashed_items_.size()) {
    SPDLOG_INFO("Removing {} items pairs from SenderDB", hashed_items_.size());
  }

  // Lock the database for writing
  auto lock = GetWriterLock();

  ClearInternal();
}

void SenderDB::GenerateCaches() {
  STOPWATCH(sender_stopwatch, "SenderDB::GenerateCaches");
  SPDLOG_INFO("Start generating bin bundle caches");

  for (auto &bundle_idx : bin_bundles_) {
    for (auto &bb : bundle_idx) {
      bb.regen_cache();
    }
  }

  SPDLOG_INFO("Finished generating bin bundle caches");
}

std::vector<std::reference_wrapper<const apsi::sender::BinBundleCache>>
SenderDB::GetCacheAt(uint32_t bundle_idx) {
  return CollectCaches(
      &(bin_bundles_.at(seal::util::safe_cast<size_t>(bundle_idx))));
}

void SenderDB::strip() {
  // Lock the database for writing
  auto lock = GetWriterLock();

  stripped_ = true;

  memset(oprf_key_.data(), 0, oprf_key_.size());
  hashed_items_.clear();

  apsi::ThreadPoolMgr tpm;

  std::vector<std::future<void>> futures;
  for (auto &bundle_idx : bin_bundles_) {
    for (auto &bb : bundle_idx) {
      futures.push_back(tpm.thread_pool().enqueue([&bb]() { bb.strip(); }));
    }
  }

  // Wait for the tasks to finish
  for (auto &f : futures) {
    f.get();
  }

  SPDLOG_INFO("SenderDB has been stripped");

  return;
}

std::vector<uint8_t> SenderDB::GetOprfKey() const {
  if (stripped_) {
    SPDLOG_ERROR("Cannot return the OPRF key from a stripped SenderDB");
    YACL_THROW("failed to return OPRF key");
  }
  return oprf_key_;
}

void SenderDB::InsertOrAssign(
    const std::vector<std::pair<apsi::Item, apsi::Label>> &data) {
  if (stripped_) {
    SPDLOG_ERROR("Cannot insert data to a stripped SenderDB");
    YACL_THROW("failed to insert data");
  }
  if (!IsLabeled()) {
    SPDLOG_ERROR(
        "Attempted to insert labeled data but this is an unlabeled SenderDB");
    YACL_THROW("failed to insert data");
  }

  SPDLOG_INFO("Start inserting {} items in SenderDB", data.size());

  // First compute the hashes for the input data
  // auto hashed_data = OPRFSender::ComputeHashes(
  //    data, oprf_key_, label_byte_count_, nonce_byte_count_);
  std::vector<std::string> data_str(data.size());
  for (size_t i = 0; i < data.size(); ++i) {
    std::string temp_str(data[i].first.value().size(), '\0');

    std::memcpy(&temp_str[0], data[i].first.value().data(),
                data[i].first.value().size());
    data_str[i] = temp_str;
  }
  std::vector<std::string> oprf_out = oprf_server_->FullEvaluate(data_str);
  std::vector<std::pair<apsi::HashedItem, apsi::EncryptedLabel>> hashed_data;
  for (size_t i = 0; i < oprf_out.size(); ++i) {
    apsi::HashedItem hashed_item;
    std::memcpy(hashed_item.value().data(), &oprf_out[i][0],
                hashed_item.value().size());

    apsi::LabelKey key;
    std::memcpy(key.data(), &oprf_out[i][hashed_item.value().size()],
                key.size());

    apsi::EncryptedLabel encrypted_label = encrypt_label(
        data[i].second, key, label_byte_count_, nonce_byte_count_);

    hashed_data.push_back(std::make_pair(hashed_item, encrypted_label));
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

  auto new_item_count = distance(hashed_data.begin(), new_data_end);
  auto existing_item_count = distance(new_data_end, hashed_data.end());

  if (existing_item_count) {
    SPDLOG_INFO("Found {} existing items to replace in SenderDB",
                existing_item_count);

    // Break the data into field element representation. Also compute the items'
    // cuckoo indices.
    std::vector<std::pair<apsi::util::AlgItemLabel, size_t>> data_with_indices =
        PreprocessLabeledData(new_data_end, hashed_data.end(), params_);

    DispatchInsertOrAssign(data_with_indices, &bin_bundles_, crypto_context_,
                           bins_per_bundle, label_size, max_bin_size,
                           ps_low_degree, true, /* overwrite items */
                           compressed_);

    // Release memory that is no longer needed
    hashed_data.erase(new_data_end, hashed_data.end());
  }

  if (new_item_count) {
    SPDLOG_INFO("Found {} new items to insert in SenderDB", new_item_count);

    // Process and add the new data. Break the data into field element
    // representation. Also compute the items' cuckoo indices.
    std::vector<std::pair<apsi::util::AlgItemLabel, size_t>> data_with_indices =
        PreprocessLabeledData(hashed_data.begin(), hashed_data.end(), params_);

    DispatchInsertOrAssign(data_with_indices, &bin_bundles_, crypto_context_,
                           bins_per_bundle, label_size, max_bin_size,
                           ps_low_degree, false, /* don't overwrite items */
                           compressed_);
  }

  // Generate the BinBundle caches
  GenerateCaches();

  SPDLOG_INFO("Finished inserting {} items in SenderDB", data.size());
}

void SenderDB::InsertOrAssign(const std::vector<apsi::Item> &data) {
  if (stripped_) {
    SPDLOG_ERROR("Cannot insert data to a stripped SenderDB");
    YACL_THROW("failed to insert data");
  }
  if (IsLabeled()) {
    SPDLOG_ERROR(
        "Attempted to insert unlabeled data but this is a labeled SenderDB");
    YACL_THROW("failed to insert data");
  }

  STOPWATCH(sender_stopwatch, "SenderDB::insert_or_assign (unlabeled)");
  SPDLOG_INFO("Start inserting {} items in SenderDB", data.size());

  // First compute the hashes for the input data
  // auto hashed_data = OPRFSender::ComputeHashes(data, oprf_key_);
  std::vector<std::string> data_str(data.size());
  for (size_t i = 0; i < data.size(); ++i) {
    std::string item_str(data[i].value().size(), '\0');
    std::memcpy(&item_str[0], data[i].value().data(), data[i].value().size());
    data_str[i] = item_str;
  }
  std::vector<std::string> oprf_out = oprf_server_->FullEvaluate(data_str);
  std::vector<apsi::HashedItem> hashed_data;
  for (size_t i = 0; i < oprf_out.size(); ++i) {
    apsi::Item::value_type value{};
    std::memcpy(value.data(), &oprf_out[i][0], value.size());

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

  // Break the new data down into its field element representation. Also compute
  // the items' cuckoo indices.
  std::vector<std::pair<apsi::util::AlgItem, size_t>> data_with_indices =
      PreprocessUnlabeledData(hashed_data.begin(), hashed_data.end(), params_);

  // Dispatch the insertion
  uint32_t bins_per_bundle = params_.bins_per_bundle();
  uint32_t max_bin_size = params_.table_params().max_items_per_bin;
  uint32_t ps_low_degree = params_.query_params().ps_low_degree;

  DispatchInsertOrAssign(data_with_indices, &bin_bundles_, crypto_context_,
                         bins_per_bundle, 0, /* label size */
                         max_bin_size, ps_low_degree,
                         false, /* don't overwrite items */
                         compressed_);

  // Generate the BinBundle caches
  GenerateCaches();

  SPDLOG_INFO("Finished inserting {} items in SenderDB", data.size());
}

void SenderDB::remove(const std::vector<apsi::Item> &data) {
  if (stripped_) {
    SPDLOG_ERROR("Cannot remove data from a stripped SenderDB");
    YACL_THROW("failed to remove data");
  }

  STOPWATCH(sender_stopwatch, "SenderDB::remove");
  SPDLOG_INFO("Start removing {} items from SenderDB", data.size());

  // First compute the hashes for the input data
  // auto hashed_data = OPRFSender::ComputeHashes(data, oprf_key_);
  std::vector<std::string> data_str(data.size());
  for (size_t i = 0; i < data.size(); ++i) {
    data_str[i].reserve(data[i].value().size());
    std::memcpy(&data_str[i][0], data[i].value().data(),
                data[i].value().size());
  }
  std::vector<std::string> oprf_out = oprf_server_->FullEvaluate(data_str);
  std::vector<apsi::HashedItem> hashed_data;
  for (size_t i = 0; i < oprf_out.size(); ++i) {
    apsi::Item::value_type value{};
    std::memcpy(value.data(), &oprf_out[i][0], value.size());

    hashed_data.emplace_back(value);
  }

  // Lock the database for writing
  auto lock = GetWriterLock();

  // Remove items that do not exist in the database.
  auto existing_data_end = std::remove_if(
      hashed_data.begin(), hashed_data.end(), [&](const auto &item) {
        bool found = hashed_items_.find(item) != hashed_items_.end();
        if (found) {
          // Remove from hashed_items_ already at this point!
          hashed_items_.erase(item);
          item_count_--;
        }

        // Remove those that were not found
        return !found;
      });

  // This distance is always non-negative
  size_t existing_item_count =
      static_cast<size_t>(distance(existing_data_end, hashed_data.end()));
  if (existing_item_count) {
    SPDLOG_WARN("Ignoring {} items that are not present in the SenderDB",
                existing_item_count);
  }

  // Break the data down into its field element representation. Also compute the
  // items' cuckoo indices.
  std::vector<std::pair<apsi::util::AlgItem, size_t>> data_with_indices =
      PreprocessUnlabeledData(hashed_data.begin(), hashed_data.end(), params_);

  // Dispatch the removal
  uint32_t bins_per_bundle = params_.bins_per_bundle();
  DispatchRemove(data_with_indices, &bin_bundles_, bins_per_bundle);

  // Generate the BinBundle caches
  GenerateCaches();

  SPDLOG_INFO("Finished removing {} items from SenderDB", data.size());
}

bool SenderDB::HasItem(const apsi::Item &item) const {
  if (stripped_) {
    SPDLOG_ERROR(
        "Cannot retrieve the presence of an item from a stripped SenderDB");
    YACL_THROW("failed to retrieve the presence of item");
  }

  // First compute the hash for the input item
  // auto hashed_item = OPRFSender::ComputeHashes({&item, 1}, oprf_key_)[0];
  std::string item_str;
  item_str.reserve(item.value().size());
  std::memcpy(&item_str[0], item.value().data(), item.value().size());
  std::string oprf_out = oprf_server_->FullEvaluate(item_str);
  apsi::HashedItem hashed_item;
  std::memcpy(hashed_item.value().data(), &oprf_out[0],
              hashed_item.value().size());

  // Lock the database for reading
  auto lock = GetReaderLock();

  return hashed_items_.find(hashed_item) != hashed_items_.end();
}

apsi::Label SenderDB::GetLabel(const apsi::Item &item) const {
  if (stripped_) {
    SPDLOG_ERROR("Cannot retrieve a label from a stripped SenderDB");
    YACL_THROW("failed to retrieve label");
  }
  if (!IsLabeled()) {
    SPDLOG_ERROR(
        "Attempted to retrieve a label but this is an unlabeled SenderDB");
    YACL_THROW("failed to retrieve label");
  }

  // First compute the hash for the input item
  apsi::HashedItem hashed_item;
  apsi::LabelKey key;
  // tie(hashed_item, key) = OPRFSender::GetItemHash(item, oprf_key_);

  std::string item_str;
  item_str.reserve(item.value().size());
  std::memcpy(&item_str[0], item.value().data(), item.value().size());
  std::string oprf_out = oprf_server_->FullEvaluate(item_str);
  std::memcpy(hashed_item.value().data(), &oprf_out[0],
              hashed_item.value().size());

  // Lock the database for reading
  auto lock = GetReaderLock();

  // Check if this item is in the DB. If not, throw an exception
  if (hashed_items_.find(hashed_item) == hashed_items_.end()) {
    SPDLOG_ERROR(
        "Cannot retrieve label for an item that is not in the SenderDB");
    YACL_THROW("failed to retrieve label");
  }

  uint32_t bins_per_bundle = params_.bins_per_bundle();

  // Preprocess a single element. This algebraizes the item and gives back its
  // field element representation as well as its cuckoo hash. We only read one
  // of the locations because the labels are the same in each location.
  apsi::util::AlgItem alg_item;
  size_t cuckoo_idx;
  std::tie(alg_item, cuckoo_idx) =
      PreprocessUnlabeledData(hashed_item, params_)[0];

  // Now figure out where to look to get the label
  size_t bin_idx, bundle_idx;
  std::tie(bin_idx, bundle_idx) = UnpackCuckooIdx(cuckoo_idx, bins_per_bundle);

  // Retrieve the algebraic labels from one of the BinBundles at this index
  const std::vector<apsi::sender::BinBundle> &bundle_set =
      bin_bundles_[bundle_idx];
  std::vector<felt_t> alg_label;
  bool got_labels = false;
  for (const apsi::sender::BinBundle &bundle : bundle_set) {
    // Try to retrieve the contiguous labels from this BinBundle
    if (bundle.try_get_multi_label(alg_item, bin_idx, alg_label)) {
      got_labels = true;
      break;
    }
  }

  // It shouldn't be possible to have items in your set but be unable to
  // retrieve the associated label. Throw an exception because something is
  // terribly wrong.
  if (!got_labels) {
    SPDLOG_ERROR(
        "Failed to retrieve label for an item that was supposed to be in the "
        "SenderDB");
    YACL_THROW("failed to retrieve label");
  }

  // All good. Now just reconstruct the big label from its split-up parts
  apsi::EncryptedLabel encrypted_label = dealgebraize_label(
      alg_label,
      alg_label.size() * static_cast<size_t>(params_.item_bit_count_per_felt()),
      params_.seal_params().plain_modulus());

  // Resize down to the effective byte count
  encrypted_label.resize(nonce_byte_count_ + label_byte_count_);

  // Decrypt the label
  return decrypt_label(encrypted_label, key, nonce_byte_count_);
}

}  // namespace spu::psi
