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

}

namespace labeled_psi {

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

}  // namespace labeled_psi

ISenderDB::ISenderDB(const apsi::PSIParams &params,
                     yacl::ByteContainerView oprf_key,
                     std::size_t label_byte_count, std::size_t nonce_byte_count,
                     bool compressed)
    : params_(params),
      crypto_context_(params_),
      label_byte_count_(label_byte_count),
      nonce_byte_count_(label_byte_count_ != 0 ? nonce_byte_count : 0),
      item_count_(0),
      compressed_(compressed) {
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

  oprf_key_.resize(oprf_key.size());
  std::memcpy(oprf_key_.data(), oprf_key.data(), oprf_key.size());

  oprf_server_ =
      CreateEcdhOprfServer(oprf_key, OprfType::Basic, CurveType::CURVE_FOURQ);
  oprf_server_->SetCompareLength(kEccKeySize);
}

double ISenderDB::GetPackingRate() const {
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

std::vector<uint8_t> ISenderDB::GetOprfKey() const {
  if (stripped_) {
    SPDLOG_ERROR("Cannot return the OPRF key from a stripped SenderDB");
    SPU_THROW("failed to return OPRF key");
  }
  return oprf_key_;
}

}  // namespace spu::psi
