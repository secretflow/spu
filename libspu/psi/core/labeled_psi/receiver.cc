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

#include "libspu/psi/core/labeled_psi/receiver.h"

#include <algorithm>
#include <cstddef>
#include <future>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "apsi/plaintext_powers.h"
#include "apsi/util/db_encoding.h"
#include "apsi/util/label_encryptor.h"
#include "spdlog/spdlog.h"
#include "yacl/utils/parallel.h"

#include "libspu/psi/core/ecdh_oprf/ecdh_oprf_selector.h"
#include "libspu/psi/core/labeled_psi/package.h"
#include "libspu/psi/core/labeled_psi/psi_params.h"
#include "libspu/psi/utils/utils.h"

namespace spu::psi {

namespace {

constexpr std::uint64_t kCuckooTableInsertAttempts = 500;

template <typename T>
bool HasNZeros(T *ptr, size_t count) {
  return std::all_of(ptr, ptr + count,
                     [](auto a) { return a == static_cast<T>(0); });
}
}  // namespace

LabelPsiReceiver::LabelPsiReceiver(const apsi::PSIParams &params,
                                   bool has_label)
    : psi_params_(params), has_label_(has_label) {
  Initialize();
}

void LabelPsiReceiver::Initialize() {
  SPDLOG_DEBUG("PSI parameters set to: {}", psi_params_.to_string());
  SPDLOG_DEBUG(
      "item_bit_count_per_felt: {} ; item_bit_count: {};"
      " bins_per_bundle: {}; bundle_idx_count: {}",
      psi_params_.item_bit_count_per_felt(), psi_params_.item_bit_count(),
      psi_params_.bins_per_bundle(), psi_params_.bundle_idx_count());

  // Initialize the CryptoContext with a new SEALContext
  crypto_context_ = apsi::CryptoContext(psi_params_);

  // Set up the PowersDag
  ResetPowersDag(psi_params_.query_params().query_powers);

  // Create new keys
  ResetKeys();
}

void LabelPsiReceiver::ResetKeys() {
  // Generate new keys
  seal::KeyGenerator generator(*GetSealContext());

  // Set the symmetric key, encryptor, and decryptor
  crypto_context_.set_secret(generator.secret_key());

  // Create Serializable<RelinKeys> and move to relin_keys_ for storage
  relin_keys_.clear();
  if (GetSealContext()->using_keyswitching()) {
    seal::Serializable<seal::RelinKeys> relin_keys(
        generator.create_relin_keys());
    relin_keys_.set(std::move(relin_keys));
  }
}

std::uint32_t LabelPsiReceiver::ResetPowersDag(
    const std::set<std::uint32_t> &source_powers) {
  // First compute the target powers
  std::set<uint32_t> target_powers = apsi::util::create_powers_set(
      psi_params_.query_params().ps_low_degree,
      psi_params_.table_params().max_items_per_bin);

  // Configure the PowersDag
  pd_.configure(source_powers, target_powers);

  // Check that the PowersDag is valid
  if (!pd_.is_configured()) {
    SPDLOG_ERROR(
        "Failed to configure PowersDag (source_powers:{} target_powers:{})",
        apsi::util::to_string(source_powers),
        apsi::util::to_string(target_powers));

    SPU_THROW("failed to configure PowersDag");
  }
  SPDLOG_DEBUG("Configured PowersDag with depth {}", pd_.depth());

  return pd_.depth();
}

apsi::PSIParams LabelPsiReceiver::RequestPsiParams(
    size_t items_size, const std::shared_ptr<yacl::link::Context> &link_ctx) {
  yacl::Buffer buffer(&items_size, sizeof(items_size));

  link_ctx->SendAsyncThrottled(
      link_ctx->NextRank(), buffer,
      fmt::format("send client items size:{}", items_size));

  yacl::Buffer psi_params_buffer = link_ctx->Recv(
      link_ctx->NextRank(), fmt::format("recv psi params message"));

  return ParsePsiParamsProto(psi_params_buffer);
}

std::pair<std::vector<apsi::HashedItem>, std::vector<apsi::LabelKey>>
LabelPsiReceiver::RequestOPRF(
    const std::vector<std::string> &items,
    const std::shared_ptr<yacl::link::Context> &link_ctx) {
  std::vector<std::string> blind_items(items.size());
  std::vector<std::shared_ptr<IEcdhOprfClient>> oprf_clients(items.size());

  yacl::parallel_for(0, items.size(), 1, [&](int64_t begin, int64_t end) {
    for (int idx = begin; idx < end; ++idx) {
      oprf_clients[idx] =
          CreateEcdhOprfClient(OprfType::Basic, CurveType::CURVE_FOURQ);
      oprf_clients[idx]->SetCompareLength(kEccKeySize);

      blind_items[idx] = oprf_clients[idx]->Blind(items[idx]);
    }
  });

  proto::OprfProto oprf_proto;
  for (auto &blind_item : blind_items) {
    oprf_proto.add_data(blind_item.data(), blind_item.length());
  }

  yacl::Buffer blind_buffer(oprf_proto.ByteSizeLong());
  oprf_proto.SerializePartialToArray(blind_buffer.data(), blind_buffer.size());

  link_ctx->SendAsyncThrottled(
      link_ctx->NextRank(), blind_buffer,
      fmt::format("send oprf blind items buffer size:{}", blind_buffer.size()));

  yacl::Buffer evaluated_buffer = link_ctx->Recv(
      link_ctx->NextRank(), fmt::format("recv oprf evaluated message"));

  proto::OprfProto evaluated_proto;
  SPU_ENFORCE(evaluated_proto.ParseFromArray(evaluated_buffer.data(),
                                             evaluated_buffer.size()));

  std::vector<std::string> items_oprf(evaluated_proto.data_size());
  yacl::parallel_for(0, evaluated_proto.data_size(), 1,
                     [&](int64_t begin, int64_t end) {
                       for (int idx = begin; idx < end; ++idx) {
                         items_oprf[idx] = oprf_clients[idx]->Finalize(
                             items[idx], evaluated_proto.data(idx));
                       }
                     });

  std::vector<apsi::HashedItem> hashed_items(items_oprf.size());
  std::vector<apsi::LabelKey> label_keys(items_oprf.size());

  for (size_t idx = 0; idx < items_oprf.size(); ++idx) {
    std::memcpy(hashed_items[idx].value().data(), items_oprf[idx].data(),
                hashed_items[idx].value().size());

    std::memcpy(label_keys[idx].data(),
                &items_oprf[idx][hashed_items[idx].value().size()],
                label_keys[idx].size());
  }

  return std::make_pair(std::move(hashed_items), std::move(label_keys));
}

std::pair<std::vector<size_t>, std::vector<std::string>>
LabelPsiReceiver::RequestQuery(
    const std::vector<apsi::HashedItem> &hashed_items,
    const std::vector<apsi::LabelKey> &label_keys,
    const std::shared_ptr<yacl::link::Context> &link_ctx) {
  kuku::KukuTable cuckoo(
      psi_params_.table_params().table_size,       // Size of the hash table
      0,                                           // Not using a stash
      psi_params_.table_params().hash_func_count,  // Number of hash functions
      {0, 0},                      // Hardcoded { 0, 0 } as the seed
      kCuckooTableInsertAttempts,  // The number of insertion attempts
      {0, 0});

  SPDLOG_INFO("cuckoo table_size:{}", cuckoo.table_size());

  // Hash the data into a cuckoo hash table
  // cuckoo_hashing
  for (size_t item_idx = 0; item_idx < hashed_items.size(); item_idx++) {
    const auto &item = hashed_items[item_idx];
    if (!cuckoo.insert(item.get_as<kuku::item_type>().front())) {
      // Insertion can fail for two reasons:
      //
      //     (1) The item was already in the table, in which case the
      //     "leftover item" is empty; (2) Cuckoo hashing failed due to too
      //     small table or too few hash functions.
      //
      // In case (1) simply move on to the next item and log this issue. Case
      // (2) is a critical issue so we throw and exception.
      if (cuckoo.is_empty_item(cuckoo.leftover_item())) {
        SPDLOG_INFO("Skipping repeated insertion of items{}:{}", item_idx,
                    item.to_string());
      } else {
        SPDLOG_INFO("Failed to insert items[{}:{}; cuckoo table fill-rate: {}",
                    item_idx, item.to_string(), cuckoo.fill_rate());
        SPU_THROW("failed to insert item into cuckoo table");
      }
    }
  }
  SPDLOG_INFO(
      "Finished inserting items with {} hash functions; cuckoo table "
      "fill-rate: {}",
      cuckoo.loc_func_count(), cuckoo.fill_rate());

  apsi::receiver::IndexTranslationTable itt;
  itt.item_count_ = hashed_items.size();

  for (size_t item_idx = 0; item_idx < hashed_items.size(); item_idx++) {
    auto item_loc =
        cuckoo.query(hashed_items[item_idx].get_as<kuku::item_type>().front());
    itt.table_idx_to_item_idx_[item_loc.location()] = item_idx;
  }

  // Set up unencrypted query data
  std::vector<apsi::receiver::PlaintextPowers> plain_powers;

  // prepare_data
  {
    STOPWATCH(apsi::util::recv_stopwatch,
              "Receiver::create_query::prepare_data");
    for (uint32_t bundle_idx = 0; bundle_idx < psi_params_.bundle_idx_count();
         bundle_idx++) {
      SPDLOG_DEBUG("Preparing data for bundle index {}", bundle_idx);

      // First, find the items for this bundle index
      absl::Span<const kuku::item_type> bundle_items = absl::MakeSpan(
          cuckoo.table().data() +
              static_cast<size_t>(bundle_idx * psi_params_.items_per_bundle()),
          psi_params_.items_per_bundle());

      std::vector<uint64_t> alg_items;
      for (const auto &item : bundle_items) {
        // Now set up a BitstringView to this item
        gsl::span<const unsigned char> item_bytes(
            reinterpret_cast<const unsigned char *>(item.data()), sizeof(item));
        apsi::BitstringView<const unsigned char> item_bits(
            item_bytes, psi_params_.item_bit_count());

        // Create an algebraic item by breaking up the item into modulo
        // plain_modulus parts
        std::vector<uint64_t> alg_item = apsi::util::bits_to_field_elts(
            item_bits, psi_params_.seal_params().plain_modulus());
        copy(alg_item.cbegin(), alg_item.cend(), back_inserter(alg_items));
      }

      // Now that we have the algebraized items for this bundle index, we
      // create a PlaintextPowers object that computes all necessary powers of
      // the algebraized items.
      plain_powers.emplace_back(std::move(alg_items), psi_params_, pd_);
    }
  }
  SPDLOG_INFO("plain_powers size:{}, using_keyswitching:{}",
              plain_powers.size(), GetSealContext()->using_keyswitching());

  // The very last thing to do is encrypt the plain_powers and consolidate the
  // matching powers for different bundle indices
  std::unordered_map<uint32_t, std::vector<apsi::SEALObject<seal::Ciphertext>>>
      encrypted_powers;

  // encrypt_data
  {
    SPDLOG_DEBUG("Receiver::create_query::encrypt_data");
    for (uint32_t bundle_idx = 0; bundle_idx < psi_params_.bundle_idx_count();
         bundle_idx++) {
      SPDLOG_DEBUG("Encoding and encrypting data for bundle index {}",
                   bundle_idx);

      // Encrypt the data for this power
      auto encrypted_power(plain_powers[bundle_idx].encrypt(crypto_context_));

      // Move the encrypted data to encrypted_powers
      for (auto &e : encrypted_power) {
        encrypted_powers[e.first].emplace_back(std::move(e.second));
      }
    }
  }
  SPDLOG_INFO("encrypted_powers size: {}", encrypted_powers.size());

  std::vector<uint8_t> temp;
  if (GetSealContext()->using_keyswitching()) {
    temp.resize(relin_keys_.save_size(compr_mode_));
    relin_keys_.save(temp, compr_mode_);
  }

  proto::QueryRequestProto query_proto;
  query_proto.set_relin_keys(temp.data(), temp.size());

  for (const auto &q : encrypted_powers) {
    proto::EncryptedPowersProto *powers_proto =
        query_proto.add_encrypted_powers();
    powers_proto->set_power(q.first);
    for (const auto &ct : q.second) {
      // Save each SEALObject<seal::Ciphertext>
      temp.resize(ct.save_size(compr_mode_));
      auto size = ct.save(temp, compr_mode_);
      powers_proto->add_ciphertexts(temp.data(), size);
    }
    SPDLOG_DEBUG("ciphertexts_size:{}", powers_proto->ciphertexts_size());
  }

  yacl::Buffer query_buffer(query_proto.ByteSizeLong());
  query_proto.SerializePartialToArray(query_buffer.data(), query_buffer.size());

  link_ctx->SendAsyncThrottled(
      link_ctx->NextRank(), query_buffer,
      fmt::format("send query buffer size:{}", query_buffer.size()));

  yacl::Buffer response_buffer = link_ctx->Recv(
      link_ctx->NextRank(), fmt::format("recv server query response message"));

  proto::QueryResponseProto response_proto;
  SPU_ENFORCE(response_proto.ParseFromArray(response_buffer.data(),
                                            response_buffer.size()));

  std::vector<std::pair<size_t, std::string>> query_result_vec;

  std::vector<std::vector<std::pair<size_t, std::string>>> results(
      response_proto.results_size());

  yacl::parallel_for(0, response_proto.results_size(), 1,
                     [&](int64_t begin, int64_t end) {
                       for (int idx = begin; idx < end; ++idx) {
                         const proto::QueryResultProto &query_result_proto =
                             response_proto.results(idx);
                         results[idx] = LabelPsiReceiver::ProcessQueryResult(
                             query_result_proto, itt, label_keys);
                       }
                     });

  for (int idx = 0; idx < response_proto.results_size(); ++idx) {
    query_result_vec.insert(query_result_vec.end(), results[idx].begin(),
                            results[idx].end());
  }

  std::sort(query_result_vec.begin(), query_result_vec.end(),
            [](const std::pair<size_t, std::string> &a,
               const std::pair<size_t, std::string> &b) {
              return a.first < b.first;
            });

  std::vector<size_t> query_result;
  std::vector<std::string> query_labels;

  for (const auto &[index, value] : query_result_vec) {
    query_result.emplace_back(index);
    if (has_label_) {
      query_labels.emplace_back(value);
    }
  }

  return std::make_pair(std::move(query_result), std::move(query_labels));
}

std::vector<std::pair<size_t, std::string>>
LabelPsiReceiver::ProcessQueryResult(
    const proto::QueryResultProto &query_result_proto,
    const apsi::receiver::IndexTranslationTable &itt,
    const std::vector<apsi::LabelKey> &label_keys) {
  auto seal_context = GetSealContext();
  ResultPackage result_package;

  result_package.bundle_idx = query_result_proto.bundle_idx();

  auto query_cipher_data = query_result_proto.ciphertext();
  gsl::span<const unsigned char> query_cipher_data_span(
      reinterpret_cast<const unsigned char *>(query_cipher_data.data()),
      query_cipher_data.length());
  result_package.psi_result.load(seal_context, query_cipher_data_span);

  result_package.label_byte_count = query_result_proto.label_byte_count();
  result_package.nonce_byte_count = query_result_proto.nonce_byte_count();

  for (int idx = 0; idx < query_result_proto.label_results_size(); ++idx) {
    auto label_data = query_result_proto.label_results(idx);
    gsl::span<const unsigned char> label_data_span(
        reinterpret_cast<const unsigned char *>(label_data.data()),
        label_data.length());
    apsi::SEALObject<seal::Ciphertext> temp;
    temp.load(seal_context, label_data_span);
    result_package.label_result.emplace_back(std::move(temp));
  }

  PlainResultPackage plain_rp = result_package.extract(crypto_context_);

  size_t item_count = itt.item_count();

  auto felts_per_item =
      seal::util::safe_cast<size_t>(psi_params_.item_params().felts_per_item);
  auto items_per_bundle =
      seal::util::safe_cast<size_t>(psi_params_.items_per_bundle());
  size_t bundle_start = seal::util::mul_safe(
      seal::util::safe_cast<size_t>(plain_rp.bundle_idx), items_per_bundle);

  SPDLOG_DEBUG("bundle_start:{},felts_per_item:{}", bundle_start,
               felts_per_item);

  // Check if we are supposed to have label data present but don't have for
  // some reason
  auto label_byte_count =
      seal::util::safe_cast<size_t>(plain_rp.label_byte_count);
  if ((label_byte_count != 0) && plain_rp.label_result.empty()) {
    SPDLOG_WARN(
        "Expected {}-byte labels in this result part, but label data is "
        "missing entirely",
        label_byte_count);

    // Just ignore the label data
    label_byte_count = 0;
  }

  // Read the nonce byte count and compute the effective label byte count; set
  // the nonce byte count to zero if no label is expected anyway.
  size_t nonce_byte_count =
      label_byte_count != 0
          ? seal::util::safe_cast<size_t>(plain_rp.nonce_byte_count)
          : 0;
  size_t effective_label_byte_count =
      seal::util::add_safe(nonce_byte_count, label_byte_count);

  // How much label data did we actually receive?
  size_t received_label_bit_count = seal::util::mul_safe(
      seal::util::safe_cast<size_t>(psi_params_.item_bit_count()),
      plain_rp.label_result.size());

  // Compute the received label byte count and check that it is not less than
  // what was expected
  size_t received_label_byte_count = received_label_bit_count / 8;
  if (received_label_byte_count < nonce_byte_count) {
    SPDLOG_WARN(
        "Expected {} bytes of nonce data in this result part but only {} bytes "
        "were received; ignoring the label data",
        nonce_byte_count, received_label_byte_count);

    // Just ignore the label data
    label_byte_count = 0;
    effective_label_byte_count = 0;
  } else if (received_label_byte_count < effective_label_byte_count) {
    SPDLOG_WARN(
        "Expected {} bytes of label data in this result part but only {} bytes "
        "were received",
        label_byte_count, (received_label_byte_count - nonce_byte_count));

    // Reset our expectations to what was actually received
    label_byte_count = received_label_byte_count - nonce_byte_count;
    effective_label_byte_count = received_label_byte_count;
  }

  // If there is a label, then we better have the appropriate label encryption
  // keys available
  if ((label_byte_count != 0) && label_keys.size() != item_count) {
    SPDLOG_WARN(
        "Expected {} label encryption keys but only {} were given; ignoring "
        "the label data",
        item_count, label_keys.size());

    SPDLOG_INFO(
        "Expected {} label encryption keys but only {} were given; ignoring "
        "the label data",
        item_count, label_keys.size());

    // Just ignore the label data
    label_byte_count = 0;
    effective_label_byte_count = 0;
  }

  seal::util::StrideIter<const uint64_t *> plain_rp_iter(
      plain_rp.psi_result.data(), felts_per_item);

  std::vector<std::pair<size_t, std::string>> match_ids;
  seal_for_each_n(
      seal::util::iter(plain_rp_iter, static_cast<size_t>(0)), items_per_bundle,
      [&](auto &&i) {
        // Find felts_per_item consecutive zeros
        bool match = HasNZeros(std::get<0>(i).ptr(), felts_per_item);

        if (!match) {
          return;
        }

        // Compute the cuckoo table index for this item. Then find
        // the corresponding index
        // in the input items vector so we know where to place the
        // result.
        size_t table_idx = seal::util::add_safe(std::get<1>(i), bundle_start);
        auto item_idx = itt.find_item_idx(table_idx);

        // If this table_idx doesn't match any item_idx, ignore the
        // result no matter what it is
        if (item_idx == itt.item_count()) {
          return;
        }

        SPDLOG_DEBUG("Match found for items[{}] at cuckoo table index {}",
                     item_idx, table_idx);
        apsi::Label label;
        if (label_byte_count) {
          SPDLOG_DEBUG(
              "Found {} label parts for items[{}]; expecting {}-byte label ",
              plain_rp.label_result.size(), item_idx, label_byte_count);

          // Collect the entire label into this vector
          apsi::util::AlgLabel alg_label;

          size_t label_offset =
              seal::util::mul_safe(std::get<1>(i), felts_per_item);
          for (auto &label_parts : plain_rp.label_result) {
            gsl::span<apsi::util::felt_t> label_part(
                label_parts.data() + label_offset, felts_per_item);
            std::copy(label_part.begin(), label_part.end(),
                      back_inserter(alg_label));
          }

          // Create the label
          apsi::EncryptedLabel encrypted_label = apsi::util::dealgebraize_label(
              alg_label, received_label_bit_count,
              psi_params_.seal_params().plain_modulus());

          // Resize down to the effective byte count
          encrypted_label.resize(effective_label_byte_count);

          // Decrypt the label
          label = apsi::util::decrypt_label(
              encrypted_label, label_keys[item_idx], nonce_byte_count);
        }

        std::string label_str;
        if (label.size() > 0) {
          label_str = UnPaddingData(label);
        }

        match_ids.push_back(std::make_pair(item_idx, label_str));
      });

  std::sort(match_ids.begin(), match_ids.end(),
            [](const std::pair<size_t, std::string> &a,
               const std::pair<size_t, std::string> &b) {
              return a.first < b.first;
            });

  return match_ids;
}

}  // namespace spu::psi
