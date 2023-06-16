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

#include "libspu/psi/core/labeled_psi/sender.h"

#include <algorithm>
#include <cmath>
#include <future>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "apsi/powers.h"
#include "apsi/util/label_encryptor.h"
#include "gsl/span"
#include "yacl/utils/parallel.h"

#include "libspu/psi/core/labeled_psi/package.h"
#include "libspu/psi/core/labeled_psi/sender_db.h"

#include "libspu/psi/core/labeled_psi/serializable.pb.h"

namespace spu::psi {

namespace {

class QueryRequest {
 public:
  QueryRequest(apsi::SEALObject<seal::RelinKeys> *relin_keys,
               std::unordered_map<
                   uint32_t, std::vector<apsi::SEALObject<seal::Ciphertext>>>
                   &encrypted_powers,
               const std::shared_ptr<spu::psi::SenderDB> &sender_db) {
    auto seal_context = sender_db->GetSealContext();

    for (auto &q : encrypted_powers) {
      SPDLOG_DEBUG("Extracting {} ciphertexts for exponent {}", q.second.size(),
                   q.first);
      std::vector<seal::Ciphertext> cts;
      for (auto &ct : q.second) {
        cts.push_back(ct.extract(seal_context));
        if (!is_valid_for(cts.back(), *seal_context)) {
          SPDLOG_ERROR("Extracted ciphertext is invalid for SEALContext");
          SPU_THROW("Extracted ciphertext is invalid for SEALContext");
          return;
        }
      }
      data_[q.first] = std::move(cts);
    }

    if (seal_context->using_keyswitching()) {
      relin_keys_ = relin_keys->extract(seal_context);
      if (!is_valid_for(relin_keys_, *seal_context)) {
        SPU_THROW("Extracted relinearization keys are invalid for SEALContext");
      }
    }
  }

  const seal::RelinKeys &relin_keys() const noexcept { return relin_keys_; }

  auto &data() const noexcept { return data_; }

  seal::compr_mode_type compr_mode() const noexcept { return compr_mode_; }

 private:
  seal::RelinKeys relin_keys_;

  /**
  Holds the encrypted query data. In the map the key labels the exponent of the
  query ciphertext and the vector holds the ciphertext data for different bundle
  indices.
  */
  std::unordered_map<std::uint32_t, std::vector<seal::Ciphertext>> data_;

  seal::compr_mode_type compr_mode_ = seal::Serialization::compr_mode_default;
};

using CiphertextPowers = std::vector<seal::Ciphertext>;

uint32_t reset_powers_dag(apsi::PowersDag *pd, const apsi::PSIParams &params,
                          const std::set<uint32_t> &source_powers) {
  // First compute the target powers
  std::set<uint32_t> target_powers =
      apsi::util::create_powers_set(params.query_params().ps_low_degree,
                                    params.table_params().max_items_per_bin);
  SPDLOG_DEBUG("target_powers size:{}", target_powers.size());

  // Configure the PowersDag
  pd->configure(source_powers, target_powers);

  // Check that the PowersDag is valid
  if (!pd->is_configured()) {
    SPDLOG_INFO(
        "Failed to configure PowersDag ("
        "source_powers: {}, target_powers: {}",
        apsi::util::to_string(source_powers),
        apsi::util::to_string(target_powers));
    SPU_THROW("failed to configure PowersDag");
  }
  SPDLOG_INFO("Configured PowersDag with depth {}", pd->depth());

  return pd->depth();
}

}  // namespace

LabelPsiSender::LabelPsiSender(std::shared_ptr<spu::psi::SenderDB> sender_db)
    : sender_db_(std::move(sender_db)) {
  apsi::PSIParams params(sender_db_->GetParams());

  crypto_context_ = apsi::CryptoContext(sender_db_->GetParams());

  SPDLOG_INFO("begin set PowersDag");
  reset_powers_dag(&pd_, params, params.query_params().query_powers);

  SPDLOG_INFO("pd_ is_configured:{}", pd_.is_configured());
}

void LabelPsiSender::RunPsiParams(
    size_t items_size, const std::shared_ptr<yacl::link::Context> &link_ctx) {
  yacl::Buffer nr_buffer =
      link_ctx->Recv(link_ctx->NextRank(), fmt::format("recv psi item size"));

  size_t nr;
  SPU_ENFORCE(sizeof(nr) == nr_buffer.size());
  std::memcpy(&nr, nr_buffer.data(), nr_buffer.size());

  apsi::PSIParams psi_params = spu::psi::GetPsiParams(nr, items_size);

  yacl::Buffer params_buffer = PsiParamsToBuffer(psi_params);

  link_ctx->SendAsyncThrottled(
      link_ctx->NextRank(), params_buffer,
      fmt::format("send psi params buffer size:{}", params_buffer.size()));
}

void LabelPsiSender::RunOPRF(
    const std::shared_ptr<IEcdhOprfServer> &oprf_server,
    const std::shared_ptr<yacl::link::Context> &link_ctx) {
  oprf_server->SetCompareLength(kEccKeySize);

  yacl::Buffer blind_buffer = link_ctx->Recv(
      link_ctx->NextRank(), fmt::format("recv oprf blind message"));

  proto::OprfProto blind_proto;
  SPU_ENFORCE(
      blind_proto.ParseFromArray(blind_buffer.data(), blind_buffer.size()));

  proto::OprfProto evaluated_proto;
  std::vector<std::string> evaluated_vec(blind_proto.data_size());
  yacl::parallel_for(
      0, blind_proto.data_size(), 1, [&](int64_t begin, int64_t end) {
        for (int idx = begin; idx < end; ++idx) {
          evaluated_vec[idx] = oprf_server->Evaluate(blind_proto.data(idx));
        }
      });

  for (int idx = 0; idx < blind_proto.data_size(); ++idx) {
    evaluated_proto.add_data(evaluated_vec[idx].data(),
                             evaluated_vec[idx].length());
  }

  yacl::Buffer evaluated_buffer(evaluated_proto.ByteSizeLong());
  evaluated_proto.SerializePartialToArray(evaluated_buffer.data(),
                                          evaluated_buffer.size());

  link_ctx->SendAsyncThrottled(
      link_ctx->NextRank(), evaluated_buffer,
      fmt::format("send evaluated items buffer size:{}",
                  evaluated_buffer.size()));
}

std::vector<std::shared_ptr<ResultPackage>> SenderRunQuery(
    const QueryRequest &query,
    const std::shared_ptr<spu::psi::SenderDB> &sender_db,
    const apsi::PowersDag &pd);

void LabelPsiSender::RunQuery(
    const std::shared_ptr<yacl::link::Context> &link_ctx) {
  yacl::Buffer query_buffer = link_ctx->Recv(
      link_ctx->NextRank(), fmt::format("recv client query message"));

  proto::QueryRequestProto query_proto;
  SPU_ENFORCE(
      query_proto.ParseFromArray(query_buffer.data(), query_buffer.size()));

  auto seal_context = sender_db_->GetSealContext();

  apsi::SEALObject<seal::RelinKeys> relin_keys;
  if (seal_context->using_keyswitching()) {
    auto relin_keys_data = query_proto.relin_keys();
    gsl::span<const unsigned char> relin_keys_data_span(
        reinterpret_cast<const unsigned char *>(relin_keys_data.data()),
        relin_keys_data.length());

    relin_keys.load(seal_context, relin_keys_data_span);
  }

  std::unordered_map<uint32_t, std::vector<apsi::SEALObject<seal::Ciphertext>>>
      encrypted_powers;
  for (int idx = 0; idx < query_proto.encrypted_powers_size(); ++idx) {
    const proto::EncryptedPowersProto &encrypted_powers_proto =
        query_proto.encrypted_powers(idx);

    std::vector<apsi::SEALObject<seal::Ciphertext>> ciphertexts;
    ciphertexts.reserve(encrypted_powers_proto.ciphertexts_size());

    for (int cipher_idx = 0;
         cipher_idx < encrypted_powers_proto.ciphertexts_size(); ++cipher_idx) {
      auto ct = encrypted_powers_proto.ciphertexts(cipher_idx);
      gsl::span<const unsigned char> ct_span(
          reinterpret_cast<const unsigned char *>(ct.data()), ct.length());
      apsi::SEALObject<seal::Ciphertext> temp;
      temp.load(seal_context, ct_span);
      ciphertexts.emplace_back(std::move(temp));
    }
    encrypted_powers.emplace(encrypted_powers_proto.power(),
                             std::move(ciphertexts));
  }

  QueryRequest request(&relin_keys, encrypted_powers, sender_db_);

  std::vector<std::shared_ptr<ResultPackage>> query_result =
      SenderRunQuery(request, sender_db_, pd_);

  proto::QueryResponseProto response_proto;
  for (auto &result : query_result) {
    proto::QueryResultProto *result_proto = response_proto.add_results();
    result_proto->set_bundle_idx(result->bundle_idx);
    std::vector<uint8_t> temp;
    temp.resize(result->psi_result.save_size(compr_mode_));
    auto size = result->psi_result.save(temp, compr_mode_);
    result_proto->set_ciphertext(temp.data(), temp.size());
    result_proto->set_label_byte_count(result->label_byte_count);
    result_proto->set_nonce_byte_count(result->nonce_byte_count);

    for (auto &r : result->label_result) {
      temp.resize(r.save_size(compr_mode_));
      size = r.save(temp, compr_mode_);
      result_proto->add_label_results(temp.data(), size);
    }
  }

  yacl::Buffer response_buffer(response_proto.ByteSizeLong());
  response_proto.SerializePartialToArray(response_buffer.data(),
                                         response_buffer.size());

  SPDLOG_DEBUG("response_buffer size:{}, query_result size:{}",
               response_buffer.size(), query_result.size());

  link_ctx->SendAsyncThrottled(
      link_ctx->NextRank(), response_buffer,
      fmt::format("send query response size:{}", response_buffer.size()));
}

void ComputePowers(const std::shared_ptr<spu::psi::SenderDB> &sender_db,
                   const apsi::CryptoContext &crypto_context,
                   std::vector<CiphertextPowers> *all_powers,
                   const apsi::PowersDag &pd, uint32_t bundle_idx,
                   seal::MemoryPoolHandle *pool);

void ProcessBinBundleCache(
    const std::shared_ptr<spu::psi::SenderDB> &sender_db,
    const apsi::CryptoContext &crypto_context,
    const std::shared_ptr<apsi::sender::BinBundle> &bundle,
    std::vector<CiphertextPowers> *all_powers, uint32_t bundle_idx,
    seal::compr_mode_type compr_mode, seal::MemoryPoolHandle *pool,
    const std::shared_ptr<ResultPackage> &result);

std::vector<std::shared_ptr<ResultPackage>> SenderRunQuery(
    const QueryRequest &query,
    const std::shared_ptr<spu::psi::SenderDB> &sender_db,
    const apsi::PowersDag &pd) {
  // We use a custom SEAL memory that is freed after the query is done
  auto pool = seal::MemoryManager::GetPool(seal::mm_force_new);

  apsi::ThreadPoolMgr tpm;

  // Acquire read lock on SenderDB
  // auto sender_db = sender_db;
  auto sender_db_lock = sender_db->GetReaderLock();

  SPDLOG_INFO("Start processing query request on database with {} items",
              sender_db->GetItemCount());

  // Copy over the CryptoContext from SenderDB; set the Evaluator for this local
  // instance. Relinearization keys may not have been included in the query. In
  // that case query.relin_keys() simply holds an empty seal::RelinKeys
  // instance. There is no problem with the below call to
  // CryptoContext::set_evaluator.
  apsi::CryptoContext crypto_context(sender_db->GetCryptoContext());
  crypto_context.set_evaluator(query.relin_keys());

  // Get the PSIParams
  apsi::PSIParams params(sender_db->GetParams());

  uint32_t bundle_idx_count = params.bundle_idx_count();

  uint32_t max_items_per_bin = params.table_params().max_items_per_bin;

  // For each bundle index i, we need a vector of powers of the query Qᵢ. We
  // need powers all the way up to Qᵢ^max_items_per_bin. We don't store the
  // zeroth power. If Paterson-Stockmeyer is used, then only a subset of the
  // powers will be populated.
  std::vector<CiphertextPowers> all_powers(bundle_idx_count);

  // Initialize powers
  for (CiphertextPowers &powers : all_powers) {
    // The + 1 is because we index by power. The 0th power is a dummy value. I
    // promise this makes things easier to read.
    size_t powers_size = static_cast<size_t>(max_items_per_bin) + 1;
    powers.reserve(powers_size);
    for (size_t i = 0; i < powers_size; i++) {
      powers.emplace_back(pool);
    }
  }

  // Load inputs provided in the query
  for (const auto &q : query.data()) {
    // The exponent of all the query powers we're about to iterate through
    auto exponent = static_cast<size_t>(q.first);

    // Load Qᵢᵉ for all bundle indices i, where e is the exponent specified
    // above
    for (size_t bundle_idx = 0; bundle_idx < all_powers.size(); bundle_idx++) {
      // Load input^power to all_powers[bundle_idx][exponent]

      SPDLOG_DEBUG("Extracting query ciphertext power {} for bundle index {}",
                   exponent, bundle_idx);
      all_powers[bundle_idx][exponent] = q.second[bundle_idx];
    }
  }

  // Compute query powers for the bundle indexes
  for (size_t bundle_idx = 0; bundle_idx < bundle_idx_count; bundle_idx++) {
    ComputePowers(sender_db, crypto_context, &all_powers, pd,
                  static_cast<uint32_t>(bundle_idx), &pool);
  }

  SPDLOG_INFO("Finished computing powers for all bundle indices");
  SPDLOG_INFO("Start processing bin bundle caches");

  std::vector<std::shared_ptr<ResultPackage>> query_results;

  for (size_t bundle_idx = 0; bundle_idx < bundle_idx_count; bundle_idx++) {
    size_t cache_count =
        sender_db->GetBinBundleCount(static_cast<uint32_t>(bundle_idx));

    std::vector<std::future<void>> futures;

    for (size_t cache_idx = 0; cache_idx < cache_count; ++cache_idx) {
      std::shared_ptr<apsi::sender::BinBundle> bundle =
          sender_db->GetCacheAt(static_cast<uint32_t>(bundle_idx), cache_idx);

      query_results.push_back(std::make_shared<ResultPackage>());
      std::shared_ptr<ResultPackage> result = *(query_results.rbegin());

      futures.push_back(tpm.thread_pool().enqueue([&, bundle_idx, bundle,
                                                   result]() {
        ProcessBinBundleCache(sender_db, crypto_context, bundle, &all_powers,
                              static_cast<uint32_t>(bundle_idx),
                              query.compr_mode(), &pool, result);
      }));
    }

    // Wait until all bin bundle caches have been processed
    for (auto &f : futures) {
      f.get();
    }
  }

  SPDLOG_INFO("Finished processing query request");

  return query_results;
}

void ComputePowers(const std::shared_ptr<spu::psi::SenderDB> &sender_db,
                   const apsi::CryptoContext &crypto_context,
                   std::vector<CiphertextPowers> *all_powers,
                   const apsi::PowersDag &pd, uint32_t bundle_idx,
                   seal::MemoryPoolHandle *pool) {
  SPDLOG_DEBUG("Sender::ComputePowers");

  // Compute all powers of the query
  SPDLOG_DEBUG("Computing all query ciphertext powers for bundle index {}",
               bundle_idx);

  auto evaluator = crypto_context.evaluator();
  auto relin_keys = crypto_context.relin_keys();

  CiphertextPowers &powers_at_this_bundle_idx = (*all_powers)[bundle_idx];
  bool relinearize = crypto_context.seal_context()->using_keyswitching();
  pd.parallel_apply([&](const apsi::PowersDag::PowersNode &node) {
    if (!node.is_source()) {
      auto parents = node.parents;
      seal::Ciphertext prod(*pool);
      if (parents.first == parents.second) {
        evaluator->square(powers_at_this_bundle_idx[parents.first], prod,
                          *pool);
      } else {
        evaluator->multiply(powers_at_this_bundle_idx[parents.first],
                            powers_at_this_bundle_idx[parents.second], prod,
                            *pool);
      }
      if (relinearize) {
        evaluator->relinearize_inplace(prod, *relin_keys, *pool);
      }
      powers_at_this_bundle_idx[node.power] = std::move(prod);
    }
  });

  // Now that all powers of the ciphertext have been computed, we need to
  // transform them to NTT form. This will substantially improve the polynomial
  // evaluation, because the plaintext polynomials are already in NTT
  // transformed form, and the ciphertexts are used repeatedly for each bin
  // bundle at this index. This computation is separate from the graph
  // processing above, because the multiplications must all be done before
  // transforming to NTT form. We omit the first ciphertext in the vector,
  // because it corresponds to the zeroth power of the query and is included
  // only for convenience of the indexing; the ciphertext is actually not set or
  // valid for use.

  apsi::ThreadPoolMgr tpm;

  // After computing all powers we will modulus switch down to parameters that
  // one more level for low powers than for high powers; same choice must be
  // used when encoding/NTT transforming the SenderDB data.
  auto high_powers_parms_id =
      get_parms_id_for_chain_idx(*crypto_context.seal_context(), 1);
  auto low_powers_parms_id =
      get_parms_id_for_chain_idx(*crypto_context.seal_context(), 2);

  uint32_t ps_low_degree = sender_db->GetParams().query_params().ps_low_degree;

  std::vector<std::future<void>> futures;
  for (uint32_t power : pd.target_powers()) {
    futures.push_back(tpm.thread_pool().enqueue([&, power]() {
      if (ps_low_degree == 0) {
        // Only one ciphertext-plaintext multiplication is needed after this
        evaluator->mod_switch_to_inplace(powers_at_this_bundle_idx[power],
                                         high_powers_parms_id, *pool);

        // All powers must be in NTT form
        evaluator->transform_to_ntt_inplace(powers_at_this_bundle_idx[power]);
      } else {
        if (power <= ps_low_degree) {
          // Low powers must be at a higher level than high powers
          evaluator->mod_switch_to_inplace(powers_at_this_bundle_idx[power],
                                           low_powers_parms_id, *pool);

          // Low powers must be in NTT form
          evaluator->transform_to_ntt_inplace(powers_at_this_bundle_idx[power]);
        } else {
          // High powers are only modulus switched
          evaluator->mod_switch_to_inplace(powers_at_this_bundle_idx[power],
                                           high_powers_parms_id, *pool);
        }
      }
    }));
  }

  for (auto &f : futures) {
    f.get();
  }
}

void ProcessBinBundleCache(
    const std::shared_ptr<spu::psi::SenderDB> &sender_db,
    const apsi::CryptoContext &crypto_context,
    const std::shared_ptr<apsi::sender::BinBundle> &bundle,
    std::vector<CiphertextPowers> *all_powers, uint32_t bundle_idx,
    seal::compr_mode_type compr_mode, seal::MemoryPoolHandle *pool,
    const std::shared_ptr<ResultPackage> &result) {
  SPDLOG_DEBUG("Sender::ProcessBinBundleCache");

  std::reference_wrapper<const apsi::sender::BinBundleCache> cache =
      std::cref(bundle->get_cache());

  // Package for the result data
  result->compr_mode = compr_mode;

  result->bundle_idx = bundle_idx;
  result->nonce_byte_count =
      seal::util::safe_cast<uint32_t>(sender_db->GetNonceByteCount());
  result->label_byte_count =
      seal::util::safe_cast<uint32_t>(sender_db->GetLabelByteCount());

  // Compute the matching result and move to rp
  const apsi::sender::BatchedPlaintextPolyn &matching_polyn =
      cache.get().batched_matching_polyn;

  // Determine if we use Paterson-Stockmeyer or not
  uint32_t ps_low_degree = sender_db->GetParams().query_params().ps_low_degree;
  uint32_t degree =
      seal::util::safe_cast<uint32_t>(matching_polyn.batched_coeffs.size()) - 1;
  bool using_ps = (ps_low_degree > 1) && (ps_low_degree < degree);

  if (using_ps) {
    result->psi_result = matching_polyn.eval_patstock(
        crypto_context, (*all_powers)[bundle_idx],
        seal::util::safe_cast<size_t>(ps_low_degree), *pool);
  } else {
    result->psi_result = matching_polyn.eval((*all_powers)[bundle_idx], *pool);
  }

  for (const auto &interp_polyn : cache.get().batched_interp_polyns) {
    // Compute the label result and move to rp
    degree =
        seal::util::safe_cast<uint32_t>(interp_polyn.batched_coeffs.size()) - 1;
    using_ps = (ps_low_degree > 1) && (ps_low_degree < degree);
    if (using_ps) {
      result->label_result.emplace_back(interp_polyn.eval_patstock(
          crypto_context, (*all_powers)[bundle_idx], ps_low_degree, *pool));
    } else {
      result->label_result.emplace_back(
          interp_polyn.eval((*all_powers)[bundle_idx], *pool));
    }
  }
}

}  // namespace spu::psi
