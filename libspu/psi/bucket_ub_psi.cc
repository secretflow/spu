// Copyright 2023 Ant Group Co., Ltd.
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

#include "libspu/psi/bucket_ub_psi.h"

#include <unordered_set>

#include "yacl/crypto/utils/rand.h"

#include "libspu/psi/io/io.h"
#include "libspu/psi/utils/serialize.h"
#include "libspu/psi/utils/utils.h"

namespace spu::psi {

namespace {

std::vector<uint8_t> ReadEcSecretKeyFile(const std::string& file_path) {
  size_t file_byte_size = 0;
  try {
    file_byte_size = std::filesystem::file_size(file_path);
  } catch (std::filesystem::filesystem_error& e) {
    SPU_THROW("ReadEcSecretKeyFile {} Error: {}", file_path, e.what());
  }
  SPU_ENFORCE(file_byte_size == kEccKeySize,
              "error format: key file bytes is not {}", kEccKeySize);

  std::vector<uint8_t> secret_key(kEccKeySize);

  auto in = io::BuildInputStream(io::FileIoOptions(file_path));
  in->Read(secret_key.data(), kEccKeySize);
  in->Close();

  return secret_key;
}

std::vector<std::string> GetItemsByIndices(
    const std::string& input_path,
    const std::vector<std::string>& selected_fields,
    const std::vector<uint64_t>& indices, size_t batch_size = 8192) {
  std::vector<std::string> items;
  std::unordered_set<size_t> indices_set;

  indices_set.insert(indices.begin(), indices.end());

  std::shared_ptr<IBatchProvider> batch_provider =
      std::make_shared<CsvBatchProvider>(input_path, selected_fields);

  size_t item_index = 0;
  while (true) {
    auto batch_items = batch_provider->ReadNextBatch(batch_size);
    if (batch_items.empty()) {
      break;
    }
    for (size_t i = 0; i < batch_items.size(); ++i) {
      if (indices_set.find(item_index) != indices_set.end()) {
        items.push_back(batch_items[i]);
      }
      item_index++;
    }
  }
  return items;
}

}  // namespace

std::pair<std::vector<uint64_t>, size_t> UbPsi(
    BucketPsiConfig config, std::shared_ptr<yacl::link::Context> lctx) {
  spu::psi::EcdhOprfPsiOptions psi_options;
  psi_options.link0 = lctx;

  if (config.psi_type() == PsiType::ECDH_OPRF_UB_PSI_2PC_GEN_CACHE) {
    psi_options.link1 = lctx;
  } else {
    psi_options.link1 = lctx->Spawn();
  }

  psi_options.curve_type = config.curve_type();

  std::string tmp_dir =
      fmt::format("bucket_tmp_{}", yacl::crypto::SecureRandU64());
  std::filesystem::create_directory(tmp_dir);

  // register remove of temp dir.
  ON_SCOPE_EXIT([&] {
    if (!tmp_dir.empty()) {
      std::error_code ec;
      std::filesystem::remove_all(tmp_dir, ec);
      if (ec.value() != 0) {
        SPDLOG_WARN("can not remove tmp dir: {}, msg: {}", tmp_dir,
                    ec.message());
      }
    }
  });

  SPDLOG_INFO("input file path:{}", config.input_params().path());
  SPDLOG_INFO("output file path:{}", config.output_params().path());

  std::pair<std::vector<uint64_t>, size_t> results;

  switch (config.psi_type()) {
    case PsiType::ECDH_OPRF_UB_PSI_2PC_GEN_CACHE:
      results = UbPsiServerGenCache(config, lctx, psi_options);
      break;
    case PsiType::ECDH_OPRF_UB_PSI_2PC_TRANSFER_CACHE:
      if (lctx->Rank() == config.receiver_rank()) {
        results = UbPsiClientTransferCache(config, lctx, psi_options, tmp_dir);
      } else {
        results = UbPsiServerTransferCache(config, lctx, psi_options, tmp_dir);
      }
      break;
    case PsiType::ECDH_OPRF_UB_PSI_2PC_SHUFFLE_ONLINE:
      if (lctx->Rank() == config.receiver_rank()) {
        results = UbPsiServerShuffleOnline(config, lctx, psi_options, tmp_dir);
      } else {
        results = UbPsiClientShuffleOnline(config, lctx, psi_options, tmp_dir);
      }
      break;
    case PsiType::ECDH_OPRF_UB_PSI_2PC_OFFLINE:
      if (lctx->Rank() == config.receiver_rank()) {
        results = UbPsiClientOffline(config, lctx, psi_options, tmp_dir);
      } else {
        results = UbPsiServerOffline(config, lctx, psi_options, tmp_dir);
      }
      break;
    case PsiType::ECDH_OPRF_UB_PSI_2PC_ONLINE:
      if (lctx->Rank() == config.receiver_rank()) {
        results = UbPsiClientOnline(config, lctx, psi_options, tmp_dir);
      } else {
        results = UbPsiServerOnline(config, lctx, psi_options, tmp_dir);
      }
      break;
    default:
      SPU_THROW("Invalid unbalanced psi subprotocol: {}", config.psi_type());
  }

  if (config.psi_type() != PsiType::ECDH_OPRF_UB_PSI_2PC_GEN_CACHE) {
    SPDLOG_INFO("rank:{} Start end sync", lctx->Rank());
    AllGatherItemsSize(lctx, 0);
    SPDLOG_INFO("rank:{} After end sync", lctx->Rank());
  }

  return results;
}

// generate cache
std::pair<std::vector<uint64_t>, size_t> UbPsiServerGenCache(
    BucketPsiConfig config, std::shared_ptr<yacl::link::Context> lctx,
    const spu::psi::EcdhOprfPsiOptions& psi_options) {
  std::vector<uint8_t> server_private_key =
      ReadEcSecretKeyFile(config.ecdh_secret_key_path());

  std::shared_ptr<EcdhOprfPsiServer> dh_oprf_psi_server =
      std::make_shared<EcdhOprfPsiServer>(psi_options, server_private_key);

  std::vector<std::string> selected_fields;
  selected_fields.insert(selected_fields.end(),
                         config.input_params().select_fields().begin(),
                         config.input_params().select_fields().end());

  std::shared_ptr<IShuffleBatchProvider> batch_provider =
      std::make_shared<CachedCsvBatchProvider>(config.input_params().path(),
                                               selected_fields,
                                               config.bucket_size(), true);

  std::shared_ptr<IUbPsiCache> ub_cache = std::make_shared<UbPsiCache>(
      config.output_params().path(), dh_oprf_psi_server->GetCompareLength(),
      selected_fields);

  size_t self_items_count =
      dh_oprf_psi_server->FullEvaluate(batch_provider, ub_cache);

  std::vector<uint64_t> results;

  return std::make_pair(results, self_items_count);
}

// transfer cache
std::pair<std::vector<uint64_t>, size_t> UbPsiClientTransferCache(
    BucketPsiConfig config, std::shared_ptr<yacl::link::Context> lctx,
    const spu::psi::EcdhOprfPsiOptions& psi_options,
    const std::string& tmp_dir) {
  std::shared_ptr<EcdhOprfPsiClient> ub_psi_client_transfer_cache =
      std::make_shared<EcdhOprfPsiClient>(psi_options);

  std::string self_cipher_store_path =
      fmt::format("{}/tmp-self-cipher-store-{}.csv", tmp_dir, lctx->Rank());

  std::shared_ptr<CachedCsvCipherStore> cipher_store =
      std::make_shared<CachedCsvCipherStore>(
          self_cipher_store_path, config.preprocess_path(), false, false);

  SPDLOG_INFO("Start Sync");
  AllGatherItemsSize(lctx, 0);
  SPDLOG_INFO("After Sync");

  ub_psi_client_transfer_cache->RecvFinalEvaluatedItems(cipher_store);

  cipher_store->FlushPeer();

  std::vector<uint64_t> results;

  return std::make_pair(results, 0);
}

std::pair<std::vector<uint64_t>, size_t> UbPsiServerTransferCache(
    BucketPsiConfig config, std::shared_ptr<yacl::link::Context> lctx,
    const spu::psi::EcdhOprfPsiOptions& psi_options,
    const std::string& tmp_dir) {
  std::array<uint8_t, spu::psi::kEccKeySize> server_private_key;

  std::shared_ptr<EcdhOprfPsiServer> ub_psi_server_transfer_cache =
      std::make_shared<EcdhOprfPsiServer>(psi_options, server_private_key);

  std::shared_ptr<IBatchProvider> batch_provider =
      std::make_shared<UbPsiCacheProvider>(
          config.input_params().path(),
          ub_psi_server_transfer_cache->GetCompareLength());

  SPDLOG_INFO("Start sync");
  AllGatherItemsSize(lctx, 0);
  SPDLOG_INFO("After sync");

  size_t self_items_count =
      ub_psi_server_transfer_cache->SendFinalEvaluatedItems(batch_provider);

  std::vector<uint64_t> results;

  return std::make_pair(results, self_items_count);
}

// online with shuffling
std::pair<std::vector<uint64_t>, size_t> UbPsiClientShuffleOnline(
    BucketPsiConfig config, std::shared_ptr<yacl::link::Context> lctx,
    const spu::psi::EcdhOprfPsiOptions& psi_options,
    const std::string& tmp_dir) {
  std::vector<uint8_t> private_key = yacl::crypto::RandBytes(kEccKeySize);

  std::shared_ptr<EcdhOprfPsiClient> ub_psi_client_shuffle_online =
      std::make_shared<EcdhOprfPsiClient>(psi_options, private_key);

  std::vector<std::string> selected_fields;
  selected_fields.insert(selected_fields.end(),
                         config.input_params().select_fields().begin(),
                         config.input_params().select_fields().end());

  std::shared_ptr<IBatchProvider> batch_provider =
      std::make_shared<CsvBatchProvider>(config.input_params().path(),
                                         selected_fields);

  std::string cipher_store_path1 =
      fmt::format("{}/tmp-self-cipher-store-{}.csv", tmp_dir, lctx->Rank());

  std::shared_ptr<CachedCsvCipherStore> cipher_store =
      std::make_shared<CachedCsvCipherStore>(
          cipher_store_path1, config.preprocess_path(), false, true);

  SPDLOG_INFO("shuffle online protocol CachedCsvCipherStore: {} {}",
              cipher_store_path1, config.preprocess_path());

  size_t self_items_count =
      ub_psi_client_shuffle_online->SendBlindedItems(batch_provider);

  ub_psi_client_shuffle_online->RecvEvaluatedItems(cipher_store);

  std::vector<uint64_t> indices;
  std::vector<std::string> masked_items;
  std::tie(indices, masked_items) =
      cipher_store->FinalizeAndComputeIndices(config.bucket_size());
  SPDLOG_INFO("indices size:{}", indices.size());

  std::shared_ptr<IBatchProvider> intersection_masked_provider =
      std::make_shared<MemoryBatchProvider>(masked_items);
  ub_psi_client_shuffle_online->SendIntersectionMaskedItems(
      intersection_masked_provider);

  std::vector<uint64_t> results;

  return std::make_pair(results, self_items_count);
}

std::pair<std::vector<uint64_t>, size_t> UbPsiServerShuffleOnline(
    BucketPsiConfig config, std::shared_ptr<yacl::link::Context> lctx,
    const spu::psi::EcdhOprfPsiOptions& psi_options,
    const std::string& tmp_dir) {
  std::vector<uint8_t> server_private_key =
      ReadEcSecretKeyFile(config.ecdh_secret_key_path());

  std::shared_ptr<EcdhOprfPsiServer> ub_psi_server_shuffle_online =
      std::make_shared<EcdhOprfPsiServer>(psi_options, server_private_key);

  ub_psi_server_shuffle_online->RecvBlindAndShuffleSendEvaluate();

  std::shared_ptr<IShuffleBatchProvider> cache_provider =
      std::make_shared<UbPsiCacheProvider>(
          config.preprocess_path(),
          ub_psi_server_shuffle_online->GetCompareLength());

  size_t self_items_size;
  std::vector<uint64_t> results;
  std::tie(results, self_items_size) =
      ub_psi_server_shuffle_online->RecvIntersectionMaskedItems(
          cache_provider, config.bucket_size());

  return std::make_pair(results, self_items_size);
}

// offline
std::pair<std::vector<uint64_t>, size_t> UbPsiClientOffline(
    BucketPsiConfig config, std::shared_ptr<yacl::link::Context> lctx,
    const spu::psi::EcdhOprfPsiOptions& psi_options,
    const std::string& tmp_dir) {
  std::shared_ptr<EcdhOprfPsiClient> dh_oprf_psi_client_offline =
      std::make_shared<EcdhOprfPsiClient>(psi_options);

  std::string self_cipher_store_path =
      fmt::format("{}/tmp-self-cipher-store-{}.csv", tmp_dir, lctx->Rank());

  std::shared_ptr<CachedCsvCipherStore> cipher_store =
      std::make_shared<CachedCsvCipherStore>(
          self_cipher_store_path, config.preprocess_path(), false, false);

  SPDLOG_INFO("Start Sync");
  AllGatherItemsSize(lctx, 0);
  SPDLOG_INFO("After Sync");

  dh_oprf_psi_client_offline->RecvFinalEvaluatedItems(cipher_store);

  cipher_store->FlushPeer();

  std::vector<uint64_t> results;

  return std::make_pair(results, 0);
}

std::pair<std::vector<uint64_t>, size_t> UbPsiServerOffline(
    BucketPsiConfig config, std::shared_ptr<yacl::link::Context> lctx,
    const spu::psi::EcdhOprfPsiOptions& psi_options,
    const std::string& tmp_dir) {
  std::vector<uint8_t> server_private_key =
      ReadEcSecretKeyFile(config.ecdh_secret_key_path());

  std::shared_ptr<EcdhOprfPsiServer> dh_oprf_psi_server_offline =
      std::make_shared<EcdhOprfPsiServer>(psi_options, server_private_key);

  std::vector<std::string> selected_fields;
  selected_fields.insert(selected_fields.end(),
                         config.input_params().select_fields().begin(),
                         config.input_params().select_fields().end());

  std::shared_ptr<IShuffleBatchProvider> batch_provider =
      std::make_shared<CachedCsvBatchProvider>(config.input_params().path(),
                                               selected_fields,
                                               config.bucket_size(), true);

  SPDLOG_INFO("Start sync");
  AllGatherItemsSize(lctx, 0);
  SPDLOG_INFO("After sync");

  size_t self_items_count =
      dh_oprf_psi_server_offline->FullEvaluateAndSend(batch_provider);

  std::vector<uint64_t> results;

  return std::make_pair(results, self_items_count);
}

// online
std::pair<std::vector<uint64_t>, size_t> UbPsiClientOnline(
    BucketPsiConfig config, std::shared_ptr<yacl::link::Context> lctx,
    const spu::psi::EcdhOprfPsiOptions& psi_options,
    const std::string& tmp_dir) {
  std::shared_ptr<EcdhOprfPsiClient> dh_oprf_psi_client_online =
      std::make_shared<EcdhOprfPsiClient>(psi_options);

  std::vector<std::string> selected_fields;
  selected_fields.insert(selected_fields.end(),
                         config.input_params().select_fields().begin(),
                         config.input_params().select_fields().end());

  std::shared_ptr<IBatchProvider> batch_provider =
      std::make_shared<CsvBatchProvider>(config.input_params().path(),
                                         selected_fields);

  std::string cipher_store_path1 =
      fmt::format("{}/tmp-self-cipher-store-{}.csv", tmp_dir, lctx->Rank());

  std::shared_ptr<CachedCsvCipherStore> cipher_store =
      std::make_shared<CachedCsvCipherStore>(
          cipher_store_path1, config.preprocess_path(), false, true);

  SPDLOG_INFO("online protocol CachedCsvCipherStore: {} {}", cipher_store_path1,
              config.preprocess_path());

  std::future<size_t> f_client_send_blind = std::async([&] {
    return dh_oprf_psi_client_online->SendBlindedItems(batch_provider);
  });

  dh_oprf_psi_client_online->RecvEvaluatedItems(cipher_store);

  size_t self_items_count = f_client_send_blind.get();

  std::vector<uint64_t> results;
  std::vector<std::string> masked_items;
  std::tie(results, masked_items) =
      cipher_store->FinalizeAndComputeIndices(config.bucket_size());

  SPU_ENFORCE(results.size() == masked_items.size());
  SPDLOG_INFO("indices size:{}", results.size());

  if (config.broadcast_result()) {
    // send intersection size
    lctx->SendAsyncThrottled(
        lctx->NextRank(), utils::SerializeSize(results.size()),
        fmt::format("EC-OPRF:PSI:INTERSECTION_SIZE={}", results.size()));

    SPDLOG_INFO("rank:{} begin broadcast {} intersection results", lctx->Rank(),
                results.size());

    if (results.size() > 0) {
      std::vector<std::string> result_items = GetItemsByIndices(
          config.input_params().path(), selected_fields, results);

      auto recv_res_buf =
          yacl::link::Broadcast(lctx, utils::SerializeStrItems(result_items),
                                config.receiver_rank(), "broadcast psi result");

      SPDLOG_INFO("rank:{} result size:{}", lctx->Rank(), result_items.size());
    }
    SPDLOG_INFO("rank:{} end broadcast {} intersection results", lctx->Rank(),
                results.size());
  }

  return std::make_pair(results, self_items_count);
}

std::pair<std::vector<uint64_t>, size_t> UbPsiServerOnline(
    BucketPsiConfig config, std::shared_ptr<yacl::link::Context> lctx,
    const spu::psi::EcdhOprfPsiOptions& psi_options,
    const std::string& tmp_dir) {
  std::vector<uint8_t> server_private_key =
      ReadEcSecretKeyFile(config.ecdh_secret_key_path());

  std::shared_ptr<EcdhOprfPsiServer> dh_oprf_psi_server_online =
      std::make_shared<EcdhOprfPsiServer>(psi_options, server_private_key);

  dh_oprf_psi_server_online->RecvBlindAndSendEvaluate();

  std::vector<uint64_t> results;

  if (config.broadcast_result()) {
    size_t intersection_size = utils::DeserializeSize(lctx->Recv(
        lctx->NextRank(), fmt::format("EC-OPRF:PSI:INTERSECTION_SIZE")));

    SPDLOG_INFO("rank:{} begin recv broadcast {} intersection results",
                lctx->Rank(), results.size(), intersection_size);

    if (intersection_size > 0) {
      std::vector<std::string> result_items;

      auto recv_res_buf =
          yacl::link::Broadcast(lctx, utils::SerializeStrItems(result_items),
                                config.receiver_rank(), "broadcast psi result");

      utils::DeserializeStrItems(recv_res_buf, &result_items);

      std::vector<std::string> selected_fields;
      selected_fields.insert(selected_fields.end(),
                             config.input_params().select_fields().begin(),
                             config.input_params().select_fields().end());

      SPDLOG_INFO("begin GetIndicesByItems");
      results = GetIndicesByItems(config.input_params().path(), selected_fields,
                                  result_items, config.bucket_size());
      SPDLOG_INFO("end GetIndicesByItems");

      SPDLOG_INFO("rank:{} result size:{}", lctx->Rank(), result_items.size());
    }
    SPDLOG_INFO("rank:{} end recv broadcast {} intersection results",
                lctx->Rank(), results.size());
  }

  return std::make_pair(results, 0);
}

}  // namespace spu::psi
