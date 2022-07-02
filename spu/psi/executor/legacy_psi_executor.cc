// Copyright (c) 2019 Ant Financial. All rights reserved.
#include "spu/psi/executor/legacy_psi_executor.h"

#include <filesystem>
#include <numeric>
#include <type_traits>

#include "spdlog/spdlog.h"
#include "yasl/base/exception.h"
#include "yasl/crypto/hash_util.h"
#include "yasl/utils/parallel.h"
#include "yasl/utils/serialize.h"

#include "spu/psi/core/ecdh_psi.h"
#include "spu/psi/core/ecdh_psi_3party.h"
#include "spu/psi/core/kkrt_psi.h"
#include "spu/psi/core/utils.h"
#include "spu/psi/cryptor/cryptor_selector.h"
#include "spu/psi/io/io.h"
#include "spu/psi/provider/batch_provider_impl.h"
#include "spu/psi/store/cipher_store_impl.h"

namespace spu::psi {

namespace {

constexpr size_t kKkrtReadBatchSize = 4096;
constexpr size_t kKkrtBucketSize = 1 << 20;

constexpr size_t kKkrtPsiSenderRank = 0;
constexpr size_t kKkrtPsiReceiverRank = 1;

constexpr size_t kEcdh3PartyPsiMasterRank = 0;

// how much time we should wait for peer's respond
constexpr size_t kPsiWindowThrottleTimeoutMs = 30 * 60 * 1000;

std::vector<size_t> AllGatherItemsSize(
    const std::shared_ptr<yasl::link::Context>& link_ctx, size_t self_size) {
  std::vector<size_t> items_size_list(link_ctx->WorldSize());

  std::vector<yasl::Buffer> items_size_buf_list = yasl::link::AllGather(
      link_ctx, utils::SerializeSize(self_size), "PSI:SYNC_SIZE");

  for (size_t idx = 0; idx < items_size_buf_list.size(); idx++) {
    items_size_list[idx] = utils::DeserializeSize(items_size_buf_list[idx]);
  }

  return items_size_list;
}

void RunBucketPsi(const PsiOptions& psi_options, const std::string& in_path,
                  std::string& cache_dir, const std::string& psi_protocol,
                  size_t self_items_count, std::vector<unsigned>* indices,
                  bool broadcast_result) {
  YASL_ENFORCE(
      (psi_protocol == kPsiProtocolKkrt) || (psi_protocol == kPsiProtocolEcdh),
      "unsupported protocol={}", psi_protocol);

  std::vector<size_t> items_size_list =
      AllGatherItemsSize(psi_options.link_ctx, self_items_count);

  std::vector<size_t> bucket_count_list(items_size_list.size());
  size_t max_bucket_count = 0;
  size_t min_item_size = self_items_count;

  for (size_t idx = 0; idx < items_size_list.size(); idx++) {
    bucket_count_list[idx] =
        (items_size_list[idx] + kKkrtBucketSize - 1) / kKkrtBucketSize;
    max_bucket_count = std::max(max_bucket_count, bucket_count_list[idx]);
    min_item_size = std::min(min_item_size, items_size_list[idx]);

    SPDLOG_INFO("psi protocol={}, rank={} item_size={}", psi_protocol, idx,
                items_size_list[idx]);
  }

  // one party item_size is 0, no need to do intersection
  if (min_item_size == 0) {
    SPDLOG_INFO("psi protocol={}, min_item_size=0", psi_protocol);
    return;
  }

  SPDLOG_INFO("psi protocol={}, bucket_count={}", psi_protocol,
              max_bucket_count);

  std::shared_ptr<DiskCipherStore> kkrt_bucket_store =
      std::make_shared<DiskCipherStore>(cache_dir, max_bucket_count);

  while (true) {
    auto items = psi_options.batch_provider->ReadNextBatch(kKkrtReadBatchSize);

    // last_batch is empty
    if (items.empty()) {
      break;
    }

    for (const auto& it : items) {
      kkrt_bucket_store->SaveSelf(it);
    }
  }

  kkrt_bucket_store->Finalize();

  size_t kkrt_sender_rank = kKkrtPsiSenderRank;
  size_t kkrt_receiver_rank = kKkrtPsiReceiverRank;

  if (psi_protocol == kPsiProtocolKkrt) {
    size_t self_size = items_size_list[psi_options.link_ctx->Rank()];
    size_t peer_size = items_size_list[psi_options.link_ctx->NextRank()];

    // compare set size, large size as sender
    if (self_size > peer_size) {
      kkrt_sender_rank = psi_options.link_ctx->Rank();
      kkrt_receiver_rank = psi_options.link_ctx->NextRank();
    } else if (self_size < peer_size) {
      kkrt_sender_rank = psi_options.link_ctx->NextRank();
      kkrt_receiver_rank = psi_options.link_ctx->Rank();
    }
  }

  for (size_t bucket_idx = 0; bucket_idx < max_bucket_count; bucket_idx++) {
    SPDLOG_INFO("psi protocol={}, bucket_idx={}", psi_protocol, bucket_idx);

    auto bucket_items_vec = kkrt_bucket_store->LoadSelfBinFile(bucket_idx);

    size_t min_bucket_item_size = bucket_items_vec.size();
    std::vector<size_t> bucket_items_size_list =
        AllGatherItemsSize(psi_options.link_ctx, bucket_items_vec.size());
    for (size_t idx = 0; idx < bucket_items_size_list.size(); idx++) {
      SPDLOG_INFO(
          "psi protocol={}, bucket_idx={}, rank={}, "
          "bucket_item_size={}",
          psi_protocol, bucket_idx, idx, bucket_items_size_list[idx]);

      // find min_bucket_item_size
      min_bucket_item_size =
          std::min(min_bucket_item_size, bucket_items_size_list[idx]);
    }
    // no need do intersection
    if (min_bucket_item_size == 0) {
      SPDLOG_INFO(
          "psi protocol={}, bucket_idx={}, min_bucket_item_size=0, "
          "no need do intersection",
          psi_protocol, bucket_idx);
      continue;
    }

    std::vector<uint128_t> bucket_items_hash(bucket_items_vec.size());

    std::vector<std::string> bucket_items_string(bucket_items_vec.size());

    size_t bucket_items_size = bucket_items_vec.size();

    yasl::parallel_for(
        0, bucket_items_size, 1, [&](int64_t begin, int64_t end) {
          for (int64_t idx = begin; idx < end; ++idx) {
            bucket_items_hash[idx] =
                yasl::crypto::Blake3_128(bucket_items_vec[idx].item);
            bucket_items_string[idx] = bucket_items_vec[idx].item;
          }
        });

    std::vector<size_t> psi_result_size_list;
    std::vector<unsigned> bucket_indices;
    std::vector<uint128_t> intersection_hash_list;
    size_t broadcast_rank = 0;

    if (psi_protocol == kPsiProtocolEcdh) {
      std::vector<std::string> ecdh_3party_psi_result = RunShuffleEcdhPsi3Party(
          psi_options.link_ctx, kEcdh3PartyPsiMasterRank, bucket_items_string);

      psi_result_size_list = AllGatherItemsSize(psi_options.link_ctx,
                                                ecdh_3party_psi_result.size());

      if (ecdh_3party_psi_result.size() > 0) {
        std::sort(ecdh_3party_psi_result.begin(), ecdh_3party_psi_result.end());

        for (size_t k = 0; k < bucket_items_vec.size(); ++k) {
          if (std::binary_search(ecdh_3party_psi_result.begin(),
                                 ecdh_3party_psi_result.end(),
                                 bucket_items_vec[k].item)) {
            bucket_indices.push_back(bucket_items_vec[k].index);
            intersection_hash_list.push_back(bucket_items_hash[k]);
          }
        }
      }
      broadcast_rank = kEcdh3PartyPsiMasterRank;
    } else if (psi_protocol == kPsiProtocolKkrt) {
      if (kkrt_sender_rank == psi_options.link_ctx->Rank()) {
        yasl::BaseRecvOptions recv_opts;

        GetKkrtOtSenderOptions(psi_options.link_ctx, 512, &recv_opts);

        KkrtPsiSend(psi_options.link_ctx, recv_opts, bucket_items_hash);

        psi_result_size_list = AllGatherItemsSize(psi_options.link_ctx, 0);

      } else if (kkrt_receiver_rank == psi_options.link_ctx->Rank()) {
        yasl::BaseSendOptions send_opts;

        GetKkrtOtReceiverOptions(psi_options.link_ctx, 512, &send_opts);

        std::vector<size_t> kkrt_psi_result =
            KkrtPsiRecv(psi_options.link_ctx, send_opts, bucket_items_hash);

        psi_result_size_list =
            AllGatherItemsSize(psi_options.link_ctx, kkrt_psi_result.size());

        // send intersection hash to peer
        if (kkrt_psi_result.size() > 0) {
          intersection_hash_list.resize(kkrt_psi_result.size());
          bucket_indices.resize(kkrt_psi_result.size());

          for (size_t k = 0; k < kkrt_psi_result.size(); ++k) {
            intersection_hash_list[k] = bucket_items_hash[kkrt_psi_result[k]];
            bucket_indices[k] = bucket_items_vec[kkrt_psi_result[k]].index;
          }
        }
      }
      broadcast_rank = kkrt_receiver_rank;
    }

    size_t bucket_max_psi_result_size = 0;
    for (size_t idx = 0; idx < psi_result_size_list.size(); idx++) {
      bucket_max_psi_result_size =
          std::max(bucket_max_psi_result_size, psi_result_size_list[idx]);
    }

    SPDLOG_INFO(
        "psi protocol={}, bucket_idx={}, "
        "bucket_psi_result_size={}",
        psi_protocol, bucket_idx, bucket_max_psi_result_size);

    // check max psi result,
    //   case result 0, no need to broadcast psi result
    if (bucket_max_psi_result_size == 0) {
      continue;
    }

    if (broadcast_result == false) {
      std::sort(bucket_indices.begin(), bucket_indices.end());
      indices->insert(indices->end(), bucket_indices.begin(),
                      bucket_indices.end());
      continue;
    }

    if (psi_result_size_list[psi_options.link_ctx->Rank()] > 0) {
      yasl::link::Broadcast(
          psi_options.link_ctx,
          {reinterpret_cast<const std::byte*>(intersection_hash_list.data()),
           intersection_hash_list.size() * sizeof(uint128_t)},
          broadcast_rank, "send intersection hash");

      SPDLOG_INFO(
          "psi protocol={}, bucket_idx={}, rank={}, "
          "broadcast send intersection_hash_buf size={}",
          psi_protocol, bucket_idx, psi_options.link_ctx->Rank(),
          intersection_hash_list.size() * sizeof(uint128_t));

    } else {
      auto intersection_hash_buf = yasl::link::Broadcast(
          psi_options.link_ctx, {}, broadcast_rank, "recv intersection hash");

      SPDLOG_INFO(
          "psi protocol={}, bucket_idx={}, rank={}, "
          "broadcast recv intersection_hash_buf size={}",
          psi_protocol, bucket_idx, psi_options.link_ctx->Rank(),
          intersection_hash_buf.size());

      YASL_ENFORCE((intersection_hash_buf.size() % sizeof(uint128_t) == 0),
                   "recv intersection_hash_buf size:{} not align protocol={}, "
                   "rank:{}",
                   intersection_hash_buf.size(), psi_protocol,
                   psi_options.link_ctx->Rank());

      intersection_hash_list.resize(intersection_hash_buf.size() /
                                    sizeof(uint128_t));

      std::memcpy(intersection_hash_list.data(), intersection_hash_buf.data(),
                  intersection_hash_buf.size());

      std::sort(intersection_hash_list.begin(), intersection_hash_list.end());
      for (uint64_t index = 0; index < bucket_items_hash.size(); index++) {
        if (std::binary_search(intersection_hash_list.begin(),
                               intersection_hash_list.end(),
                               bucket_items_hash[index])) {
          bucket_indices.push_back(bucket_items_vec[index].index);
        }
      }
    }

    std::sort(bucket_indices.begin(), bucket_indices.end());
    indices->insert(indices->end(), bucket_indices.begin(),
                    bucket_indices.end());
  }

  // sort indices
  std::sort(indices->begin(), indices->end());
}

}  // namespace

LegacyPsiExecutor::LegacyPsiExecutor(LegacyPsiOptions options)
    : PsiExecutorBase(std::move(options.base_options)),
      psi_protocol_(std::move(options.psi_protocol)),
      num_bins_(options.num_bins),
      broadcast_result_(options.broadcast_result) {}

void LegacyPsiExecutor::OnRun(std::vector<unsigned>* indices) {
  PsiOptions psi_options;
  std::string cache_dir =
      std::filesystem::path(options_.out_path).parent_path();
  {
    psi_options.batch_provider = std::make_shared<CsvBatchProvider>(
        options_.in_path, options_.field_names);
    psi_options.ecc_cryptor = CreateEccCryptor(CurveType::Curve25519);
    psi_options.link_ctx = options_.link_ctx;
    psi_options.target_rank = yasl::link::kAllRank;
    psi_options.window_throttle_timeout_ms = kPsiWindowThrottleTimeoutMs;
  }

  if ((psi_protocol_ == kPsiProtocolEcdh) ||
      (psi_protocol_ == kPsiProtocolKkrt)) {
    if (psi_protocol_ == kPsiProtocolEcdh) {
      YASL_ENFORCE(psi_options.link_ctx->WorldSize() == 3);
    } else if (psi_protocol_ == kPsiProtocolKkrt) {
      YASL_ENFORCE(psi_options.link_ctx->WorldSize() == 2);
    }

    RunBucketPsi(psi_options, options_.in_path, cache_dir, psi_protocol_,
                 input_data_count_, indices, broadcast_result_);

  } else if (psi_protocol_ == kPsiProtocolEcdh2PC) {
    std::shared_ptr<DiskCipherStore> cipher_store =
        std::make_shared<DiskCipherStore>(cache_dir, num_bins_);
    psi_options.cipher_store = cipher_store;

    // Launch ECDH-PSI core.
    RunEcdhPsi(psi_options);

    // We are using unsigned as index, the PSI data scale should be limited
    // to 2^32, i.e. about 4 billion
    *indices = cipher_store->FinalizeAndComputeIndices();
  } else {
    YASL_THROW("not support protocol={}", psi_protocol_);
  }
}

void LegacyPsiExecutor::OnStop() { YASL_THROW_LOGIC_ERROR("Not implemented"); }

void LegacyPsiExecutor::OnInit() {
  // options sanity check.
  if (psi_protocol_ == kPsiProtocolEcdh) {
    if (options_.link_ctx->WorldSize() != 3) {
      YASL_THROW(
          "psi_protocol:{}, only three parties supported, got "
          "{}",
          psi_protocol_, options_.link_ctx->WorldSize());
    }
  } else {
    if (options_.link_ctx->WorldSize() != 2) {
      YASL_THROW("psi_protocol:{}, only two parties supported, got {}",
                 psi_protocol_, options_.link_ctx->WorldSize());
    }
  }
}

}  // namespace spu::psi
