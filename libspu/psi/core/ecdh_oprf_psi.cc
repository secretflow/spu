// Copyright 2022 Ant Group Co., Ltd.
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

#include "libspu/psi/core/ecdh_oprf_psi.h"

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <future>
#include <unordered_set>
#include <utility>

#include "absl/strings/escaping.h"
#include "yacl/utils/parallel.h"
#include "yacl/utils/serialize.h"

#include "libspu/core/prelude.h"
#include "libspu/psi/core/communication.h"
#include "libspu/psi/cryptor/ecc_utils.h"
#include "libspu/psi/utils/ub_psi_cache.h"

namespace spu::psi {

size_t EcdhOprfPsiServer::FullEvaluateAndSend(
    const std::shared_ptr<IShuffleBatchProvider>& batch_provider,
    const std::shared_ptr<IUbPsiCache>& ub_cache) {
  return FullEvaluate(batch_provider, ub_cache, true);
}

size_t EcdhOprfPsiServer::SendFinalEvaluatedItems(
    const std::shared_ptr<IBatchProvider>& batch_provider) {
  size_t items_count = 0;
  size_t batch_count = 0;

  size_t compare_length = oprf_server_->GetCompareLength();

  while (true) {
    PsiDataBatch batch;
    auto items = batch_provider->ReadNextBatch(options_.batch_size);
    batch.is_last_batch = items.empty();

    if (!batch.is_last_batch) {
      batch.flatten_bytes.reserve(items.size() * compare_length);

      for (const auto& item : items) {
        batch.flatten_bytes.append(item);
      }
    }

    // Send x^a.
    const auto tag =
        fmt::format("EcdhOprfPSI:FinalEvaluatedItems:{}", batch_count);
    options_.link0->SendAsyncThrottled(options_.link0->NextRank(),
                                       batch.Serialize(), tag);

    if (batch.is_last_batch) {
      SPDLOG_INFO("{} Last batch triggered, batch_count={}", __func__,
                  batch_count);
      break;
    }

    items_count += items.size();
    batch_count++;
  }

  SPDLOG_INFO("{} finished, batch_count={}", __func__, batch_count);

  return items_count;
}

size_t EcdhOprfPsiServer::FullEvaluate(
    const std::shared_ptr<IShuffleBatchProvider>& batch_provider,
    const std::shared_ptr<IUbPsiCache>& ub_cache, bool send_flag) {
  size_t items_count = 0;
  size_t batch_count = 1;
  size_t compare_length = oprf_server_->GetCompareLength();

  bool stop_flag = false;
  omp_lock_t lck_read, lck_send;

  omp_init_lock(&lck_read);
  omp_init_lock(&lck_send);

  int tid;
  batch_count = 0;
  std::vector<std::string> batch_items;
  std::vector<size_t> batch_indices;
  std::vector<size_t> shuffle_indices;
  PsiDataBatch batch;
  size_t i;
  size_t local_batch_count;
  int nthreads = omp_get_num_threads();
  int mcpus = omp_get_num_procs();
  SPDLOG_INFO("omp_get_num_threads:{} cpus:{}", nthreads, mcpus);
  omp_set_num_threads(mcpus);

#pragma omp parallel private(tid, nthreads, i, batch_items, batch_indices, \
                             shuffle_indices, batch, local_batch_count)    \
    shared(lck_read, lck_send, batch_count, items_count, compare_length,   \
           stop_flag)
  {
    tid = omp_get_thread_num();
    if ((tid == 0) && (batch_count == 0)) {
      nthreads = omp_get_num_threads();
      SPDLOG_INFO("tid:{} omp_get_num_threads:{}", tid, nthreads);
    }
    while (!stop_flag) {
      omp_set_lock(&lck_read);

      if (stop_flag) {
        omp_unset_lock(&lck_read);
        break;
      }
      std::tie(batch_items, batch_indices, shuffle_indices) =
          batch_provider->ReadNextBatchWithIndex(options_.batch_size);

      batch.is_last_batch = batch_items.empty();

      if (batch_items.empty()) {
        stop_flag = true;
      } else {
        items_count += batch_items.size();
        batch_count++;
        if ((batch_count % 1000) == 0) {
          SPDLOG_INFO("batch_count:{}", batch_count);
        }
        local_batch_count = batch_count;
      }

      omp_unset_lock(&lck_read);

      if (batch_items.size() == 0) {
        break;
      }

      batch.flatten_bytes.reserve(batch_items.size() * compare_length);

      batch.flatten_bytes = oprf_server_->SimpleEvaluate(batch_items[0]);
      for (i = 1; i < batch_items.size(); i++) {
        std::string masked_item = oprf_server_->SimpleEvaluate(batch_items[i]);
        batch.flatten_bytes.append(masked_item);
      }

      omp_set_lock(&lck_send);

      if (send_flag) {
        // Send x^a.
        options_.link0->SendAsyncThrottled(
            options_.link0->NextRank(), batch.Serialize(),
            fmt::format("EcdhOprfPSI:FinalEvaluatedItems:{}",
                        local_batch_count));
      }

      if (ub_cache != nullptr) {
        for (size_t i = 0; i < batch_items.size(); i++) {
          std::string cache_data =
              batch.flatten_bytes.substr(i * compare_length, compare_length);
          ub_cache->SaveData(cache_data, batch_indices[i], shuffle_indices[i]);
        }
      }

      omp_unset_lock(&lck_send);
    }
  }

  if (send_flag) {
    batch.is_last_batch = true;
    batch.flatten_bytes.resize(0);
    options_.link0->SendAsyncThrottled(
        options_.link0->NextRank(), batch.Serialize(),
        fmt::format("EcdhOprfPSI last batch,FinalEvaluatedItems:{}",
                    batch_count));
  }
  if (ub_cache != nullptr) {
    ub_cache->Flush();
  }

  SPDLOG_INFO("{} finished, batch_count={} items_count={}", __func__,
              batch_count, items_count);

  return items_count;
}

void EcdhOprfPsiServer::RecvBlindAndSendEvaluate() {
  size_t batch_count = 0;

  size_t ec_point_length = oprf_server_->GetEcPointLength();

  while (true) {
    const auto tag = fmt::format("EcdhOprfPSI:BlindItems:{}", batch_count);
    PsiDataBatch blinded_batch = PsiDataBatch::Deserialize(
        options_.link1->Recv(options_.link1->NextRank(), tag));

    PsiDataBatch evaluated_batch;
    evaluated_batch.is_last_batch = blinded_batch.is_last_batch;

    const auto tag_send =
        fmt::format("EcdhOprfPSI:EvaluatedItems:{}", batch_count);

    if (blinded_batch.is_last_batch) {
      SPDLOG_INFO("{} Last batch triggered, batch_count={}", __func__,
                  batch_count);

      options_.link1->SendAsyncThrottled(options_.link1->NextRank(),
                                         evaluated_batch.Serialize(), tag_send);
      break;
    }

    // Fetch blinded y^r.
    SPU_ENFORCE(blinded_batch.flatten_bytes.size() % ec_point_length == 0);
    size_t num_items = blinded_batch.flatten_bytes.size() / ec_point_length;

    std::vector<std::string> blinded_items(num_items);
    for (size_t idx = 0; idx < num_items; ++idx) {
      blinded_items[idx] = blinded_batch.flatten_bytes.substr(
          idx * ec_point_length, ec_point_length);
    }
    // (x^r)^s
    std::vector<std::string> evaluated_items =
        oprf_server_->Evaluate(blinded_items);

    evaluated_batch.flatten_bytes.reserve(evaluated_items.size() *
                                          ec_point_length);
    for (const auto& item : evaluated_items) {
      evaluated_batch.flatten_bytes.append(item);
    }

    options_.link1->SendAsyncThrottled(options_.link1->NextRank(),
                                       evaluated_batch.Serialize(), tag_send);

    batch_count++;
  }
  SPDLOG_INFO("{} finished, batch_count={}", __func__, batch_count);
}

void EcdhOprfPsiServer::RecvBlindAndShuffleSendEvaluate() {
  size_t batch_count = 0;

  size_t ec_point_length = oprf_server_->GetEcPointLength();

  std::vector<std::string> evaluated_items;

  while (true) {
    const auto tag = fmt::format("EcdhOprfPSI:BlindItems:{}", batch_count);
    PsiDataBatch blinded_batch = PsiDataBatch::Deserialize(
        options_.link1->Recv(options_.link1->NextRank(), tag));

    if (blinded_batch.is_last_batch) {
      break;
    }

    // Fetch blinded y^r.
    SPU_ENFORCE(blinded_batch.flatten_bytes.size() % ec_point_length == 0);
    size_t num_items = blinded_batch.flatten_bytes.size() / ec_point_length;

    std::vector<std::string> blinded_items(num_items);
    for (size_t idx = 0; idx < num_items; ++idx) {
      blinded_items[idx] = blinded_batch.flatten_bytes.substr(
          idx * ec_point_length, ec_point_length);
    }
    // (x^r)^s
    std::vector<std::string> batch_evaluated_items =
        oprf_server_->Evaluate(blinded_items);

    evaluated_items.insert(evaluated_items.end(), batch_evaluated_items.begin(),
                           batch_evaluated_items.end());

    batch_count++;
  }
  SPDLOG_INFO("recv Blind finished, batch_count={}", batch_count);

  std::sort(evaluated_items.begin(), evaluated_items.end());

  std::unique_ptr<IBatchProvider> provider =
      std::make_unique<MemoryBatchProvider>(evaluated_items);

  batch_count = 0;
  while (true) {
    std::vector<std::string> batch_evaluated_items =
        provider->ReadNextBatch(options_.batch_size);

    PsiDataBatch evaluated_batch;
    evaluated_batch.is_last_batch = batch_evaluated_items.empty();

    const auto tag_send =
        fmt::format("EcdhOprfPSI:EvaluatedItems:{}", batch_count);

    if (evaluated_batch.is_last_batch) {
      SPDLOG_INFO("{} Last batch triggered, batch_count={}", __func__,
                  batch_count);

      options_.link1->SendAsyncThrottled(options_.link1->NextRank(),
                                         evaluated_batch.Serialize(), tag_send);
      break;
    }

    evaluated_batch.flatten_bytes.reserve(batch_evaluated_items.size() *
                                          ec_point_length);
    for (const auto& item : batch_evaluated_items) {
      evaluated_batch.flatten_bytes.append(item);
    }

    options_.link1->SendAsyncThrottled(options_.link1->NextRank(),
                                       evaluated_batch.Serialize(), tag_send);

    batch_count++;
  }
  SPDLOG_INFO("send evaluated finished, batch_count={}", batch_count);
}

std::pair<std::vector<uint64_t>, size_t>
EcdhOprfPsiServer::RecvIntersectionMaskedItems(
    const std::shared_ptr<IShuffleBatchProvider>& cache_provider,
    size_t batch_size) {
  std::unordered_set<std::string> client_masked_items;

  size_t compare_length = oprf_server_->GetCompareLength();
  size_t batch_count = 0;

  while (true) {
    const auto tag = fmt::format("EcdhOprfPSI:batch_count:{}", batch_count);
    PsiDataBatch masked_batch = PsiDataBatch::Deserialize(
        options_.link1->Recv(options_.link1->NextRank(), tag));

    if (masked_batch.is_last_batch) {
      break;
    }

    // Fetch blinded y^r.
    SPU_ENFORCE(masked_batch.flatten_bytes.size() % compare_length == 0);
    size_t num_items = masked_batch.flatten_bytes.size() / compare_length;

    std::vector<std::string> batch_masked_items(num_items);
    for (size_t idx = 0; idx < num_items; ++idx) {
      batch_masked_items[idx] = masked_batch.flatten_bytes.substr(
          idx * compare_length, compare_length);
    }

    client_masked_items.insert(batch_masked_items.begin(),
                               batch_masked_items.end());

    batch_count++;
  }
  SPDLOG_INFO("Recv intersection masked finished, batch_count={}", batch_count);

  std::vector<uint64_t> indices;

  size_t item_index = 0;
  batch_count = 0;
  size_t compare_thread_num = omp_get_num_procs();

  while (true) {
    std::vector<std::string> server_masked_items;
    std::vector<size_t> batch_indices;
    std::vector<size_t> batch_shuffled_indices;

    std::tie(server_masked_items, batch_indices, batch_shuffled_indices) =
        cache_provider->ReadNextBatchWithIndex(batch_size);
    if (server_masked_items.empty()) {
      break;
    }
    SPU_ENFORCE(server_masked_items.size() == batch_shuffled_indices.size());

    size_t compare_size =
        (server_masked_items.size() + compare_thread_num - 1) /
        compare_thread_num;

    std::vector<std::vector<uint64_t>> batch_result(compare_thread_num);

    auto compare_proc = [&](int idx) -> void {
      uint64_t begin = idx * compare_size;
      uint64_t end =
          std::min<uint64_t>(server_masked_items.size(), begin + compare_size);

      for (uint64_t i = begin; i < end; ++i) {
        if (client_masked_items.find(server_masked_items[i]) !=
            client_masked_items.end()) {
          batch_result[idx].push_back(batch_shuffled_indices[i]);
        }
      }
    };

    std::vector<std::future<void>> f_compare(compare_thread_num);
    for (size_t i = 0; i < compare_thread_num; i++) {
      f_compare[i] = std::async(compare_proc, i);
    }

    for (size_t i = 0; i < compare_thread_num; i++) {
      f_compare[i].get();
    }

    for (const auto& r : batch_result) {
      indices.insert(indices.end(), r.begin(), r.end());
    }

    batch_count++;

    item_index += server_masked_items.size();
    SPDLOG_INFO("GetIndices batch count:{}, item_index:{}", batch_count,
                item_index);
  }

  return std::make_pair(indices, item_index);
}

void EcdhOprfPsiClient::RecvFinalEvaluatedItems(
    const std::shared_ptr<ICipherStore>& cipher_store) {
  SPDLOG_INFO("Begin Recv FinalEvaluatedItems items");

  size_t batch_count = 0;
  while (true) {
    const auto tag =
        fmt::format("EcdhOprfPSI:FinalEvaluatedItems:{}", batch_count);

    // Fetch y^b.
    PsiDataBatch masked_batch = PsiDataBatch::Deserialize(
        options_.link0->Recv(options_.link0->NextRank(), tag));

    if (masked_batch.is_last_batch) {
      SPDLOG_INFO("{} Last batch triggered, batch_count={}", __func__,
                  batch_count);
      break;
    }

    SPU_ENFORCE(masked_batch.flatten_bytes.length() % compare_length_ == 0);
    size_t num_items = masked_batch.flatten_bytes.length() / compare_length_;

    std::vector<std::string> evaluated_items(num_items);
    for (size_t idx = 0; idx < num_items; ++idx) {
      evaluated_items[idx] =
          absl::Base64Escape(masked_batch.flatten_bytes.substr(
              idx * compare_length_, compare_length_));
    }
    cipher_store->SavePeer(evaluated_items);

    batch_count++;
  }
  SPDLOG_INFO("End Recv FinalEvaluatedItems items");
}

size_t EcdhOprfPsiClient::SendBlindedItems(
    const std::shared_ptr<IBatchProvider>& batch_provider) {
  size_t batch_count = 0;
  size_t items_count = 0;

  SPDLOG_INFO("Begin Send BlindedItems items");

  while (true) {
    auto items = batch_provider->ReadNextBatch(options_.batch_size);

    PsiDataBatch blinded_batch;
    blinded_batch.is_last_batch = items.empty();

    const auto tag = fmt::format("EcdhOprfPSI:BlindItems:{}", batch_count);

    if (blinded_batch.is_last_batch) {
      SPDLOG_INFO("{} Last batch triggered, batch_count={}", __func__,
                  batch_count);

      options_.link1->SendAsyncThrottled(options_.link1->NextRank(),
                                         blinded_batch.Serialize(), tag);
      break;
    }

    std::vector<std::shared_ptr<IEcdhOprfClient>> oprf_clients(items.size());
    std::vector<std::string> blinded_items(items.size());

    yacl::parallel_for(0, items.size(), 1, [&](int64_t begin, int64_t end) {
      for (int64_t idx = begin; idx < end; ++idx) {
        if (oprf_client_ == nullptr) {
          std::shared_ptr<IEcdhOprfClient> oprf_client_ptr =
              CreateEcdhOprfClient(options_.oprf_type, options_.curve_type);

          oprf_clients[idx] = oprf_client_ptr;
        } else {
          oprf_clients[idx] = oprf_client_;
        }

        blinded_items[idx] = oprf_clients[idx]->Blind(items[idx]);
      }
    });

    blinded_batch.flatten_bytes.reserve(items.size() * ec_point_length_);

    for (uint64_t idx = 0; idx < items.size(); ++idx) {
      blinded_batch.flatten_bytes.append(blinded_items[idx]);
    }

    // push to oprf_client_queue_
    if (oprf_client_ == nullptr) {
      std::unique_lock<std::mutex> lock(mutex_);
      queue_push_cv_.wait(lock, [&] {
        return (oprf_client_queue_.size() < options_.window_size);
      });
      oprf_client_queue_.push(std::move(oprf_clients));
      queue_pop_cv_.notify_one();
      // SPDLOG_INFO("push to queue size:{}", oprf_client_queue_.size());
    }

    options_.link1->SendAsyncThrottled(options_.link1->NextRank(),
                                       blinded_batch.Serialize(), tag);

    items_count += items.size();
    batch_count++;
  }
  SPDLOG_INFO("{} finished, batch_count={} items_count={}", __func__,
              batch_count, items_count);

  return items_count;
}

void EcdhOprfPsiClient::RecvEvaluatedItems(
    const std::shared_ptr<ICipherStore>& cipher_store) {
  SPDLOG_INFO("Begin Recv EvaluatedItems items");

  size_t batch_count = 0;

  while (true) {
    const auto tag = fmt::format("EcdhOprfPSI:EvaluatedItems:{}", batch_count);
    PsiDataBatch masked_batch = PsiDataBatch::Deserialize(
        options_.link1->Recv(options_.link1->NextRank(), tag));
    // Fetch evaluate y^rs.

    if (masked_batch.is_last_batch) {
      SPDLOG_INFO("{} Last batch triggered, batch_count={}", __func__,
                  batch_count);
      break;
    }

    SPU_ENFORCE(masked_batch.flatten_bytes.size() % ec_point_length_ == 0);
    size_t num_items = masked_batch.flatten_bytes.size() / ec_point_length_;

    std::vector<std::string> evaluate_items(num_items);
    for (size_t idx = 0; idx < num_items; ++idx) {
      evaluate_items[idx] = masked_batch.flatten_bytes.substr(
          idx * ec_point_length_, ec_point_length_);
    }

    std::vector<std::string> oprf_items(num_items);
    std::vector<std::shared_ptr<IEcdhOprfClient>> oprf_clients;

    // get oprf_clients
    if (oprf_client_ == nullptr) {
      std::unique_lock<std::mutex> lock(mutex_);
      queue_pop_cv_.wait(lock, [&] { return (!oprf_client_queue_.empty()); });

      oprf_clients = std::move(oprf_client_queue_.front());
      oprf_client_queue_.pop();
      queue_push_cv_.notify_one();
    } else {
      oprf_clients.resize(num_items);
      for (size_t i = 0; i < oprf_clients.size(); ++i) {
        oprf_clients[i] = oprf_client_;
      }
    }

    SPU_ENFORCE(oprf_clients.size() == num_items,
                "EcdhOprfServer should not be nullptr");

    yacl::parallel_for(0, num_items, 1, [&](int64_t begin, int64_t end) {
      for (int64_t idx = begin; idx < end; ++idx) {
        oprf_items[idx] = absl::Base64Escape(
            oprf_clients[idx]->Finalize(evaluate_items[idx]));
      }
    });

    cipher_store->SaveSelf(oprf_items);

    batch_count++;
  }
  SPDLOG_INFO("End Recv EvaluatedItems");
}

void EcdhOprfPsiClient::SendIntersectionMaskedItems(
    const std::shared_ptr<IBatchProvider>& batch_provider) {
  size_t batch_count = 0;
  size_t items_count = 0;

  SPDLOG_INFO("Begin Send IntersectionMaskedItems items");

  while (true) {
    auto items = batch_provider->ReadNextBatch(options_.batch_size);

    PsiDataBatch blinded_batch;
    blinded_batch.is_last_batch = items.empty();

    const auto tag = fmt::format("EcdhOprfPSI:batch_count:{}", batch_count);

    if (blinded_batch.is_last_batch) {
      SPDLOG_INFO("{} Last batch triggered, batch_count={}", __func__,
                  batch_count);

      options_.link1->SendAsyncThrottled(options_.link1->NextRank(),
                                         blinded_batch.Serialize(), tag);
      break;
    }

    blinded_batch.flatten_bytes.reserve(items.size() * compare_length_);

    for (uint64_t idx = 0; idx < items.size(); ++idx) {
      std::string b64_dest;
      absl::Base64Unescape(items[idx], &b64_dest);

      blinded_batch.flatten_bytes.append(b64_dest);
    }

    options_.link1->SendAsyncThrottled(options_.link1->NextRank(),
                                       blinded_batch.Serialize(), tag);

    items_count += items.size();
    batch_count++;
  }
  SPDLOG_INFO("{} finished, batch_count={} items_count={}", __func__,
              batch_count, items_count);

  return;
}

}  // namespace spu::psi
