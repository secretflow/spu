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

#include <future>
#include <utility>

#include "yacl/utils/parallel.h"
#include "yacl/utils/serialize.h"

#include "libspu/psi/core/communication.h"
#include "libspu/psi/cryptor/ecc_utils.h"

namespace spu::psi {

size_t EcdhOprfPsiServer::FullEvaluate(
    const std::shared_ptr<IBatchProvider>& batch_provider,
    const std::shared_ptr<ICipherStore>& cipher_store) {
  size_t items_count = 0;
  size_t batch_count = 0;
  while (true) {
    auto items = batch_provider->ReadNextBatch(options_.batch_size);
    if (items.empty()) {
      break;
    }
    auto masked_items = oprf_server_->FullEvaluate(items);
    for (const auto& masked_item : masked_items) {
      cipher_store->SaveSelf(masked_item);
    }
    items_count += items.size();
    batch_count++;
  }
  SPDLOG_INFO("{} finished, batch_count={} items_count={}", __func__,
              batch_count, items_count);
  return items_count;
}

void EcdhOprfPsiServer::SendFinalEvaluatedItems(
    const std::shared_ptr<IBatchProvider>& batch_provider) {
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
    options_.link0->SendAsync(options_.link0->NextRank(), batch.Serialize(),
                              tag);

    if (batch.is_last_batch) {
      SPDLOG_INFO("{} Last batch triggered, batch_count={}", __func__,
                  batch_count);
      break;
    }

    batch_count++;
  }

  SPDLOG_INFO("{} finished, batch_count={}", __func__, batch_count);
}

size_t EcdhOprfPsiServer::FullEvaluateAndSend(
    const std::shared_ptr<IBatchProvider>& batch_provider) {
  size_t items_count = 0;
  size_t batch_count = 1;
  size_t compare_length = oprf_server_->GetCompareLength();
  std::vector<std::string> batch_items_next =
      batch_provider->ReadNextBatch(options_.batch_size);

  SPDLOG_INFO("Begin EvaluateAndSend items");

  while (true) {
    PsiDataBatch batch;
    auto items = batch_items_next;
    batch.is_last_batch = items.empty();

    if (items.empty()) {
      const auto tag = fmt::format(
          "EcdhOprfPSI last batch,FinalEvaluatedItems:{}", batch_count);
      options_.link0->SendAsync(options_.link0->NextRank(), batch.Serialize(),
                                tag);
      break;
    }
    std::future<void> f_prefetch = std::async([&] {
      batch_items_next = batch_provider->ReadNextBatch(options_.batch_size);
    });

    auto masked_items = oprf_server_->FullEvaluate(items);

    batch.flatten_bytes.reserve(items.size() * compare_length);
    for (const auto& masked_item : masked_items) {
      batch.flatten_bytes.append(masked_item);
    }
    // Send x^a.
    const auto tag =
        fmt::format("EcdhOprfPSI:FinalEvaluatedItems:{}", batch_count);
    options_.link0->SendAsync(options_.link0->NextRank(), batch.Serialize(),
                              tag);
    batch_count++;
    items_count += items.size();
    f_prefetch.get();
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

      options_.link1->SendAsync(options_.link1->NextRank(),
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

    options_.link1->SendAsync(options_.link1->NextRank(),
                              evaluated_batch.Serialize(), tag_send);

    batch_count++;
  }
  SPDLOG_INFO("{} finished, batch_count={}", __func__, batch_count);
}

void EcdhOprfPsiClient::RecvFinalEvaluatedItems(
    const std::shared_ptr<ICipherStore>& cipher_store) {
  SPDLOG_INFO("Begin Recv FinalEvaluatedItems items");

  size_t batch_count = 0;
  while (true) {
    const auto tag =
        fmt::format("EcdhOprfPSI:FinalEvaluatedItems:{}", batch_count);
    PsiDataBatch masked_batch = PsiDataBatch::Deserialize(
        options_.link0->Recv(options_.link0->NextRank(), tag));
    // Fetch y^b.

    if (masked_batch.is_last_batch) {
      SPDLOG_INFO("{} Last batch triggered, batch_count={}", __func__,
                  batch_count);
      break;
    }

    SPU_ENFORCE(masked_batch.flatten_bytes.size() % compare_length_ == 0);
    size_t num_items = masked_batch.flatten_bytes.size() / compare_length_;

    std::vector<std::string> evaluated_items(num_items);
    for (size_t idx = 0; idx < num_items; ++idx) {
      evaluated_items[idx] = masked_batch.flatten_bytes.substr(
          idx * compare_length_, compare_length_);
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

      options_.link1->SendAsync(options_.link1->NextRank(),
                                blinded_batch.Serialize(), tag);
      break;
    }

    std::vector<std::shared_ptr<IEcdhOprfClient>> oprf_clients(items.size());
    std::vector<std::string> blinded_items(items.size());

    yacl::parallel_for(0, items.size(), 1, [&](int64_t begin, int64_t end) {
      for (int64_t idx = begin; idx < end; ++idx) {
        std::shared_ptr<IEcdhOprfClient> oprf_client =
            CreateEcdhOprfClient(options_.oprf_type, options_.curve_type);

        oprf_clients[idx] = oprf_client;

        blinded_items[idx] = oprf_client->Blind(items[idx]);
      }
    });

    blinded_batch.flatten_bytes.reserve(items.size() * ec_point_length_);

    for (uint64_t idx = 0; idx < items.size(); ++idx) {
      blinded_batch.flatten_bytes.append(blinded_items[idx]);
    }

    // push to oprf_client_queue_
    {
      std::unique_lock<std::mutex> lock(mutex_);
      queue_push_cv_.wait(lock, [&] {
        return (oprf_client_queue_.size() < options_.window_size);
      });
      oprf_client_queue_.push(std::move(oprf_clients));
      queue_pop_cv_.notify_one();
      // SPDLOG_INFO("push to queue size:{}", oprf_client_queue_.size());
    }

    options_.link1->SendAsync(options_.link1->NextRank(),
                              blinded_batch.Serialize(), tag);

    items_count += items.size();
    batch_count++;
  }
  SPDLOG_INFO("{} finished, batch_count={} items_count={}", __func__,
              batch_count, items_count);

  return items_count;
}

void EcdhOprfPsiClient::RecvEvaluatedItems(
    const std::shared_ptr<IBatchProvider>& batch_provider,
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
    auto items = batch_provider->ReadNextBatch(options_.batch_size);

    SPU_ENFORCE(masked_batch.flatten_bytes.size() % ec_point_length_ == 0);
    size_t num_items = masked_batch.flatten_bytes.size() / ec_point_length_;

    SPU_ENFORCE(items.size() % num_items == 0);

    std::vector<std::string> evaluate_items(num_items);
    for (size_t idx = 0; idx < num_items; ++idx) {
      evaluate_items[idx] = masked_batch.flatten_bytes.substr(
          idx * ec_point_length_, ec_point_length_);
    }

    std::vector<std::string> oprf_items(items.size());
    std::vector<std::shared_ptr<IEcdhOprfClient>> oprf_clients;

    // get oprf_clients
    {
      std::unique_lock<std::mutex> lock(mutex_);
      queue_pop_cv_.wait(lock, [&] {
        // SPDLOG_INFO("pop queue size:{}", oprf_client_queue_.size());
        return (!oprf_client_queue_.empty());
      });

      oprf_clients = std::move(oprf_client_queue_.front());
      oprf_client_queue_.pop();
      queue_push_cv_.notify_one();
    }

    SPU_ENFORCE(oprf_clients.size() == items.size(),
                "EcdhOprfServer should not be nullptr");

    yacl::parallel_for(0, items.size(), 1, [&](int64_t begin, int64_t end) {
      for (int64_t idx = begin; idx < end; ++idx) {
        oprf_items[idx] =
            oprf_clients[idx]->Finalize(items[idx], evaluate_items[idx]);
      }
    });

    cipher_store->SaveSelf(oprf_items);

    batch_count++;
  }
  SPDLOG_INFO("End Recv EvaluatedItems");
}

}  // namespace spu::psi
