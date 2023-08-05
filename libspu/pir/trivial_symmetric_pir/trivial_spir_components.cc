#include "libspu/pir/trivial_symmetric_pir/trivial_spir_components.h"

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <future>
#include <unordered_set>
#include <utility>

#include "absl/strings/escaping.h"
#include "yacl/crypto/tools/prg.h"
#include "yacl/utils/parallel.h"
#include "yacl/utils/serialize.h"

#include "libspu/core/prelude.h"
#include "libspu/psi/core/communication.h"
#include "libspu/psi/cryptor/ecc_utils.h"
#include "libspu/psi/utils/ub_psi_cache.h"

namespace spu::pir {
namespace {
std::string OneTimePad(std::string& key, std::string& item) {
  // extract the first 16 bytes of the key as the PRG seed
  uint128_t seed = 0;
  for (int i = 0; i < 16; i++) {
    seed = seed | (((uint128_t)key[i]) << (120 - 8 * i));
  }
  yacl::crypto::Prg<uint8_t> prg;
  prg.SetSeed(seed);
  std::vector<uint8_t> c(item.size());
  // std::cout << "label plaintext length is: " << item.size() << std::endl;
  for (size_t i = 0; i < item.size(); i++) {
    c[i] = item[i] ^ prg();
  }
  std::string s(c.begin(), c.end());
  return s;
}

std::string EncryptLabel(std::string& oprf_item, std::string& label) {
  return OneTimePad(oprf_item, label);
}

std::string DecryptLabel(std::string& oprf_item, std::string& label_cipher) {
  return OneTimePad(oprf_item, label_cipher);
}
}  // namespace

size_t LabeledEcdhOprfPsiServer::FullEvaluateAndSend(
    const std::shared_ptr<spu::psi::IBatchProvider>& batch_provider) {
  size_t items_count = 0;
  size_t batch_count = 1;
  size_t id_compare_length = oprf_server_id_->GetCompareLength();
  size_t label_cipher_length = label_length_;

  bool stop_flag = false;
  omp_lock_t lck_read, lck_send;

  omp_init_lock(&lck_read);
  omp_init_lock(&lck_send);

  int tid;
  batch_count = 0;
  std::vector<std::string> batch_ids;
  std::vector<std::string> batch_labels;
  spu::psi::PsiDataBatch batch;
  size_t i;
  size_t local_batch_count;
  int nthreads = omp_get_num_threads();
  int mcpus = omp_get_num_procs();
  SPDLOG_INFO("omp_get_num_threads:{} cpus:{}", nthreads, mcpus);
  omp_set_num_threads(mcpus);

#pragma omp parallel private(tid, nthreads, i, batch_ids, batch_labels, batch, \
                             local_batch_count)                                \
    shared(lck_read, lck_send, batch_count, items_count, id_compare_length,    \
           label_cipher_length, stop_flag)
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
      std::tie(batch_ids, batch_labels) =
          batch_provider->ReadNextBatchWithLabel(options_.batch_size);

      batch.is_last_batch = batch_ids.empty();

      if (batch_ids.empty()) {
        stop_flag = true;
      } else {
        items_count += batch_ids.size();
        batch_count++;
        if ((batch_count % 1000) == 0) {
          SPDLOG_INFO("batch_count:{}", batch_count);
        }
        local_batch_count = batch_count;
      }

      omp_unset_lock(&lck_read);

      if (batch_ids.size() == 0) {
        break;
      }

      batch.flatten_bytes.reserve(batch_ids.size() *
                                  (id_compare_length + label_cipher_length));

      // std::string
      batch.flatten_bytes = oprf_server_id_->SimpleEvaluate(batch_ids[0]);
      std::string oprf_label = oprf_server_label_->SimpleEvaluate(batch_ids[0]);
      std::string label_cipher = EncryptLabel(oprf_label, batch_labels[0]);
      batch.flatten_bytes.append(label_cipher);
      for (i = 1; i < batch_ids.size(); i++) {
        std::string masked_id = oprf_server_id_->SimpleEvaluate(batch_ids[i]);
        batch.flatten_bytes.append(masked_id);

        std::string oprf_label =
            oprf_server_label_->SimpleEvaluate(batch_ids[i]);
        std::string label_cipher = EncryptLabel(oprf_label, batch_labels[i]);
        batch.flatten_bytes.append(label_cipher);
      }

      omp_set_lock(&lck_send);

      // Send x^a and the label cipher encrypted with x^b
      options_.link0->SendAsync(
          options_.link0->NextRank(), batch.Serialize(),
          fmt::format("LabeledEcdhOprfPSI:FinalEvaluatedItems:{}",
                      local_batch_count));

      omp_unset_lock(&lck_send);
    }
  }

  batch.is_last_batch = true;
  batch.flatten_bytes.resize(0);
  options_.link0->SendAsync(
      options_.link0->NextRank(), batch.Serialize(),
      fmt::format("LabeledEcdhOprfPSI last batch,FinalEvaluatedItems:{}",
                  batch_count));

  SPDLOG_INFO("{} finished, batch_count={} items_count={}", __func__,
              batch_count, items_count);

  return items_count;
}

void LabeledEcdhOprfPsiClient::RecvFinalEvaluatedItems(
    std::vector<std::string>* server_ids,
    std::vector<std::string>* server_labels) {
  SPDLOG_INFO("Begin Recv FinalEvaluatedItems items");

  size_t batch_count = 0;
  while (true) {
    const auto tag =
        fmt::format("LabeledEcdhOprfPSI:FinalEvaluatedItems:{}", batch_count);

    // Fetch y^b.
    spu::psi::PsiDataBatch masked_batch = spu::psi::PsiDataBatch::Deserialize(
        options_.link0->Recv(options_.link0->NextRank(), tag));

    if (masked_batch.is_last_batch) {
      SPDLOG_INFO("{} Last batch triggered, batch_count={}", __func__,
                  batch_count);
      break;
    }

    size_t label_cipher_length_ = label_length_;
    size_t item_length_ = compare_length_ + label_cipher_length_;
    SPU_ENFORCE(masked_batch.flatten_bytes.length() % item_length_ == 0);
    size_t num_items = masked_batch.flatten_bytes.length() / item_length_;

    std::vector<std::string> evaluated_items(num_items);
    (*server_ids).reserve(num_items);
    (*server_labels).reserve(num_items);

    for (size_t idx = 0; idx < num_items; ++idx) {
      (*server_ids)
          .emplace_back(masked_batch.flatten_bytes.substr(idx * item_length_,
                                                          compare_length_));
      (*server_labels)
          .emplace_back(masked_batch.flatten_bytes.substr(
              idx * item_length_ + compare_length_, label_cipher_length_));
    }

    batch_count++;
  }
  SPDLOG_INFO("End Recv FinalEvaluatedItems items");
}

size_t LabeledEcdhOprfPsiClient::SendBlindedItems(
    const std::unique_ptr<spu::psi::CsvBatchProvider>& batch_provider,
    std::vector<std::string>* client_ids) {
  size_t batch_count = 0;
  size_t items_count = 0;

  SPDLOG_INFO("Begin Send BlindedItems items");

  while (true) {
    auto items = batch_provider->ReadNextBatch(options_.batch_size);

    spu::psi::PsiDataBatch blinded_batch;
    blinded_batch.is_last_batch = items.empty();

    const auto tag =
        fmt::format("LabeledEcdhOprfPSI:BlindItems:{}", batch_count);

    if (blinded_batch.is_last_batch) {
      SPDLOG_INFO("{} Last batch triggered, batch_count={}", __func__,
                  batch_count);

      options_.link1->SendAsync(options_.link1->NextRank(),
                                blinded_batch.Serialize(), tag);
      break;
    }

    for (size_t i = 0; i < items.size(); ++i) {
      (*client_ids).emplace_back(items[i]);
    }

    std::vector<std::shared_ptr<spu::psi::IEcdhOprfClient>> oprf_clients(
        items.size());
    std::vector<std::string> blinded_items(items.size());

    // yacl::parallel_for(0, items.size(), 1, [&](int64_t begin, int64_t end) {
    //   for (int64_t idx = begin; idx < end; ++idx) {
    //     if (oprf_client_ == nullptr) {
    //       std::shared_ptr<spu::psi::IEcdhOprfClient> oprf_client_ptr =
    //           spu::psi::CreateEcdhOprfClient(options_.oprf_type,
    //           options_.curve_type);

    //       oprf_clients[idx] = oprf_client_ptr;
    //     } else {
    //       oprf_clients[idx] = oprf_client_;
    //     }
    //     blinded_items[idx] = oprf_clients[idx]->Blind(items[idx]);
    //   }
    // });

    // if the client has a large number of queries, parallel_for should be used
    for (size_t idx = 0; idx < items.size(); ++idx) {
      if (oprf_client_ == nullptr) {
        std::shared_ptr<spu::psi::IEcdhOprfClient> oprf_client_ptr =
            spu::psi::CreateEcdhOprfClient(options_.oprf_type,
                                           options_.curve_type);

        oprf_clients[idx] = oprf_client_ptr;
      } else {
        oprf_clients[idx] = oprf_client_;
      }
      blinded_items[idx] = oprf_clients[idx]->Blind(items[idx]);
    }

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

    options_.link1->SendAsync(options_.link1->NextRank(),
                              blinded_batch.Serialize(), tag);

    items_count += blinded_items.size();
    batch_count++;
  }
  SPDLOG_INFO("{} finished, batch_count={} items_count={}", __func__,
              batch_count, items_count);

  return items_count;
}

void LabeledEcdhOprfPsiServer::RecvBlindAndSendEvaluate() {
  size_t batch_count = 0;

  size_t ec_point_length = oprf_server_id_->GetEcPointLength();

  while (true) {
    const auto tag =
        fmt::format("LabeledEcdhOprfPSI:BlindItems:{}", batch_count);
    spu::psi::PsiDataBatch blinded_batch = spu::psi::PsiDataBatch::Deserialize(
        options_.link1->Recv(options_.link1->NextRank(), tag));

    spu::psi::PsiDataBatch evaluated_batch;
    evaluated_batch.is_last_batch = blinded_batch.is_last_batch;

    const auto tag_send =
        fmt::format("LabeledEcdhOprfPSI:EvaluatedItems:{}", batch_count);

    if (blinded_batch.is_last_batch) {
      SPDLOG_INFO("{} Last batch triggered, batch_count={}", __func__,
                  batch_count);

      options_.link1->SendAsync(options_.link1->NextRank(),
                                evaluated_batch.Serialize(), tag_send);
      break;
    }

    // Fetch blinded x^r.
    SPU_ENFORCE(blinded_batch.flatten_bytes.size() % ec_point_length == 0);
    size_t num_items = blinded_batch.flatten_bytes.size() / ec_point_length;

    std::vector<std::string> blinded_items(num_items);
    for (size_t idx = 0; idx < num_items; ++idx) {
      blinded_items[idx] = blinded_batch.flatten_bytes.substr(
          idx * ec_point_length, ec_point_length);
    }

    evaluated_batch.flatten_bytes.reserve(2 * num_items * ec_point_length);

    // (x^r)^a, for comparing id
    std::vector<std::string> evaluated_items_id =
        oprf_server_id_->Evaluate(blinded_items);
    // (x^r)^b, for calculating the key to decrypt the label
    std::vector<std::string> evaluated_items_label =
        oprf_server_label_->Evaluate(blinded_items);
    for (size_t idx = 0; idx < num_items; ++idx) {
      evaluated_batch.flatten_bytes.append(evaluated_items_id[idx]);
      evaluated_batch.flatten_bytes.append(evaluated_items_label[idx]);
    }

    options_.link1->SendAsync(options_.link1->NextRank(),
                              evaluated_batch.Serialize(), tag_send);

    batch_count++;
  }
  SPDLOG_INFO("{} finished, batch_count={}", __func__, batch_count);
}

void LabeledEcdhOprfPsiClient::RecvEvaluatedItems(
    std::vector<std::string>* client_evaluated_ids,
    std::vector<std::string>* client_label_keys) {
  SPDLOG_INFO("Begin Recv EvaluatedItems items");

  size_t batch_count = 0;

  while (true) {
    const auto tag =
        fmt::format("LabeledEcdhOprfPSI:EvaluatedItems:{}", batch_count);
    spu::psi::PsiDataBatch masked_batch = spu::psi::PsiDataBatch::Deserialize(
        options_.link1->Recv(options_.link1->NextRank(), tag));
    // Fetch evaluate y^rs.

    if (masked_batch.is_last_batch) {
      SPDLOG_INFO("{} Last batch triggered, batch_count={}", __func__,
                  batch_count);
      break;
    }

    SPU_ENFORCE(masked_batch.flatten_bytes.size() % (2 * ec_point_length_) ==
                0);
    size_t num_items =
        masked_batch.flatten_bytes.size() / (2 * ec_point_length_);

    std::vector<std::string> blinded_evaluated_items_id(num_items);
    std::vector<std::string> blinded_evaluated_items_label(num_items);
    for (size_t idx = 0; idx < num_items; ++idx) {
      blinded_evaluated_items_id[idx] = masked_batch.flatten_bytes.substr(
          idx * (2 * ec_point_length_), ec_point_length_);
      blinded_evaluated_items_label[idx] = masked_batch.flatten_bytes.substr(
          idx * (2 * ec_point_length_) + ec_point_length_, ec_point_length_);
    }

    std::vector<std::shared_ptr<spu::psi::IEcdhOprfClient>> oprf_clients;

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
                "EcdhOprfClient should not be nullptr");

    // yacl::parallel_for(0, num_items, 1, [&](int64_t begin, int64_t end) {
    //   for (int64_t idx = begin; idx < end; ++idx) {
    //     (*client_evaluated_ids).emplace_back(oprf_clients[idx]->Finalize(blinded_evaluated_items_id[idx]));
    //     (*client_label_keys).emplace_back(oprf_clients[idx]->Finalize(blinded_evaluated_items_label[idx]));
    //   }
    // });

    for (size_t idx = 0; idx < num_items; ++idx) {
      (*client_evaluated_ids)
          .emplace_back(
              oprf_clients[idx]->Finalize(blinded_evaluated_items_id[idx]));
      (*client_label_keys)
          .emplace_back(
              oprf_clients[idx]->Finalize(blinded_evaluated_items_label[idx]));
    }

    batch_count++;
  }
  SPDLOG_INFO("End Recv EvaluatedItems");
}

std::pair<std::vector<uint64_t>, std::vector<std::string>>
LabeledEcdhOprfPsiClient::FinalizeAndDecryptLabels(
    const std::shared_ptr<spu::psi::MemoryBatchProvider>& server_batch_provider,
    const std::vector<std::string>& client_evaluated_ids,
    const std::vector<std::string>& client_label_keys) {
  std::vector<uint64_t> indices;
  std::vector<std::string> labels;
  size_t batch_count = 0;
  size_t compare_thread_num = omp_get_num_procs();

  while (true) {
    std::vector<std::string> batch_server_ids;
    std::vector<std::string> batch_server_labels;

    std::tie(batch_server_ids, batch_server_labels) =
        server_batch_provider->ReadNextBatchWithLabel(options_.batch_size);
    if (batch_server_ids.empty()) {
      SPDLOG_INFO("{} Last batch triggered, batch_count={}", __func__,
                  batch_count);
      break;
    }
    SPU_ENFORCE(batch_server_ids.size() == batch_server_labels.size());

    size_t compare_size =
        (batch_server_ids.size() + compare_thread_num - 1) / compare_thread_num;

    std::vector<std::vector<uint64_t>> batch_result_indice(compare_thread_num);
    std::vector<std::vector<std::string>> batch_result_label(
        compare_thread_num);

    auto compare_proc = [&](int idx) -> void {
      uint64_t begin = idx * compare_size;
      uint64_t end =
          std::min<uint64_t>(batch_server_ids.size(), begin + compare_size);
      for (uint64_t i = begin; i < end; ++i) {
        auto it = std::find(client_evaluated_ids.begin(),
                            client_evaluated_ids.end(), batch_server_ids[i]);
        if (it != client_evaluated_ids.end()) {
          uint64_t index = it - client_evaluated_ids.begin();
          // begin to decrypt the label
          std::string label_key = client_label_keys[index];
          std::string label = DecryptLabel(label_key, batch_server_labels[i]);
          // the index also corresponds to the indice of the client's input csv
          batch_result_indice[idx].emplace_back(index);
          batch_result_label[idx].emplace_back(label);
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

    for (size_t i = 0; i < compare_thread_num; i++) {
      auto indice = batch_result_indice[i];
      indices.insert(indices.end(), indice.begin(), indice.end());
      auto label = batch_result_label[i];
      labels.insert(labels.end(), label.begin(), label.end());
    }

    batch_count++;

    // SPDLOG_INFO("GetIndices batch count:{}, item_index:{}", batch_count,
    //             item_index);
  }
  return std::make_pair(indices, labels);
}
}  // namespace spu::pir