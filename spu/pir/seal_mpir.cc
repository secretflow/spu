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

#include "spu/pir/seal_mpir.h"

#include <algorithm>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "spdlog/spdlog.h"

#include "spu/pir/serializable.pb.h"

namespace spu::pir {

void MultiQueryServer::GenerateSimpleHash() {
  std::vector<uint128_t> query_index_hash(
      query_options_.seal_options.element_number);

  // generate item hash, server and client use same seed
  yasl::parallel_for(0, query_options_.seal_options.element_number, 1,
                     [&](int64_t begin, int64_t end) {
                       for (int idx = begin; idx < end; ++idx) {
                         query_index_hash[idx] = HashItemIndex(idx);
                       }
                     });

  size_t num_bins = cuckoo_params_.NumBins();

  SPDLOG_INFO("element_number:{}", query_options_.seal_options.element_number);

  for (size_t idx = 0; idx < query_options_.seal_options.element_number;
       ++idx) {
    spu::psi::CuckooIndex::HashRoom itemHash(query_index_hash[idx]);

    std::vector<uint64_t> bin_idx(query_options_.cuckoo_hash_number);
    for (size_t j = 0; j < query_options_.cuckoo_hash_number; ++j) {
      bin_idx[j] = itemHash.GetHash(j) % num_bins;
      size_t k = 0;
      for (; k < j; ++k) {
        if (bin_idx[j] == bin_idx[k]) {
          break;
        }
      }
      if (k < j) {
        continue;
      }

      simple_hash_[bin_idx[j]].push_back(idx);
    }
  }

  for (size_t idx = 0; idx < simple_hash_.size(); ++idx) {
    max_bin_item_size_ = std::max(max_bin_item_size_, simple_hash_[idx].size());
  }
}

void MultiQueryServer::SetDatabase(yasl::ByteContainerView db_bytes) {
  std::vector<uint8_t> zero_bytes(query_options_.seal_options.element_size);
  std::memset(zero_bytes.data(), 0, query_options_.seal_options.element_size);

  for (size_t idx = 0; idx < cuckoo_params_.NumBins(); ++idx) {
    std::vector<yasl::ByteContainerView> db_vec;

    for (size_t j = 0; j < simple_hash_[idx].size(); ++j) {
      db_vec.emplace_back(yasl::ByteContainerView(
          &db_bytes[simple_hash_[idx][j] *
                    query_options_.seal_options.element_size],
          query_options_.seal_options.element_size));
    }
    for (size_t j = simple_hash_[idx].size(); j < max_bin_item_size_; ++j) {
      db_vec.emplace_back(yasl::ByteContainerView(zero_bytes));
    }

    pir_server_[idx]->SetDatabase(db_vec);
  }
}

void MultiQueryServer::RecvGaloisKeys(
    const std::shared_ptr<yasl::link::Context> &link_ctx) {
  yasl::Buffer galkey_buffer = link_ctx->Recv(
      link_ctx->NextRank(),
      fmt::format("recv galios key from rank-{}", link_ctx->Rank()));

  std::string galkey_str(galkey_buffer.size(), '\0');
  std::memcpy(&galkey_str[0], galkey_buffer.data(), galkey_buffer.size());
  seal::GaloisKeys galkey =
      pir_server_[0]->DeSerializeSealObject<seal::GaloisKeys>(galkey_str);
  SetGaloisKeys(galkey);
}

void MultiQueryServer::DoMultiPirAnswer(
    const std::shared_ptr<yasl::link::Context> &link_ctx) {
  yasl::Buffer multi_query_buffer =
      link_ctx->Recv(link_ctx->NextRank(), fmt::format("recv multi pir query"));

  SealMultiPirQueryProto multi_query_proto;
  multi_query_proto.ParseFromArray(multi_query_buffer.data(),
                                   multi_query_buffer.size());

  YASL_ENFORCE((uint64_t)multi_query_proto.querys().size() ==
               cuckoo_params_.NumBins());

  std::vector<yasl::Buffer> reply_cipher_buffers(
      multi_query_proto.querys().size());

  yasl::parallel_for(
      0, multi_query_proto.querys().size(), 1, [&](int64_t begin, int64_t end) {
        for (int64_t idx = begin; idx < end; ++idx) {
          std::vector<std::vector<seal::Ciphertext>> query_ciphers =
              pir_server_[idx]->DeSerializeQuery(multi_query_proto.querys(idx));

          std::vector<seal::Ciphertext> query_reply =
              pir_server_[idx]->GenerateReply(query_ciphers);
          reply_cipher_buffers[idx] =
              pir_server_[idx]->SerializeCiphertexts(query_reply);
        }
      });

  SealMultiPirAnswerProto mpir_answer_reply_proto;
  for (int idx = 0; idx < multi_query_proto.querys().size(); ++idx) {
    SealPirAnswerProto *answer = mpir_answer_reply_proto.add_answers();
    answer->set_query_size(0);
    answer->set_start_pos(0);
    answer->mutable_answer()->ParseFromArray(reply_cipher_buffers[idx].data(),
                                             reply_cipher_buffers[idx].size());
  }

  yasl::Buffer mpir_answer_buffer(mpir_answer_reply_proto.ByteSizeLong());
  mpir_answer_reply_proto.SerializePartialToArray(mpir_answer_buffer.data(),
                                                  mpir_answer_buffer.size());

  link_ctx->SendAsync(
      link_ctx->NextRank(), mpir_answer_buffer,
      fmt::format("send mpir reply buffer size:{}", mpir_answer_buffer.size()));

  return;
}

void MultiQueryClient::GenerateSimpleHashMap() {
  std::vector<uint128_t> query_index_hash(
      query_options_.seal_options.element_number);

  // generate item hash, server and client use same seed
  yasl::parallel_for(0, query_options_.seal_options.element_number, 1,
                     [&](int64_t begin, int64_t end) {
                       for (int idx = begin; idx < end; ++idx) {
                         query_index_hash[idx] = HashItemIndex(idx);
                       }
                     });

  size_t num_bins = cuckoo_params_.NumBins();
  simple_hash_map_.resize(num_bins);

  std::vector<size_t> simple_hash_counter(num_bins);
  for (size_t idx = 0; idx < num_bins; ++idx) {
    simple_hash_counter[idx] = 0;
  }

  for (size_t idx = 0; idx < query_options_.seal_options.element_number;
       ++idx) {
    spu::psi::CuckooIndex::HashRoom itemHash(query_index_hash[idx]);

    std::vector<uint64_t> bin_idx(query_options_.cuckoo_hash_number);
    for (size_t j = 0; j < query_options_.cuckoo_hash_number; ++j) {
      bin_idx[j] = itemHash.GetHash(j) % num_bins;
      size_t k = 0;
      for (; k < j; ++k) {
        if (bin_idx[j] == bin_idx[k]) {
          break;
        }
      }
      if (k < j) {
        continue;
      }
      // SPDLOG_INFO("bin index[{}]:{}", j, bin_idx[j]);
      simple_hash_map_[bin_idx[j]].emplace(query_index_hash[idx],
                                           simple_hash_counter[bin_idx[j]]);
      simple_hash_counter[bin_idx[j]]++;
    }
  }
  for (size_t idx = 0; idx < simple_hash_map_.size(); ++idx) {
    max_bin_item_size_ =
        std::max(max_bin_item_size_, simple_hash_map_[idx].size());
  }
}

std::vector<MultiQueryItem> MultiQueryClient::GenerateBatchQueryIndex(
    const std::vector<size_t> &multi_query_index) {
  std::vector<MultiQueryItem> multi_query(cuckoo_params_.NumBins());
  std::vector<uint128_t> query_index_hash(multi_query_index.size());

  yasl::parallel_for(
      0, multi_query_index.size(), 1, [&](int64_t begin, int64_t end) {
        for (int64_t idx = begin; idx < end; ++idx) {
          uint128_t item_hash = HashItemIndex(multi_query_index[idx]);

          query_index_hash[idx] = item_hash;
        }
      });

  spu::psi::CuckooIndex cuckoo_index(cuckoo_params_);
  cuckoo_index.Insert(query_index_hash);

  auto ck_bins = cuckoo_index.bins();

  std::random_device rd;

  std::mt19937 gen(rd());

  for (size_t idx = 0; idx < ck_bins.size(); ++idx) {
    if (ck_bins[idx].IsEmpty()) {
      // pad empty bin with random index
      multi_query[idx].db_index = 0;
      multi_query[idx].item_hash = 0;
      multi_query[idx].bin_item_index = gen() % simple_hash_map_[idx].size();

      continue;
    }
    size_t item_input_index = ck_bins[idx].InputIdx();

    uint128_t item_hash = query_index_hash[item_input_index];

    auto it = simple_hash_map_[idx].find(item_hash);
    if (it == simple_hash_map_[idx].end()) {
      continue;
    }

    multi_query[idx].db_index = multi_query_index[item_input_index];
    multi_query[idx].item_hash = item_hash;
    multi_query[idx].bin_item_index = it->second;
  }
  return multi_query;
}

void MultiQueryClient::SendGaloisKeys(
    const std::shared_ptr<yasl::link::Context> &link_ctx) {
  seal::GaloisKeys galkey = pir_client_->GenerateGaloisKeys();

  std::string galkey_str =
      pir_client_->SerializeSealObject<seal::GaloisKeys>(galkey);
  yasl::Buffer galkey_buffer(galkey_str.data(), galkey_str.length());

  link_ctx->SendAsync(
      link_ctx->NextRank(), galkey_buffer,
      fmt::format("send galios key to rank-{}", link_ctx->Rank()));
}

std::vector<std::vector<uint8_t>> MultiQueryClient::DoMultiPirQuery(
    const std::shared_ptr<yasl::link::Context> &link_ctx,
    const std::vector<size_t> &multi_query_index) {
  std::vector<MultiQueryItem> multi_query =
      GenerateBatchQueryIndex(multi_query_index);

  SealMultiPirQueryProto multi_query_proto;

  std::vector<SealPirQueryProto *> query_proto_vec(multi_query.size());
  for (size_t idx = 0; idx < multi_query.size(); ++idx) {
    query_proto_vec[idx] = multi_query_proto.add_querys();
  }

  for (size_t idx = 0; idx < multi_query.size(); ++idx) {
    std::vector<std::vector<seal::Ciphertext>> query_ciphers =
        pir_client_->GenerateQuery(multi_query[idx].bin_item_index);

    query_proto_vec[idx]->set_query_size(0);
    query_proto_vec[idx]->set_start_pos(0);
    for (size_t j = 0; j < query_ciphers.size(); ++j) {
      spu::pir::CiphertextsProto *ciphers_proto =
          query_proto_vec[idx]->add_query_cipher();

      for (size_t k = 0; k < query_ciphers[j].size(); ++k) {
        std::string cipher_bytes =
            pir_client_->SerializeSealObject<seal::Ciphertext>(
                query_ciphers[j][k]);

        ciphers_proto->add_ciphers(cipher_bytes.data(), cipher_bytes.length());
      }
    }
  }

  auto s = multi_query_proto.SerializeAsString();
  yasl::Buffer multi_query_buffer(s.data(), s.size());
  link_ctx->SendAsync(
      link_ctx->NextRank(), multi_query_buffer,
      fmt::format("send multi pir query number:{}", multi_query.size()));

  yasl::Buffer reply_buffer =
      link_ctx->Recv(link_ctx->NextRank(), fmt::format("recv pir answer"));

  SealMultiPirAnswerProto multi_answer_proto;
  multi_answer_proto.ParseFromArray(reply_buffer.data(), reply_buffer.size());
  // SPDLOG_INFO("multi_answer_proto size:{}",
  // multi_answer_proto.answers_size());

  YASL_ENFORCE((uint64_t)multi_answer_proto.answers_size() ==
               multi_query.size());

  std::vector<std::vector<uint8_t>> answers(multi_query_index.size());
  size_t answer_count = 0;

  for (int idx = 0; idx < multi_answer_proto.answers_size(); ++idx) {
    if (multi_query[idx].item_hash == 0) {
      continue;
    }
    for (size_t j = 0; j < multi_query_index.size(); ++j) {
      if (multi_query_index[j] != multi_query[idx].db_index) {
        continue;
      }
      CiphertextsProto answer_proto = multi_answer_proto.answers(idx).answer();
      std::vector<seal::Ciphertext> reply_ciphers =
          pir_client_->DeSerializeCiphertexts(answer_proto);

      seal::Plaintext query_plain = pir_client_->DecodeReply(reply_ciphers);

      std::vector<uint8_t> plaintext_bytes =
          pir_client_->PlaintextToBytes(query_plain);

      answers[j].resize(query_options_.seal_options.element_size);

      size_t offset =
          pir_client_->GetQueryOffset(multi_query[idx].bin_item_index);

      answer_count++;

      std::memcpy(
          answers[j].data(),
          &plaintext_bytes[offset * query_options_.seal_options.element_size],
          query_options_.seal_options.element_size);
      break;
    }
  }
  YASL_ENFORCE(answer_count == multi_query_index.size());

  return answers;
}

}  // namespace spu::pir
