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

#include "libspu/psi/core/bc22_psi/bc22_psi.h"

#include <algorithm>
#include <future>
#include <random>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/escaping.h"
#include "openssl/rand.h"
#include "spdlog/spdlog.h"
#include "yacl/crypto/utils/rand.h"
#include "yacl/utils/parallel.h"

#include "libspu/psi/core/bc22_psi/emp_vole.h"
#include "libspu/psi/utils/serialize.h"

#include "libspu/psi/utils/serializable.pb.h"

namespace spu::psi {

namespace {

constexpr size_t kDefaultHashNum = 2;
constexpr size_t kMaxItemsPerBin = 3;

constexpr size_t kPcgPsiBatchSize = 4096;
constexpr size_t kPcgPsiLogBatchSize = kPcgPsiBatchSize * 100;

constexpr size_t kMaxCompareBytes = 13;

// BC22 section 2.3
// H(i, vi) = H(i, wi-delta*ui)
std::vector<uint8_t> BaRKOPRFHash(size_t bin_idx,
                                  WolverineVoleFieldType value) {
  std::string hash_input(sizeof(WolverineVoleFieldType) + sizeof(size_t), '\0');
  std::memcpy(hash_input.data(), &bin_idx, sizeof(size_t));
  std::memcpy(hash_input.data() + sizeof(size_t), &value,
              sizeof(WolverineVoleFieldType));
  auto hash_res = yacl::crypto::Blake3(hash_input);
  return {hash_res.begin(), hash_res.end()};
}

}  // namespace

Bc22PcgPsi::Bc22PcgPsi(std::shared_ptr<yacl::link::Context> link_ctx,
                       PsiRoleType role)
    : link_ctx_(std::move(link_ctx)),
      role_(role),
      batch_size_(kPcgPsiBatchSize) {}

void Bc22PcgPsi::ExchangeItemsNumber(size_t self_item_num) {
  // exchange items number, compute compare bytes size
  // oprf compare bits: 40 + log2(n1) + log2(n2)

  yacl::Buffer self_count_buffer = utils::SerializeSize(self_item_num);
  link_ctx_->SendAsyncThrottled(
      link_ctx_->NextRank(), self_count_buffer,
      fmt::format("send items count: {}", self_item_num));

  yacl::Buffer peer_items_num_buffer =
      link_ctx_->Recv(link_ctx_->NextRank(), fmt::format("peer items number"));
  peer_items_num_ = utils::DeserializeSize(peer_items_num_buffer);
}

void Bc22PcgPsi::RunPsi(absl::Span<const std::string> items) {
  ExchangeItemsNumber(items.size());

  size_t compare_bytes_size =
      utils::GetCompareBytesLength(items.size(), peer_items_num_);

  SPDLOG_INFO("self size:{}, peer size:{} compare_bytes_size:{}", items.size(),
              peer_items_num_, compare_bytes_size);

  if (role_ == PsiRoleType::Sender) {
    cuckoo_options_ =
        GetCuckooHashOption(kMaxItemsPerBin, kDefaultHashNum, peer_items_num_);
    // sender alice

    // call mBaRK-OPRF
    std::string oprfs = RunmBaRKOprfSender(items, compare_bytes_size);

    // send sender's oprfs to receiver
    PcgPsiSendOprf(items, oprfs, compare_bytes_size);
  } else if (role_ == PsiRoleType::Receiver) {
    cuckoo_options_ =
        GetCuckooHashOption(kMaxItemsPerBin, kDefaultHashNum, items.size());
    // receiver bob

    // call mBaRK-OPRF
    std::vector<std::string> oprf_encode_vec =
        RunmBaRKOprfReceiver(items, compare_bytes_size);

    // receive sender's oprfs, and compute intersection
    PcgPsiRecvOprf(items, oprf_encode_vec, compare_bytes_size);
  } else {
    SPU_THROW("wrong psi role type: {}", static_cast<int>(role_));
  }
}

// mBaRK-OPRF
std::string Bc22PcgPsi::RunmBaRKOprfSender(absl::Span<const std::string> items,
                                           size_t compare_bytes_size) {
  WolverineVole vole(role_, link_ctx_);

  SimpleHashTable simple_table(cuckoo_options_);

  // insert simple hash table
  std::future<void> table_thread = std::async([&] {
    SPDLOG_INFO("begin insert simple hash table");

    simple_table.Insert(items);

    SPDLOG_INFO("after insert simple hash table");
  });

  uint64_t vole_count_needed = cuckoo_options_.NumBins() * kMaxItemsPerBin;

  // vole extension
  SPDLOG_INFO("begin pcg vole extension");
  // wi = delta * ui + vi
  // alice : delta wi
  // bob : ui || vi as one __uint128_t
  std::vector<WolverineVoleFieldType> vole_blocks =
      vole.Extend(vole_count_needed);
  SPDLOG_INFO("after pcg vole extension");

  WolverineVoleFieldType delta = vole.Delta();

  const size_t coeff_byte_size =
      kMaxItemsPerBin * sizeof(WolverineVoleFieldType);

  // sender alice
  // recv masked polynomial coeff
  size_t recv_bin_idx = 0;
  size_t bins_num = cuckoo_options_.NumBins();
  SPDLOG_INFO("cuckoo_options_.NumBins: {}", bins_num);

  SPDLOG_INFO("begin recv receiver's masked coeff");

  std::vector<std::array<WolverineVoleFieldType, kMaxItemsPerBin>>
      masked_coeffs(bins_num);

  while (true) {
    yacl::Buffer masked_coeff_buffer = link_ctx_->Recv(
        link_ctx_->NextRank(), fmt::format("recv {} bin", recv_bin_idx));
    SPU_ENFORCE((masked_coeff_buffer.size() % coeff_byte_size) == 0);

    size_t num_bin = masked_coeff_buffer.size() / coeff_byte_size;

    std::memcpy(&masked_coeffs[recv_bin_idx],
                reinterpret_cast<uint8_t *>(masked_coeff_buffer.data()),
                masked_coeff_buffer.size());

    recv_bin_idx += num_bin;
    // every kPcgPsiLogBatchSize bin print percentage
    if (recv_bin_idx % kPcgPsiLogBatchSize == 0) {
      SPDLOG_INFO(
          "recv receiver's masked coeff, recv_bin_idx: {} Bins_Num:{} "
          "percentage:{}",
          recv_bin_idx, bins_num, (double)recv_bin_idx / bins_num);
    }

    if (recv_bin_idx == bins_num) {
      break;
    }
  }

  SPDLOG_INFO("after recv receiver's masked coeff, recv_bin_idx: {}",
              recv_bin_idx);

  table_thread.get();

  const std::vector<std::vector<CuckooIndex::Bin>> &bins = simple_table.bins();

  const std::vector<uint64_t> &items_hash_low64 =
      simple_table.GetItemsHashLow64();

  SPDLOG_INFO("role:{} items:{} bins size: {},items_hash_low64 size: {}",
              (role_ == PsiRoleType::Sender) ? "sender" : "receiver",
              items.size(), bins.size(), items_hash_low64.size());

  std::string oprfs;
  oprfs.resize(items.size() * cuckoo_options_.num_hash * kMaxCompareBytes);

  // shuffle sender's oprfs
  std::mt19937 rng(yacl::crypto::SecureRandU64());

  std::vector<size_t> shuffled_idx_vec(items.size());
  std::iota(shuffled_idx_vec.begin(), shuffled_idx_vec.end(), 0);
  std::shuffle(shuffled_idx_vec.begin(), shuffled_idx_vec.end(), rng);

  // randomize conflict oprfs
  const std::vector<size_t> &conflict_idx = simple_table.GetConflictIdx();

  for (auto i : conflict_idx) {
    size_t shuffled_idx = shuffled_idx_vec[i];

    SPU_ENFORCE(
        RAND_bytes(reinterpret_cast<unsigned char *>(
                       &oprfs[((shuffled_idx * cuckoo_options_.num_hash)) *
                              compare_bytes_size]),
                   2 * compare_bytes_size) == 1);
  }

  // compute sender's oprf
  yacl::parallel_for(0, bins.size(), 1, [&](int64_t begin, int64_t end) {
    for (int64_t bin_idx = begin; bin_idx < end; ++bin_idx) {
      std::vector<WolverineVoleFieldType> oprf_key(kMaxItemsPerBin);

      size_t vole_start = bin_idx * kMaxItemsPerBin;

      std::array<WolverineVoleFieldType, kMaxItemsPerBin> masked_coeff =
          masked_coeffs[bin_idx];

      // delta * masked_coeff_i + w_i
      // = delta * coeff_i - delta * u_i + w_i
      // = delta * coeff_i + v_i
      for (size_t j = 0; j < kMaxItemsPerBin; ++j) {
        WolverineVoleFieldType tmp = mod(delta * masked_coeff[j], pr);

        oprf_key[j] = mod(vole_blocks[vole_start + j] + tmp, pr);
      }

      for (size_t j = 0; j < bins[bin_idx].size(); ++j) {
        size_t item_idx = bins[bin_idx][j].InputIdx();
        size_t hash_idx = bins[bin_idx][j].HashIdx();

        absl::string_view item_hash_str = absl::string_view(
            reinterpret_cast<const char *>(&items_hash_low64[item_idx]),
            sizeof(uint64_t));

        // delta * Poly(x) + vi_0 + vi_1*x+vi_1*x^2
        WolverineVoleFieldType eval =
            EvaluatePolynomial(absl::MakeSpan(oprf_key), item_hash_str, delta);

        std::vector<uint8_t> hash_res = BaRKOPRFHash(bin_idx, eval);

        // copy to shuffled pos
        size_t shuffled_idx = shuffled_idx_vec[item_idx];

        std::memcpy(
            &oprfs[((shuffled_idx * cuckoo_options_.num_hash) + hash_idx) *
                   compare_bytes_size],
            hash_res.data(), compare_bytes_size);
      }
    }
  });

  SPDLOG_INFO("after compute sender's oprf");

  return oprfs;
}

std::vector<std::string> Bc22PcgPsi::RunmBaRKOprfReceiver(
    absl::Span<const std::string> items, size_t compare_bytes_size) {
  WolverineVole vole(role_, link_ctx_);

  GeneralizedCuckooHashTable cuckoo_table(cuckoo_options_, kMaxItemsPerBin);

  uint64_t vole_count_needed = cuckoo_options_.NumBins() * kMaxItemsPerBin;

  std::future<void> table_thread = std::async([&] {
    SPDLOG_INFO("begin insert hash table");

    cuckoo_table.Insert(items);

    SPDLOG_INFO("after insert hash table");
  });

  // emp vole extension

  SPDLOG_INFO("begin pcg vole extension");
  // wi = delta * ui + vi
  // alice : delta wi
  // bob : ui || vi as one __uint128_t
  std::vector<WolverineVoleFieldType> vole_blocks =
      vole.Extend(vole_count_needed);
  SPDLOG_INFO("after pcg vole extension");

  table_thread.get();

  const std::vector<std::vector<CuckooIndex::Bin>> &bins = cuckoo_table.bins();

  const std::vector<uint64_t> &items_hash_low64 =
      cuckoo_table.GetItemsHashLow64();

  SPDLOG_INFO("role:{} items:{} bins size: {}, items_hash_low64 size: {}",
              (role_ == PsiRoleType::Sender) ? "sender" : "receiver",
              items.size(), bins.size(), items_hash_low64.size());

  const size_t coeff_byte_size =
      kMaxItemsPerBin * sizeof(WolverineVoleFieldType);

  // receiver bob
  // send mask
  std::vector<std::string> oprf_encode_vec(items.size());

  SPDLOG_INFO("begin compute and send receiver's masked coeff");

  for (size_t idx = 0; idx < bins.size(); idx += batch_size_) {
    size_t current_batch_size = std::min(batch_size_, bins.size() - idx);

    yacl::Buffer masked_coeff_buffer(coeff_byte_size * current_batch_size);

    std::vector<std::string> oprf_blocks_batch(current_batch_size);

    yacl::parallel_for(
        0, current_batch_size, 1, [&](int64_t begin, int64_t end) {
          for (int64_t j = begin; j < end; ++j) {
            size_t bin_idx = idx + j;

            std::vector<std::string> bin_data(kMaxItemsPerBin);
            size_t pos = j * coeff_byte_size;

            size_t k = 0;
            for (; k < bins[bin_idx].size(); ++k) {
              bin_data[k] = absl::string_view(
                  reinterpret_cast<const char *>(
                      &items_hash_low64[bins[bin_idx][k].InputIdx()]),
                  sizeof(uint64_t));  // use 64 bit
            }

            for (; k < kMaxItemsPerBin; ++k) {
              std::string buf(sizeof(uint64_t), '\0');
              SPU_ENFORCE(RAND_bytes(reinterpret_cast<uint8_t *>(buf.data()),
                                     buf.length()) == 1);
              bin_data[k] = buf;
            }
            std::vector<WolverineVoleFieldType> coeff_blocks =
                GetPolynomialCoefficients(bin_data);

            // use vole mask polynomial coefficient
            // coeff_i - ui
            size_t vole_start = (idx + j) * kMaxItemsPerBin;
            for (k = 0; k < coeff_blocks.size(); ++k) {
              coeff_blocks[k] = mod(
                  coeff_blocks[k] + (vole_blocks[vole_start + k] >> 64), pr);
            }

            // copy to masked_coeff send buffer
            std::memcpy(
                reinterpret_cast<uint8_t *>(masked_coeff_buffer.data()) + pos,
                coeff_blocks.data(), coeff_byte_size);

            // get vi_0, vi_1, vi_2, i: bin index
            std::vector<WolverineVoleFieldType> coeff_vole(kMaxItemsPerBin);
            for (k = 0; k < coeff_vole.size(); ++k) {
              coeff_vole[k] =
                  vole_blocks[vole_start + k] & 0xFFFFFFFFFFFFFFFFLL;
            }

            for (k = 0; k < bins[bin_idx].size(); ++k) {
              size_t item_index = bins[bin_idx][k].InputIdx();

              // get item_hash as x,
              absl::string_view item_hash_str = absl::string_view(
                  reinterpret_cast<const char *>(&items_hash_low64[item_index]),
                  sizeof(uint64_t));

              // compute vi_0 + vi_1*x +vi_2*x^2, i: bin index
              WolverineVoleFieldType eval =
                  EvaluatePolynomial(coeff_vole, item_hash_str, 0);

              // compute oprf, H(i, vi_0 + vi_1*x +vi_2*x^2), i: bin index
              std::vector<uint8_t> hash_res = BaRKOPRFHash(bin_idx, eval);

              oprf_encode_vec[item_index] =
                  absl::string_view(
                      reinterpret_cast<const char *>(hash_res.data()),
                      hash_res.size())
                      .substr(0, compare_bytes_size);
            }
          }
        });

    link_ctx_->SendAsyncThrottled(
        link_ctx_->NextRank(), masked_coeff_buffer,
        fmt::format("send {} bin", current_batch_size));
  }
  SPDLOG_INFO("after send receiver's masked coeff");

  return oprf_encode_vec;
}

void Bc22PcgPsi::PcgPsiSendOprf(absl::Span<const std::string> items,
                                const std::string &oprfs,
                                size_t compare_bytes_size) {
  SPDLOG_INFO("begin send sender's oprf");
  // send oprf
  // batch send oprf
  for (size_t i = 0; i < items.size(); i += batch_size_) {
    size_t current_batch_size = std::min(batch_size_, items.size() - i);
    bool is_last_batch = false;

    if ((i + current_batch_size) == items.size()) {
      is_last_batch = true;
    }

    proto::PsiDataBatchProto proto;
    proto.set_item_num(current_batch_size);
    std::string flatten_bytes = oprfs.substr(
        i * cuckoo_options_.num_hash * compare_bytes_size,
        current_batch_size * cuckoo_options_.num_hash * compare_bytes_size);

    proto.set_flatten_bytes(flatten_bytes);
    proto.set_is_last_batch(is_last_batch);
    yacl::Buffer oprf_buffer(proto.ByteSizeLong());
    proto.SerializeToArray(oprf_buffer.data(), oprf_buffer.size());

    link_ctx_->SendAsyncThrottled(
        link_ctx_->NextRank(), oprf_buffer,
        fmt::format("send oprf buffer, bytes: {}", oprf_buffer.size()));
  }

  SPDLOG_INFO("after send sender's oprf");
}

void Bc22PcgPsi::PcgPsiRecvOprf(absl::Span<const std::string> items,
                                const std::vector<std::string> &oprf_encode_vec,
                                size_t compare_bytes_size) {
  SPDLOG_INFO("begin recv sender's oprf");

  // recv oprf
  std::string sender_oprf(
      peer_items_num_ * cuckoo_options_.num_hash * compare_bytes_size, '\0');

  size_t oprf_count = 0;
  while (true) {
    yacl::Buffer oprf_buffer =
        link_ctx_->Recv(link_ctx_->NextRank(), fmt::format("recv oprf buffer"));

    proto::PsiDataBatchProto proto;
    proto.ParseFromArray(oprf_buffer.data(), oprf_buffer.size());

    size_t current_batch_size = proto.item_num();
    const std::string &flatten_bytes = proto.flatten_bytes();
    bool is_last_batch = proto.is_last_batch();

    SPDLOG_DEBUG(
        "recv oprf_buffer size:{} item_num:{} "
        "is_last_batch:{} "
        "flatten_bytes:{}",
        oprf_buffer.size(), current_batch_size, is_last_batch,
        flatten_bytes.length());

    std::memcpy(&sender_oprf[oprf_count * cuckoo_options_.num_hash *
                             compare_bytes_size],
                flatten_bytes.data(), flatten_bytes.length());

    oprf_count += current_batch_size;

    // every kPcgPsiLogBatchSize bin print percentage
    if (oprf_count % kPcgPsiLogBatchSize == 0) {
      SPDLOG_INFO(
          "recv sender's oprf, oprf_count: {} peer_items_num_:{} "
          "percentage:{}",
          oprf_count, peer_items_num_, (double)oprf_count / peer_items_num_);
    }

    SPDLOG_DEBUG(" oprf_count:{}", oprf_count);

    if (is_last_batch) {
      break;
    }
  }

  SPDLOG_INFO("after recv sender's oprf");

  // https://abseil.io/docs/cpp/guides/container#abslflat_hash_map-and-abslflat_hash_set
  // absl flat_hash_set faster than std::unordered_set
  std::vector<absl::flat_hash_set<std::string>> sender_oprf_set(
      cuckoo_options_.num_hash);

  auto sender_oprf_insert_proc = [&](size_t hash_idx) -> void {
    size_t oprf_idx = 0;
    for (size_t i = 0; i < peer_items_num_; ++i) {
      std::string oprf = sender_oprf.substr(
          (oprf_idx + hash_idx) * compare_bytes_size, compare_bytes_size);
      sender_oprf_set[hash_idx].insert(std::move(oprf));

      oprf_idx += cuckoo_options_.num_hash;
    }
  };

  std::vector<std::future<void>> f_set;
  for (size_t i = 0; i < cuckoo_options_.num_hash; ++i) {
    f_set.push_back(std::async(sender_oprf_insert_proc, i));
  }

  for (size_t i = 0; i < cuckoo_options_.num_hash; ++i) {
    f_set[i].get();
  }

  SPDLOG_INFO("after insert unordered_set");

  SPDLOG_INFO("begin compute intersection");

  std::vector<std::string> results2;

  std::vector<size_t> result_by_hash0(cuckoo_options_.num_hash);
  std::vector<size_t> result_by_hash1(cuckoo_options_.num_hash);

  auto intersection_proc = [&](size_t begin, size_t end,
                               std::vector<std::string> *results_vec,
                               std::vector<size_t> *result_by_hash) -> void {
    for (size_t i = begin; i < end; ++i) {
      for (size_t j = 0; j < cuckoo_options_.num_hash; ++j) {
        auto it = sender_oprf_set[j].find(oprf_encode_vec[i]);
        if (it != sender_oprf_set[j].end()) {
          (*results_vec).emplace_back(items[i]);
          (*result_by_hash)[j]++;
          break;
        }
      }
    }
  };

  std::future<void> f_intersection =
      std::async(intersection_proc, items.size() / 2, items.size(), &results2,
                 &result_by_hash1);

  intersection_proc(0, items.size() / 2, &results_, &result_by_hash0);

  f_intersection.get();

  SPDLOG_INFO("result:{}, result2:{}", results_.size(), results2.size());

  results_.insert(results_.end(), results2.begin(), results2.end());

  result_by_hash0[0] += result_by_hash1[0];
  result_by_hash0[1] += result_by_hash1[1];
  SPDLOG_INFO("hash_count0:{}, hash_count1:{}", result_by_hash0[0],
              result_by_hash0[1]);

  // sender use only one cuckooHash to insert SimpleHash
  // may leak some information about receiver's items hash
  if ((result_by_hash0[0] == 0) || (result_by_hash0[1] == 0)) {
    SPDLOG_WARN("*** may be attacked");
  }

  SPDLOG_INFO("oprf_count:{} intersection size:{}", oprf_count,
              results_.size());

  SPDLOG_INFO("after compute intersection");
}

}  // namespace spu::psi
