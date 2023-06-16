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

#include "libspu/psi/core/mini_psi.h"

#include <future>
#include <map>
#include <random>
#include <set>
#include <unordered_set>

#include "absl/strings/escaping.h"
#include "absl/strings/string_view.h"
#include "openssl/crypto.h"
#include "openssl/rand.h"
#include "spdlog/spdlog.h"

extern "C" {
#include "curve25519.h"
}

#include "yacl/crypto/base/hash/hash_utils.h"
#include "yacl/crypto/base/symmetric_crypto.h"
#include "yacl/crypto/tools/prg.h"
#include "yacl/utils/parallel.h"

#include "libspu/core/prelude.h"
#include "libspu/psi/core/communication.h"
#include "libspu/psi/core/cuckoo_index.h"
#include "libspu/psi/core/polynomial/polynomial.h"
#include "libspu/psi/utils/batch_provider.h"
#include "libspu/psi/utils/serialize.h"

namespace spu::psi {

namespace {

constexpr uint32_t kLinkRecvTimeout = 30 * 60 * 1000;
// first prime over 2^256, used as module for polynomial interpolate
std::string kPrimeOver256bHexStr =
    "010000000000000000000000000000000000000000000000000000000000000129";

// batch size of Cuckoo Hash
constexpr size_t kCuckooHashBatchSize = 2000;

std::vector<std::string> HashInputs(const std::vector<std::string>& items) {
  std::vector<std::string> ret(items.size());
  yacl::parallel_for(0, items.size(), 1, [&](int64_t begin, int64_t end) {
    for (int64_t idx = begin; idx < end; ++idx) {
      auto hash = yacl::crypto::Sha256(items[idx]);
      ret[idx].resize(hash.size());
      std::memcpy(ret[idx].data(), hash.data(), hash.size());
    }
  });
  return ret;
}

struct MiniPsiSendCtx {
  MiniPsiSendCtx() {
    yacl::crypto::Prg<uint64_t> prg(0, yacl::crypto::PRG_MODE::kNistAesCtrDrbg);
    prg.Fill(absl::MakeSpan(private_key.data(), kKeySize));

    curve25519_donna_basepoint(static_cast<unsigned char*>(public_key.data()),
                               private_key.data());

    uint128_t aes_key = yacl::crypto::Blake3_128(public_key);
    aes_ecb = std::make_shared<yacl::crypto::SymmetricCrypto>(
        yacl::crypto::SymmetricCrypto::CryptoType::AES128_ECB, aes_key, 0);

    prime256_str = absl::HexStringToBytes(kPrimeOver256bHexStr);
  }

  void RecvPolynomialCoeff(
      const std::shared_ptr<yacl::link::Context>& link_ctx) {
    size_t batch_count = 0;

    yacl::link::RecvTimeoutGuard guard(link_ctx, kLinkRecvTimeout);
    while (true) {
      const auto tag = fmt::format("MINI-PSI:X^A:{}", batch_count);
      PsiDataBatch coeff_batch =
          PsiDataBatch::Deserialize(link_ctx->Recv(link_ctx->NextRank(), tag));
      // Fetch y^b.
      SPU_ENFORCE(coeff_batch.flatten_bytes.size() % kHashSize == 0);
      size_t num_items = coeff_batch.flatten_bytes.size() / kHashSize;

      if (num_items > 0) {
        absl::string_view flatten_bytes = coeff_batch.flatten_bytes;

        for (size_t i = 0; i < num_items; ++i) {
          polynomial_coeff.emplace_back(
              flatten_bytes.substr(i * kHashSize, kHashSize));
        }
      }

      if (coeff_batch.is_last_batch) {
        break;
      }
      batch_count++;
    }
  }

  void EvalPolynomial(const std::vector<std::string>& items) {
    polynomial_eval_values.resize(items.size());
    masked_values.resize(items.size());

    items_hash = HashInputs(items);

    yacl::parallel_for(0, items.size(), 1, [&](int64_t begin, int64_t end) {
      for (int64_t idx = begin; idx < end; ++idx) {
        polynomial_eval_values[idx] = spu::psi::EvalPolynomial(
            polynomial_coeff, absl::string_view(items_hash[idx]), prime256_str);

        std::array<uint8_t, kKeySize> ideal_permutation;
        // Ideal Permutation
        aes_ecb->Decrypt(absl::MakeSpan(reinterpret_cast<uint8_t*>(
                                            polynomial_eval_values[idx].data()),
                                        polynomial_eval_values[idx].length()),
                         absl::MakeSpan(ideal_permutation));

        std::string masked(kKeySize, '\0');

        curve25519_donna(
            reinterpret_cast<unsigned char*>(masked.data()), private_key.data(),
            static_cast<const unsigned char*>(ideal_permutation.data()));

        yacl::crypto::Sha256Hash sha256;
        sha256.Update(items[idx].data());
        sha256.Update(masked.data());
        std::vector<uint8_t> mask_hash = sha256.CumulativeHash();
        masked_values[idx].resize(kFinalCompareBytes);
        std::memcpy(masked_values[idx].data(), mask_hash.data(),
                    kFinalCompareBytes);
      }
    });

    // use sort as shuffle
    std::sort(masked_values.begin(), masked_values.end());
  }

  void SendMaskedEvalValues(
      const std::shared_ptr<yacl::link::Context>& link_ctx) {
    size_t batch_count = 0;

    std::shared_ptr<IBatchProvider> batch_provider =
        std::make_shared<MemoryBatchProvider>(masked_values);
    size_t batch_size = kEcdhPsiBatchSize;

    while (true) {
      PsiDataBatch batch;
      // NOTE: we still need to send one batch even there is no data.
      // This dummy batch is used to notify peer the end of data stream.
      auto items = batch_provider->ReadNextBatch(batch_size);
      batch.is_last_batch = items.empty();
      // Mask and Send this batch.
      if (!items.empty()) {
        batch.flatten_bytes.reserve(items.size() * kFinalCompareBytes);

        for (const auto& item : items) {
          batch.flatten_bytes.append(item);
        }
      }
      // Send x^a.
      const auto tag = fmt::format("MINI-PSI:X^A:{}", batch_count);
      link_ctx->SendAsyncThrottled(link_ctx->NextRank(), batch.Serialize(),
                                   tag);
      if (batch.is_last_batch) {
        SPDLOG_INFO("Last batch triggered, batch_count={}", batch_count);
        break;
      }
      batch_count++;
    }
  }

  // key
  std::array<uint8_t, kKeySize> private_key;
  std::array<uint8_t, kKeySize> public_key;

  // next prime over 2^256
  std::string prime256_str;

  // hash of items
  std::vector<std::string> items_hash;

  // polynomial_coeff
  std::vector<std::string> polynomial_coeff;

  std::vector<std::string> polynomial_eval_values;
  std::vector<std::string> masked_values;

  // use aes-128-ecb as Ideal Permutation
  std::shared_ptr<yacl::crypto::SymmetricCrypto> aes_ecb;
};

struct MiniPsiRecvCtx {
  MiniPsiRecvCtx() {
    prime256_str = absl::HexStringToBytes(kPrimeOver256bHexStr);
  }

  void GenerateSeeds(size_t data_size) {
    seeds.resize(data_size);
    seeds_point.resize(data_size);

    yacl::parallel_for(0, data_size, 1, [&](int64_t begin, int64_t end) {
      for (int64_t idx = begin; idx < end; ++idx) {
        yacl::crypto::Prg<uint64_t> prg(
            0, yacl::crypto::PRG_MODE::kNistAesCtrDrbg);
        prg.Fill(absl::MakeSpan(seeds[idx].data(), kKeySize));

        curve25519_donna_basepoint(
            static_cast<unsigned char*>(seeds_point[idx].data()),
            seeds[idx].data());
      }
    });
  }

  void InterpolatePolynomial(const std::vector<std::string>& items) {
    items_hash = HashInputs(items);

    std::vector<absl::string_view> poly_x(items_hash.size());
    std::vector<absl::string_view> poly_y(items_hash.size());
    std::vector<std::array<uint8_t, kKeySize>> poly_y_permutation(
        items_hash.size());

    for (size_t idx = 0; idx < items_hash.size(); idx++) {
      poly_x[idx] = absl::string_view(items_hash[idx]);

      // Ideal Permutation
      aes_ecb->Encrypt(absl::MakeSpan(seeds_point[idx]),
                       absl::MakeSpan(poly_y_permutation[idx]));

      poly_y[idx] = absl::string_view(
          reinterpret_cast<const char*>(poly_y_permutation[idx].data()),
          kKeySize);
    }

    // ToDo: now use newton Polynomial Interpolation, need optimize to fft
    //
    polynomial_coeff =
        spu::psi::InterpolatePolynomial(poly_x, poly_y, prime256_str);
  }

  void SendPolynomialCoeff(
      const std::shared_ptr<yacl::link::Context>& link_ctx) {
    size_t batch_count = 0;

    std::shared_ptr<IBatchProvider> batch_provider =
        std::make_shared<MemoryBatchProvider>(polynomial_coeff);
    size_t batch_size = kEcdhPsiBatchSize;

    while (true) {
      PsiDataBatch batch;
      // NOTE: we still need to send one batch even there is no data.
      // This dummy batch is used to notify peer the end of data stream.
      auto items = batch_provider->ReadNextBatch(batch_size);
      batch.is_last_batch = items.empty();
      // Mask and Send this batch.
      if (!items.empty()) {
        batch.flatten_bytes.reserve(items.size() * kHashSize);

        for (const auto& item : items) {
          batch.flatten_bytes.append(item);
        }
      }
      // Send x^a.
      const auto tag = fmt::format("MINI-PSI:X^A:{}", batch_count);
      link_ctx->SendAsyncThrottled(link_ctx->NextRank(), batch.Serialize(),
                                   tag);
      if (batch.is_last_batch) {
        SPDLOG_INFO("Last batch triggered, batch_count={}", batch_count);
        break;
      }
      batch_count++;
    }
  }

  void RecvMaskedEvalValues(
      const std::shared_ptr<yacl::link::Context>& link_ctx) {
    size_t batch_count = 0;

    yacl::link::RecvTimeoutGuard guard(link_ctx, kLinkRecvTimeout);
    while (true) {
      const auto tag = fmt::format("MINI-PSI:X^A^B:{}", batch_count);
      PsiDataBatch masked_eval_batch =
          PsiDataBatch::Deserialize(link_ctx->Recv(link_ctx->NextRank(), tag));
      // Fetch y^b.
      SPU_ENFORCE(masked_eval_batch.flatten_bytes.size() % kFinalCompareBytes ==
                  0);
      size_t num_items =
          masked_eval_batch.flatten_bytes.size() / kFinalCompareBytes;

      if (num_items > 0) {
        absl::string_view flatten_bytes = masked_eval_batch.flatten_bytes;

        for (size_t i = 0; i < num_items; ++i) {
          peer_masked_values.emplace(
              flatten_bytes.substr(i * kFinalCompareBytes, kFinalCompareBytes));
        }
      }
      if (masked_eval_batch.is_last_batch) {
        break;
      }
      batch_count++;
    }
  }

  void MaskPeerPublicKey(const std::vector<std::string>& items) {
    masked_values.resize(seeds.size());

    yacl::parallel_for(0, seeds.size(), 1, [&](int64_t begin, int64_t end) {
      for (int64_t idx = begin; idx < end; ++idx) {
        std::string masked(kKeySize, '\0');
        curve25519_donna(reinterpret_cast<unsigned char*>(masked.data()),
                         seeds[idx].data(), peer_public_key.data());

        yacl::crypto::Sha256Hash sha256;
        sha256.Update(items[idx].data());
        sha256.Update(masked.data());
        std::vector<uint8_t> mask_hash = sha256.CumulativeHash();
        masked_values[idx].resize(kFinalCompareBytes);
        std::memcpy(masked_values[idx].data(), mask_hash.data(),
                    kFinalCompareBytes);
      }
    });
  }

  std::vector<std::string> GetIntersection(
      const std::vector<std::string>& items) {
    std::vector<std::string> ret;

    for (uint32_t index = 0; index < masked_values.size(); index++) {
      if (peer_masked_values.find(masked_values[index]) !=
          peer_masked_values.end()) {
        ret.push_back(items[index]);
      }
    }

    return ret;
  }

  std::vector<std::array<uint8_t, kKeySize>> seeds;
  std::vector<std::array<uint8_t, kKeySize>> seeds_point;

  // peer's public key
  std::array<uint8_t, kKeySize> peer_public_key;

  // next prime over 2^256
  std::string prime256_str;

  // hash of items
  std::vector<std::string> items_hash;

  // polynomial_coeff
  std::vector<std::string> polynomial_coeff;

  // dual mask value
  std::vector<std::string> masked_values;
  // peer's dual mask value
  std::unordered_set<std::string> peer_masked_values;

  // use aes-128-ecb as Ideal Permutation
  std::shared_ptr<yacl::crypto::SymmetricCrypto> aes_ecb;
};

}  // namespace

// #define DEBUG_OUT

void MiniPsiSend(const std::shared_ptr<yacl::link::Context>& link_ctx,
                 const std::vector<std::string>& items) {
  MiniPsiSendCtx send_ctx;

  //
  // TODO: whether use zk to prove sender's public_key
  //    https://github.com/osu-crypto/MiniPSI/blob/master/libPSI/MiniPSI/MiniSender.cpp#L601
  //    MiniPSI code use zk prove public_key (discrete logarithm)
  //    in the origin paper no use zk
  //
  link_ctx->SendAsyncThrottled(
      link_ctx->NextRank(),
      yacl::Buffer(send_ctx.public_key.data(), send_ctx.public_key.size()),
      "MINI-PSI:X^A");

  // receive Polynomial Coefficient
  send_ctx.RecvPolynomialCoeff(link_ctx);

  std::future<void> f_eval =
      std::async([&] { send_ctx.EvalPolynomial(items); });

  f_eval.get();

  // send Polynomial evaluation and mask value to receiver
  send_ctx.SendMaskedEvalValues(link_ctx);
}

std::vector<std::string> MiniPsiRecv(
    const std::shared_ptr<yacl::link::Context>& link_ctx,
    const std::vector<std::string>& items) {
  MiniPsiRecvCtx recv_ctx;

  std::future<void> f_get_pubkey = std::async([&] {
    // receive sender's public key
    yacl::Buffer buf =
        link_ctx->Recv(link_ctx->NextRank(), fmt::format("MINI-PSI:X^A"));
    std::memcpy(recv_ctx.peer_public_key.data(), buf.data(), buf.size());

    uint128_t aes_key = yacl::crypto::Blake3_128(recv_ctx.peer_public_key);
    recv_ctx.aes_ecb = std::make_shared<yacl::crypto::SymmetricCrypto>(
        yacl::crypto::SymmetricCrypto::CryptoType::AES128_ECB, aes_key, 0);
  });

  std::future<void> f_gen_seeds = std::async([&] {
    // generate seed
    recv_ctx.GenerateSeeds(items.size());
  });
  f_get_pubkey.get();
  f_gen_seeds.get();

  std::future<void> f_interpolate =
      std::async([&] { recv_ctx.InterpolatePolynomial(items); });

  f_interpolate.get();

  // send polynomial coefficient to sender
  recv_ctx.SendPolynomialCoeff(link_ctx);

  std::future<void> f_mask_peer =
      std::async([&] { return recv_ctx.MaskPeerPublicKey(items); });

  f_mask_peer.get();

  // get sender's masked value
  recv_ctx.RecvMaskedEvalValues(link_ctx);

  // get intersection
  return recv_ctx.GetIntersection(items);
}

// big data
void MiniPsiSendBatch(const std::shared_ptr<yacl::link::Context>& link_ctx,
                      const std::vector<std::string>& items) {
  size_t peer_size = utils::DeserializeSize(
      link_ctx->Recv(link_ctx->NextRank(), fmt::format("RECV PEER SIZE")));

  CuckooIndex::Options option = CuckooIndex::SelectParams(peer_size, 0, 3);

  size_t num_bins = option.NumBins();
  std::vector<std::string> items_hash = HashInputs(items);
  //
  std::vector<std::vector<std::string>> simple_hash(num_bins);

  //
  for (size_t idx = 0; idx < items.size(); idx++) {
    uint128_t items_u128;
    std::memcpy(&items_u128, items_hash[idx].data(), sizeof(uint128_t));
    CuckooIndex::HashRoom itemHash(items_u128);
    uint64_t bin_idx0 = itemHash.GetHash(0) % num_bins;
    uint64_t bin_idx1 = itemHash.GetHash(1) % num_bins;
    uint64_t bin_idx2 = itemHash.GetHash(2) % num_bins;
    std::set<uint64_t> bin_idx_set;
    bin_idx_set.insert(bin_idx0);
    bin_idx_set.insert(bin_idx1);
    bin_idx_set.insert(bin_idx2);

    for (const auto& idx_inter : bin_idx_set) {
      simple_hash[idx_inter].push_back(items[idx]);
    }
  }
  size_t nthread = utils::DeserializeSize(link_ctx->Recv(
      link_ctx->NextRank(), fmt::format("Mini-PSI RECV THREAD Num")));

  auto thread = [&](const std::shared_ptr<yacl::link::Context>& thread_link_ctx,
                    size_t thread_idx) {
    size_t start_idx = num_bins * thread_idx / nthread;
    size_t end_idx = num_bins * (thread_idx + 1) / nthread;

    for (size_t idx = start_idx; idx < end_idx; idx += kCuckooHashBatchSize) {
      size_t current_batch_size = std::min(kCuckooHashBatchSize, end_idx - idx);
      std::set<std::string> batch_items_set;
      std::vector<std::string> batch_items_vec;
      for (size_t batch_idx = 0; batch_idx < current_batch_size; batch_idx++) {
        for (auto& item : simple_hash[idx + batch_idx]) {
          batch_items_set.insert(item);
        }
      }
      batch_items_vec.assign(batch_items_set.begin(), batch_items_set.end());
      SPDLOG_INFO("thread:{}, batch_idx:{}/{}, batch_items size:{} ",
                  thread_idx, idx, end_idx, batch_items_vec.size());
      MiniPsiSend(thread_link_ctx, batch_items_vec);
    }
  };

  std::vector<std::future<void>> futures;
  std::vector<std::shared_ptr<yacl::link::Context>> thread_link_ctxs(nthread);

  for (size_t thread_idx = 0; thread_idx < nthread; ++thread_idx) {
    thread_link_ctxs[thread_idx] = link_ctx->Spawn();
    futures.push_back(
        std::async(thread, thread_link_ctxs[thread_idx], thread_idx));
  }
  // wait thread
  for (auto& f : futures) {
    f.get();
  }
}

std::vector<std::string> MiniPsiRecvBatch(
    const std::shared_ptr<yacl::link::Context>& link_ctx,
    const std::vector<std::string>& items) {
  // send size to peer
  link_ctx->SendAsyncThrottled(link_ctx->NextRank(),
                               utils::SerializeSize(items.size()), "RECV SIZE");

  std::vector<std::string> items_hash = HashInputs(items);
  std::vector<uint128_t> items_hash_u128(items.size());

  yacl::parallel_for(0, items.size(), 1, [&](int64_t begin, int64_t end) {
    for (int64_t idx = begin; idx < end; ++idx) {
      std::memcpy(&items_hash_u128[idx], items_hash[idx].data(),
                  sizeof(uint128_t));
    }
  });
  // cuckoo hash
  CuckooIndex::Options option = CuckooIndex::SelectParams(items.size(), 0, 3);
  CuckooIndex cuckoo_index(option);

  cuckoo_index.Insert(absl::MakeSpan(items_hash_u128));

  SPU_ENFORCE(cuckoo_index.stash().empty(), "stash size not 0");

  size_t nthreads = yacl::intraop_default_num_threads();
  // send thread num
  if (items.size() < 100000) {
    nthreads = 1;
  } else {
    nthreads /= 8;
  }
  link_ctx->Send(link_ctx->NextRank(), utils::SerializeSize(nthreads),
                 "Mini-PSI SEND THREAD NUM");

  size_t num_bins = option.NumBins();
  auto ck_bins = cuckoo_index.bins();

  std::vector<std::string> ret;
  std::vector<size_t> ret_idx;
  std::vector<std::vector<size_t>> ret_idx_vec(nthreads);

  // thread func
  auto thread = [&](const std::shared_ptr<yacl::link::Context>& thread_link_ctx,
                    size_t thread_idx) {
    size_t start_idx = num_bins * thread_idx / nthreads;
    size_t end_idx = num_bins * (thread_idx + 1) / nthreads;

    std::random_device rd;
    yacl::crypto::Prg<uint64_t> prg(rd());

    for (size_t idx = start_idx; idx < end_idx; idx += kCuckooHashBatchSize) {
      size_t current_batch_size = std::min(kCuckooHashBatchSize, end_idx - idx);
      std::vector<std::string> batch_items;
      std::unordered_map<std::string, size_t> batch_map;
      for (size_t batch_idx = 0; batch_idx < current_batch_size; batch_idx++) {
        // real data
        if (!ck_bins[idx + batch_idx].IsEmpty()) {
          batch_items.push_back(items[ck_bins[idx + batch_idx].InputIdx()]);
          batch_map.emplace(items[ck_bins[idx + batch_idx].InputIdx()],
                            ck_bins[idx + batch_idx].InputIdx());
        } else {
          // insert padding data
          std::string padding_data(kKeySize, '\0');
          prg.Fill(absl::MakeSpan(padding_data.data(), padding_data.length()));
          batch_items.push_back(padding_data);
        }
      }
      SPDLOG_INFO("thread:{}, batch_idx:{}/{}, batch_items size:{} ",
                  thread_idx, idx, end_idx, batch_items.size());

      std::vector<std::string> intersection =
          MiniPsiRecv(thread_link_ctx, batch_items);

      for (auto& batch_idx : intersection) {
        auto it = batch_map.find(batch_idx);
        ret_idx_vec[thread_idx].push_back(it->second);
      }
    }
    std::sort(ret_idx_vec[thread_idx].begin(), ret_idx_vec[thread_idx].end());
  };

  std::vector<std::future<void>> futures;
  std::vector<std::shared_ptr<yacl::link::Context>> thread_link_ctxs(nthreads);

  for (size_t thread_idx = 0; thread_idx < nthreads; ++thread_idx) {
    thread_link_ctxs[thread_idx] = link_ctx->Spawn();
    futures.push_back(
        std::async(thread, thread_link_ctxs[thread_idx], thread_idx));
  }
  // wait thread
  for (auto& f : futures) {
    f.get();
  }

  for (size_t thread_idx = 0; thread_idx < nthreads; ++thread_idx) {
    ret_idx.insert(ret_idx.begin(), ret_idx_vec[thread_idx].begin(),
                   ret_idx_vec[thread_idx].end());
  }
  std::sort(ret_idx.begin(), ret_idx.end());

  ret.reserve(ret_idx.size());
  for (auto idx : ret_idx) {
    ret.push_back(items[idx]);
  }

  return ret;
}

}  // namespace spu::psi