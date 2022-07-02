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

#include "spu/psi/core/ecdh_psi.h"

#include <future>
#include <utility>

#include "spdlog/spdlog.h"
#include "yasl/base/exception.h"
#include "yasl/crypto/hash_util.h"
#include "yasl/utils/parallel.h"
#include "yasl/utils/serialize.h"

#include "spu/psi/cryptor/cryptor_selector.h"
#include "spu/psi/provider/batch_provider_impl.h"
#include "spu/psi/store/cipher_store_impl.h"

namespace spu::psi {
namespace {

std::string CreateFlattenEccBuffer(const std::vector<std::string>& items,
                                   size_t item_size,
                                   size_t chosen_size = kEccKeySize) {
  std::string ret;
  ret.reserve(items.size() * item_size);
  size_t size = std::min<size_t>(chosen_size, item_size);
  for (const auto& item : items) {
    YASL_ENFORCE(item.size() == item_size);
    ret.append(item.data(), size);
  }
  return ret;
}

std::string CreateFlattenEccBuffer(const std::vector<absl::string_view>& items,
                                   size_t item_size,
                                   size_t chosen_size = kEccKeySize) {
  std::string ret;
  ret.reserve(items.size() * item_size);
  size_t size = std::min<size_t>(chosen_size, item_size);
  for (const auto& item : items) {
    YASL_ENFORCE(item.size() == item_size);
    ret.append(item.data(), size);
  }
  return ret;
}

std::vector<std::string> CreateItemsFromFlattenEccBuffer(
    std::string_view buf, size_t item_size = kEccKeySize) {
  YASL_ENFORCE(buf.size() % item_size == 0);
  size_t num_item = buf.size() / item_size;
  std::vector<std::string> ret;
  ret.reserve(num_item);
  for (size_t i = 0; i < num_item; i++) {
    ret.emplace_back(buf.data() + i * item_size, item_size);
  }
  return ret;
}

struct RunContext {
  // Target rank which can touch the psi results.
  PsiOptions options;

  // Link context.
  //
  // Q: Does single link work ?
  // A: No. Remember the link is not thread-safe. We need dual channels due to
  // there are two logic flows.
  //
  // Q: Link usages?
  // A: - link0 is used to send/recv the single encrypted items
  //    - link1 is used to send/recv the dual encrypted items
  std::shared_ptr<yasl::link::Context> link0;
  std::shared_ptr<yasl::link::Context> link1;

  // When Alice & Bob have unbalanced compute power, we need some mechanism
  // to avoid the accumulation of ciphertext. This mechanism only consume
  // constant memory when you run PSI for many items.
  //
  // Note such mechanism relys on the feedback from peer.
  std::mutex window_mutex;
  std::condition_variable window_cv;
  size_t finished_batch_count = 0;

  bool CanTouchResults() const {
    return this->options.target_rank == yasl::link::kAllRank ||
           this->options.target_rank == this->link0->Rank();
  }

  bool PeerCanTouchResults() const {
    return this->options.target_rank == yasl::link::kAllRank ||
           this->options.target_rank == this->link0->NextRank();
  }

  // TODO(shuyan.ycf): strong-typed inputs.
  std::vector<std::string> Mask(const std::vector<std::string>& items) const {
    std::string batch_points =
        CreateFlattenEccBuffer(items, options.ecc_cryptor->GetMaskLength(),
                               options.ecc_cryptor->GetMaskLength());
    std::string out_points(batch_points.size(), '\0');
    options.ecc_cryptor->EccMask(batch_points, absl::MakeSpan(out_points));
    return CreateItemsFromFlattenEccBuffer(
        out_points, options.ecc_cryptor->GetMaskLength());
  }

  // TODO(shuyan.ycf): strong-typed inputs.
  std::vector<std::string> Mask(
      const std::vector<absl::string_view>& items) const {
    std::string batch_points =
        CreateFlattenEccBuffer(items, options.ecc_cryptor->GetMaskLength(),
                               options.ecc_cryptor->GetMaskLength());
    std::string out_points(batch_points.size(), '\0');
    options.ecc_cryptor->EccMask(batch_points, absl::MakeSpan(out_points));
    return CreateItemsFromFlattenEccBuffer(
        out_points, options.ecc_cryptor->GetMaskLength());
  }

  std::string HashInput(const std::string& item) const {
    std::vector<uint8_t> sha_bytes = options.ecc_cryptor->HashToCurve(item);
    std::string ret(sha_bytes.size(), '\0');
    std::memcpy(&ret[0], sha_bytes.data(), sha_bytes.size());
    return ret;
  }

  std::vector<std::string> HashInputs(
      const std::vector<std::string>& items) const {
    std::vector<std::string> ret(items.size());
    yasl::parallel_for(0, items.size(), 1, [&](int64_t begin, int64_t end) {
      for (int64_t idx = begin; idx < end; ++idx) {
        ret[idx] = HashInput(items[idx]);
      }
    });
    return ret;
  }
};

std::shared_ptr<RunContext> CreateRunContext(const PsiOptions& options) {
  auto run_ctx = std::make_shared<RunContext>();
  run_ctx->options = options;
  run_ctx->link0 = options.link_ctx;
  run_ctx->link1 = options.link_ctx->Spawn();
  return run_ctx;
}

void RunMaskSelf(std::shared_ptr<RunContext> run_ctx) {
  size_t batch_count = 0;
  while (true) {
    details::EcdhBatch batch;
    // NOTE: we still need to send one batch even there is no data.
    // This dummy batch is used to notify peer the end of data stream.
    auto items = run_ctx->options.batch_provider->ReadNextBatch(
        run_ctx->options.batch_size);
    batch.is_last_batch = items.empty();
    // Mask and Send this batch.
    if (!items.empty()) {
      batch.flatten_bytes.reserve(
          items.size() * run_ctx->options.ecc_cryptor->GetMaskLength());
      auto masked_items = run_ctx->Mask(run_ctx->HashInputs(items));
      for (const auto& masked_item : masked_items) {
        batch.flatten_bytes.append(masked_item);
      }
    }
    // Send x^a.
    const auto tag = fmt::format("ECDHPSI:X^A:{}", batch_count);
    run_ctx->link0->SendAsync(run_ctx->link0->NextRank(), batch.Serialize(),
                              tag);
    if (batch.is_last_batch) {
      SPDLOG_INFO("Last batch triggered, batch_count={}", batch_count);
      break;
    }
    batch_count++;
    if (run_ctx->CanTouchResults()) {
      // Throttle to limit max flighting batches.
      std::unique_lock<std::mutex> lock(run_ctx->window_mutex);
      auto now = std::chrono::system_clock::now();
      YASL_ENFORCE(run_ctx->window_cv.wait_until(
                       lock,
                       now + std::chrono::milliseconds(
                                 run_ctx->options.window_throttle_timeout_ms),
                       [&]() {
                         return batch_count - run_ctx->finished_batch_count <=
                                run_ctx->options.window_size;
                       }),
                   "Timeout when waiting for the finished batch to catch up, "
                   "batch_count={}, finished_batch_count={}",
                   batch_count, run_ctx->finished_batch_count);
    }
  }
}

void RunMaskPeer(const std::shared_ptr<RunContext>& run_ctx,
                 size_t dual_mask_size = kFinalCompareBytes) {
  const auto& link0 = run_ctx->link0;
  const auto& link1 = run_ctx->link1;
  size_t batch_count = 0;
  while (true) {
    const auto tag = fmt::format("ECDHPSI:X^A:{}", batch_count);
    details::EcdhBatch masked_batch =
        details::EcdhBatch::Deserialize(link0->Recv(link0->NextRank(), tag));
    // Fetch y^b.
    size_t mask_length = run_ctx->options.ecc_cryptor->GetMaskLength();
    YASL_ENFORCE(masked_batch.flatten_bytes.size() % mask_length == 0);
    size_t num_items = masked_batch.flatten_bytes.size() / mask_length;

    // Compute (y^b)^a.
    details::EcdhBatch dual_masked_batch;
    dual_masked_batch.is_last_batch = masked_batch.is_last_batch;

    if (num_items > 0) {
      absl::string_view flatten_bytes = masked_batch.flatten_bytes;
      std::vector<absl::string_view> y_b;
      for (size_t i = 0; i < num_items; ++i) {
        y_b.push_back(flatten_bytes.substr(i * mask_length, mask_length));
      }
      dual_masked_batch.flatten_bytes.reserve(num_items * dual_mask_size);
      for (const auto& masked : run_ctx->Mask(y_b)) {
        // In the final comparison, we only send & compare `kFinalCompareBytes`
        // number of bytes.
        // for multi-party case, intermediate should use kHashSize
        std::string cipher = masked.substr(0, dual_mask_size);
        dual_masked_batch.flatten_bytes.append(cipher);
        if (run_ctx->CanTouchResults()) {
          // Store cipher of peer items for later intersection compute.
          run_ctx->options.cipher_store->SavePeer(cipher);
        }
      }
    }
    // Should send out the dual masked items to peer.
    if (run_ctx->PeerCanTouchResults()) {
      const auto tag = fmt::format("ECDHPSI:X^A^B:{}", batch_count);
      link1->SendAsync(link1->NextRank(), dual_masked_batch.Serialize(), tag);
    }
    if (dual_masked_batch.is_last_batch) {
      break;
    }
    batch_count++;
  }
}

void RunRecvPeer(const std::shared_ptr<RunContext>& run_ctx,
                 size_t dual_mask_size = kFinalCompareBytes) {
  YASL_ENFORCE(run_ctx->CanTouchResults());

  size_t batch_count = 0;
  while (true) {
    // Receive x^a^b.
    const auto tag = fmt::format("ECDHPSI:X^A^B:{}", batch_count);
    details::EcdhBatch batch = details::EcdhBatch::Deserialize(
        run_ctx->link1->Recv(run_ctx->link1->NextRank(), tag));

    const bool is_last_batch = batch.is_last_batch;
    YASL_ENFORCE(batch.flatten_bytes.size() % dual_mask_size == 0);
    const size_t num_items = batch.flatten_bytes.size() / dual_mask_size;

    for (size_t i = 0; i < num_items; ++i) {
      run_ctx->options.cipher_store->SaveSelf(
          batch.flatten_bytes.substr(i * dual_mask_size, dual_mask_size));
    }
    if (is_last_batch) {
      break;
    }
    batch_count++;
    {
      std::unique_lock<std::mutex> lock(run_ctx->window_mutex);
      run_ctx->finished_batch_count = batch_count;
    }
    run_ctx->window_cv.notify_one();

    // Call the hook.
    if (run_ctx->options.on_batch_finished) {
      run_ctx->options.on_batch_finished(batch_count);
    }
  }
}

}  // namespace

void RunEcdhPsi(const PsiOptions& options) {
  YASL_ENFORCE(options.link_ctx->WorldSize() == 2);
  YASL_ENFORCE(options.target_rank == yasl::link::kAllRank ||
               options.target_rank < options.link_ctx->WorldSize());

  // Sanity check: the `target_rank` and 'curve_type' should match.
  std::string my_config =
      fmt::format("target_rank={},curve={}", options.target_rank,
                  static_cast<int>(options.ecc_cryptor->GetCurveType()));
  const auto* const tag = "ECDHPSI:SANITY";
  options.link_ctx->Send(options.link_ctx->NextRank(), my_config, tag);
  auto peer_config = options.link_ctx->Recv(options.link_ctx->NextRank(), tag);
  YASL_ENFORCE(std::string_view(my_config) == std::string_view(peer_config),
               "Config mismatch, mine={}, peer={}", my_config, peer_config);

  std::vector<uint32_t> indices;
  auto ctx = CreateRunContext(options);

  std::future<void> f_mask_self = std::async(RunMaskSelf, ctx);
  std::future<void> f_mask_peer =
      std::async(RunMaskPeer, ctx, kFinalCompareBytes);
  std::future<void> f_recv_peer;
  if (ctx->CanTouchResults()) {
    f_recv_peer = std::async(RunRecvPeer, ctx, kFinalCompareBytes);
  }

  // Wait for end of logic flows or exceptions.
  // Note: exception_ptr is `shared_ptr` style, hence could be used to prolong
  // the lifetime of pointed exceptions.
  std::exception_ptr mask_self_exptr = nullptr;
  std::exception_ptr mask_peer_exptr = nullptr;
  std::exception_ptr recv_peer_exptr = nullptr;

  try {
    f_mask_self.get();
  } catch (const std::exception& e) {
    mask_self_exptr = std::current_exception();
    SPDLOG_ERROR("Error in mask self: {}", e.what());
  }

  try {
    f_mask_peer.get();
  } catch (const std::exception& e) {
    mask_peer_exptr = std::current_exception();
    SPDLOG_ERROR("Error in mask peer: {}", e.what());
  }

  if (ctx->CanTouchResults()) {
    try {
      f_recv_peer.get();
    } catch (const std::exception& e) {
      recv_peer_exptr = std::current_exception();
      SPDLOG_ERROR("Error in recv peer: {}", e.what());
    }
  }

  if (mask_self_exptr) {
    std::rethrow_exception(mask_self_exptr);
  }
  if (mask_peer_exptr) {
    std::rethrow_exception(mask_peer_exptr);
  }
  if (recv_peer_exptr) {
    std::rethrow_exception(recv_peer_exptr);
  }
}

std::vector<std::string> RunEcdhPsi(
    const std::shared_ptr<yasl::link::Context>& link_ctx,
    const std::vector<std::string>& items, size_t target_rank,
    CurveType curve) {
  PsiOptions options;
  auto memory_store = std::make_shared<MemoryCipherStore>();
  {
    options.ecc_cryptor = CreateEccCryptor(curve);
    options.batch_provider = std::make_shared<MemoryBatchProvider>(items);
    options.cipher_store = memory_store;
    options.link_ctx = link_ctx;
    options.target_rank = target_rank;
  }

  RunEcdhPsi(options);

  // Originally we should setup a hashset for peer results.
  // But tests show that when items_count > 10,000,000, the performance of
  // |std::unordered_set| or |absl::flat_hash_set| drops significantly.
  // Besides, these hashset containers require more memory.
  // Here we choose the compact data structure and stable find costs.
  std::vector<std::string> ret;
  std::vector<std::string> peer_results(memory_store->peer_results());
  std::sort(peer_results.begin(), peer_results.end());
  const auto& self_results = memory_store->self_results();
  for (uint32_t index = 0; index < self_results.size(); index++) {
    if (std::binary_search(peer_results.begin(), peer_results.end(),
                           self_results[index])) {
      YASL_ENFORCE(index < items.size());
      ret.push_back(items[index]);
    }
  }
  return ret;
}

namespace {

// add send_rank for link context world size > 2
void RunMaskSelf(std::shared_ptr<RunContext> run_ctx, size_t send_rank) {
  size_t batch_count = 0;
  while (true) {
    details::EcdhBatch batch;
    // NOTE: we still need to send one batch even there is no data.
    // This dummy batch is used to notify peer the end of data stream.
    auto items = run_ctx->options.batch_provider->ReadNextBatch(
        run_ctx->options.batch_size);
    batch.is_last_batch = items.empty();
    // Mask and Send this batch.
    if (!items.empty()) {
      batch.flatten_bytes.reserve(items.size() * kHashSize);
      auto masked_items = run_ctx->Mask(run_ctx->HashInputs(items));
      for (const auto& masked_item : masked_items) {
        batch.flatten_bytes.append(masked_item);
      }
    }
    // Send x^a.
    const auto tag = fmt::format("ECDHPSI:X^A:{}", batch_count);
    run_ctx->link0->SendAsync(send_rank, batch.Serialize(), tag);
    if (batch.is_last_batch) {
      SPDLOG_INFO("Last batch triggered, batch_count={}", batch_count);
      break;
    }
    batch_count++;
    if (run_ctx->CanTouchResults()) {
      // Throttle to limit max flighting batches.
      std::unique_lock<std::mutex> lock(run_ctx->window_mutex);
      auto now = std::chrono::system_clock::now();
      YASL_ENFORCE(run_ctx->window_cv.wait_until(
                       lock,
                       now + std::chrono::milliseconds(
                                 run_ctx->options.window_throttle_timeout_ms),
                       [&]() {
                         return batch_count - run_ctx->finished_batch_count <=
                                run_ctx->options.window_size;
                       }),
                   "Timeout when waiting for the finished batch to catch up, "
                   "batch_count={}, finished_batch_count={}",
                   batch_count, run_ctx->finished_batch_count);
    }
  }
}

// add recv_rank, send_rank for link Context world size > 2
void RunMaskPeer(std::shared_ptr<RunContext> run_ctx, size_t recv_rank,
                 size_t send_rank, size_t dual_mask_size = kFinalCompareBytes) {
  const auto& link0 = run_ctx->link0;
  const auto& link1 = run_ctx->link1;
  size_t batch_count = 0;
  while (true) {
    const auto tag = fmt::format("ECDHPSI:X^A:{}", batch_count);
    details::EcdhBatch masked_batch =
        details::EcdhBatch::Deserialize(link0->Recv(recv_rank, tag));
    // Fetch y^b.
    YASL_ENFORCE(masked_batch.flatten_bytes.size() % kHashSize == 0);
    size_t num_items = masked_batch.flatten_bytes.size() / kHashSize;

    // Compute (y^b)^a.
    details::EcdhBatch dual_masked_batch;
    dual_masked_batch.is_last_batch = masked_batch.is_last_batch;
    if (num_items > 0) {
      absl::string_view flatten_bytes_sv = masked_batch.flatten_bytes;
      std::vector<absl::string_view> y_b;
      for (size_t i = 0; i < num_items; ++i) {
        y_b.push_back(flatten_bytes_sv.substr(i * kHashSize, kHashSize));
      }
      // reserve size for dual_masked_batch.flatten_bytes
      dual_masked_batch.flatten_bytes.reserve(num_items * dual_mask_size);

      for (const auto& masked : run_ctx->Mask(y_b)) {
        // In the final comparison, we only send & compare `kFinalCompareBytes`
        // number of bytes.
        std::string cipher = masked.substr(0, dual_mask_size);
        dual_masked_batch.flatten_bytes.append(cipher);
        if (run_ctx->CanTouchResults()) {
          // Store cipher of peer items for later intersection compute.
          run_ctx->options.cipher_store->SavePeer(cipher);
        }
      }
    }
    // Should send out the dual masked items to peer.
    if (run_ctx->PeerCanTouchResults()) {
      const auto tag = fmt::format("ECDHPSI:X^A^B:{}", batch_count);
      link1->SendAsync(send_rank, dual_masked_batch.Serialize(), tag);
    }
    if (dual_masked_batch.is_last_batch) {
      break;
    }
    batch_count++;
  }
}

void RunRecvLinkRank(std::shared_ptr<RunContext> run_ctx, size_t recv_rank,
                     size_t dual_mask_size = kFinalCompareBytes) {
  YASL_ENFORCE(run_ctx->CanTouchResults());

  size_t batch_count = 0;
  while (true) {
    // Receive x^a^b.
    const auto tag = fmt::format("ECDHPSI:X^A^B:{}", batch_count);
    details::EcdhBatch batch =
        details::EcdhBatch::Deserialize(run_ctx->link1->Recv(recv_rank, tag));

    const bool is_last_batch = batch.is_last_batch;

    YASL_ENFORCE(batch.flatten_bytes.size() % dual_mask_size == 0,
                 "link_{} {}%{}", run_ctx->link1->Rank(),
                 batch.flatten_bytes.size(), dual_mask_size);

    const size_t num_items = batch.flatten_bytes.size() / dual_mask_size;

    for (size_t i = 0; i < num_items; ++i) {
      run_ctx->options.cipher_store->SaveSelf(
          batch.flatten_bytes.substr(i * dual_mask_size, dual_mask_size));
    }
    if (is_last_batch) {
      break;
    }
    batch_count++;
    {
      std::unique_lock<std::mutex> lock(run_ctx->window_mutex);
      run_ctx->finished_batch_count = batch_count;
    }
    run_ctx->window_cv.notify_one();

    // Call the hook.
    if (run_ctx->options.on_batch_finished) {
      run_ctx->options.on_batch_finished(batch_count);
    }
  }
}

std::shared_ptr<RunContext> CreateRunContext(
    const PsiOptions& options,
    const std::shared_ptr<yasl::link::Context>& link0,
    const std::shared_ptr<yasl::link::Context>& link1) {
  auto run_ctx = std::make_shared<RunContext>();
  run_ctx->options = options;
  run_ctx->link0 = link0;
  run_ctx->link1 = link1;
  return run_ctx;
}

}  // namespace

EcdhPsiOp::EcdhPsiOp(const PsiOptions& options) : options_(options) {}

void EcdhPsiOp::MaskSelf(size_t target_rank, size_t send_rank) {
  PsiOptions tmp_options = options_;
  tmp_options.target_rank = target_rank;

  auto ctx =
      CreateRunContext(tmp_options, tmp_options.link_ctx, tmp_options.link_ctx);

  RunMaskSelf(ctx, send_rank);
}

void EcdhPsiOp::MaskPeer(size_t target_rank, size_t recv_rank, size_t send_rank,
                         const std::shared_ptr<yasl::link::Context>& link_ctx,
                         size_t dual_mask_size) {
  PsiOptions tmp_options = options_;
  tmp_options.target_rank = target_rank;

  auto ctx = CreateRunContext(tmp_options, tmp_options.link_ctx, link_ctx);

  RunMaskPeer(ctx, recv_rank, send_rank, dual_mask_size);
}

void EcdhPsiOp::SendBatch(
    size_t send_rank, const std::shared_ptr<IBatchProvider>& batch_provider) {
  auto ctx = CreateRunContext(options_, options_.link_ctx, options_.link_ctx);

  size_t batch_count = 0;
  while (true) {
    details::EcdhBatch batch;
    // NOTE: we still need to send one batch even there is no data.
    // This dummy batch is used to notify peer the end of data stream.
    auto items = batch_provider->ReadNextBatch(ctx->options.batch_size);
    batch.is_last_batch = items.empty();

    for (const auto& item : items) {
      batch.flatten_bytes.append(item);
    }
    // Send x^a.
    const auto tag = fmt::format("ECDHPSI:X^A:{}", batch_count);

    ctx->link0->SendAsync(send_rank, batch.Serialize(), tag);

    if (batch.is_last_batch) {
      SPDLOG_INFO("Last batch triggered, batch_count={}", batch_count);
      break;
    }
    batch_count++;
  }
}

void EcdhPsiOp::RecvPeer(size_t target_rank, size_t recv_rank,
                         const std::shared_ptr<yasl::link::Context>& link_ctx,
                         size_t dual_mask_size) {
  PsiOptions tmp_options = options_;
  tmp_options.target_rank = target_rank;

  auto ctx = CreateRunContext(tmp_options, tmp_options.link_ctx, link_ctx);

  RunRecvLinkRank(ctx, recv_rank, dual_mask_size);
}

std::vector<std::string> HashInputs(const std::vector<std::string>& items) {
  std::vector<std::string> ret(items.size());
  yasl::parallel_for(0, items.size(), 1, [&](int64_t begin, int64_t end) {
    for (int64_t idx = begin; idx < end; ++idx) {
      std::vector<uint8_t> hash = yasl::crypto::Sha256(items[idx]);
      ret[idx].resize(hash.size());
      std::memcpy(&ret[idx][0], hash.data(), hash.size());
    }
  });
  return ret;
}
}  // namespace spu::psi
