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

#include "spu/psi/core/kkrt_psi.h"

#include <algorithm>
#include <future>
#include <numeric>

#include "absl/strings/escaping.h"
#include "openssl/crypto.h"
#include "openssl/rand.h"
#include "spdlog/spdlog.h"
#include "yasl/base/exception.h"
#include "yasl/crypto/hash_util.h"
#include "yasl/crypto/utils.h"
#include "yasl/mpctools/ot/base_ot.h"
#include "yasl/mpctools/ot/iknp_ot_extension.h"
#include "yasl/mpctools/ot/kkrt_ot_extension.h"
#include "yasl/utils/rand.h"

#include "spu/psi/core/cuckoo_index.h"
#include "spu/psi/core/throttle_control_link.h"
#include "spu/psi/core/utils.h"

#include "spu/psi/core/serializable.pb.h"

namespace spu::psi {

namespace {
// constexpr size_t kKkrtPsiBatchSize = 1 << 10;
constexpr size_t kKkrtPsiBatchSize = 1024;
constexpr size_t kStashSize = 0;
constexpr size_t kCuckooHashNum = 3;
constexpr size_t kStatSecParam = 40;

constexpr size_t kKkrtOtBatchSize = (65535 / 4 / 16 * 0.8);

struct KkrtPsiBatch {
  // current batch item num
  uint64_t item_num;

  // Pack all items in a single `std::string` to save bandwidth.
  std::string flatten_bytes;

  // Metadata.
  bool is_last_batch = false;

  yasl::Buffer Serialize() const {
    KkrtPsiBatchProto proto;
    proto.set_item_num(item_num);
    proto.set_flatten_bytes(flatten_bytes);
    proto.set_is_last_batch(is_last_batch);
    yasl::Buffer ret(proto.ByteSizeLong());
    proto.SerializeToArray(ret.data(), ret.size());
    return ret;
  }

  static KkrtPsiBatch Deserialize(const yasl::Buffer& buf) {
    KkrtPsiBatchProto proto;
    proto.ParseFromArray(buf.data(), buf.size());

    KkrtPsiBatch batch;
    batch.item_num = proto.item_num();
    batch.flatten_bytes = proto.flatten_bytes();
    batch.is_last_batch = proto.is_last_batch();

    return batch;
  }
};

struct KkrtPsiRunContext {
  size_t self_size = 0;
  size_t peer_size = 0;

  size_t kkrt_ot_num = 0;

  // Link context.
  std::shared_ptr<yasl::link::Context> link0;
  std::shared_ptr<yasl::link::Context> link1;

  size_t finished_batch_count = 0;
};

// send set size to peer
// get peer's item size
//
size_t ExchangeSetSize(const std::shared_ptr<yasl::link::Context>& link_ctx,
                       size_t items_size) {
  size_t input_size = items_size;

  link_ctx->SendAsync(link_ctx->NextRank(), utils::SerializeSize(input_size),
                      fmt::format("KKRT:PSI:SELF_SIZE={}", items_size));

  size_t peer_size = utils::DeserializeSize(
      link_ctx->Recv(link_ctx->NextRank(), fmt::format("KKRT:PSI:PEER_SIZE")));

  return peer_size;
}

std::shared_ptr<KkrtPsiRunContext> CreateReceiverRunContext(
    const std::shared_ptr<yasl::link::Context>& link_ctx,
    const std::vector<uint128_t>& items, CuckooIndex* cuckoo_index) {
  YASL_ENFORCE(cuckoo_index != nullptr);
  auto run_ctx = std::make_shared<KkrtPsiRunContext>();

  run_ctx->link0 = link_ctx;
  run_ctx->link1 = link_ctx->Spawn();
  run_ctx->self_size = items.size();

  cuckoo_index->Insert(absl::MakeSpan(items));

  YASL_ENFORCE(cuckoo_index->stash().empty(), "stash size not 0");

  run_ctx->kkrt_ot_num = cuckoo_index->bins().size();

  return run_ctx;
}

std::shared_ptr<KkrtPsiRunContext> CreateSenderRunContext(
    const std::shared_ptr<yasl::link::Context>& link_ctx,
    const std::vector<uint128_t>& items) {
  auto run_ctx = std::make_shared<KkrtPsiRunContext>();

  run_ctx->link0 = link_ctx;
  run_ctx->link1 = link_ctx->Spawn();
  run_ctx->self_size = items.size();

  return run_ctx;
}

inline bool GetBit(const std::vector<uint128_t>& choices, size_t idx) {
  uint128_t mask = uint128_t(1) << (idx & 127);
  return (choices[idx / 128] & mask) ? true : false;
}

//
// stat_sec_param = 40, data_size 2^40 encode size is 15
// data_size > 2^40, use encode size 16,
// hash collision probability will > 2^-40
inline uint64_t KkrtEncodeSize(uint64_t stat_sec_param, uint128_t self_size,
                               uint128_t peer_size) {
  uint64_t encode_size =
      (stat_sec_param + std::log2l(self_size * peer_size) + 7) / 8;
  return std::min(encode_size, static_cast<uint64_t>(sizeof(uint128_t)));
}

void SendOprfEncode(std::shared_ptr<KkrtPsiRunContext> run_ctx,
                    const KkrtPsiOptions& kkrt_psi_options,
                    const std::vector<uint128_t>& items,
                    yasl::KkrtOtExtSender& sender) {
  // permute sender input data
  std::vector<size_t> input_permute;
  input_permute.resize(run_ctx->self_size);
  std::iota(input_permute.begin(), input_permute.end(), 0);

  yasl::PseudoRandomGenerator<uint128_t> prg(yasl::RandSeed());

  uint64_t mt_seed;
  prg.Fill(
      absl::MakeSpan(reinterpret_cast<uint8_t*>(&mt_seed), sizeof(mt_seed)));

  std::mt19937 rng(mt_seed);
  std::shuffle(input_permute.begin(), input_permute.end(), rng);

  uint64_t encode_size =
      KkrtEncodeSize(kkrt_psi_options.stat_sec_param, run_ctx->self_size,
                     run_ctx->peer_size);  // by byte

  size_t num_bins = run_ctx->kkrt_ot_num;

  sender.SetBatchSize(kKkrtOtBatchSize);
  uint64_t kkrtOtBatchSize = sender.GetBatchSize();

  size_t batch_count = 0;

  std::atomic<uint64_t> recv_idx(0);
  auto f_recv_corrections = std::async([&]() {
    ThrottleControlReceiver throttle_control_receiver(run_ctx->link0,
                                                      run_ctx->link1);
    // while there are more corrections for be recieved
    size_t correction_batch_idx = 0;
    while (recv_idx < num_bins) {
      // compute the  size of the current step and the end index
      size_t current_step_size = std::min(kkrtOtBatchSize, num_bins - recv_idx);

      // receive the corrections.
      auto current_correction_buf =
          throttle_control_receiver.RecvMsgWithSendCounter();
      std::string current_correction_str(current_correction_buf.data<char>(),
                                         current_correction_buf.size());
      sender.SetCorrection(current_correction_str, current_step_size);
      correction_batch_idx++;

      // notify the other thread that the corrections have arrived
      recv_idx.fetch_add(current_step_size, std::memory_order_release);
    }
  });

  size_t oprf_buf_size =
      run_ctx->self_size * kkrt_psi_options.cuckoo_hash_num * encode_size;

  std::vector<uint8_t> encode_buf(oprf_buf_size);

  std::vector<std::array<uint64_t, kCuckooHashNum>> bin_indices;
  bin_indices.resize(run_ctx->self_size);
  for (size_t i = 0; i < run_ctx->self_size; ++i) {
    CuckooIndex::HashRoom itemHash(items[i]);
    uint64_t bin_idx0 = itemHash.GetHash(0) % num_bins;
    uint64_t bin_idx1 = itemHash.GetHash(1) % num_bins;
    uint64_t bin_idx2 = itemHash.GetHash(2) % num_bins;

    bin_indices[i][0] = bin_idx0;
    // check collision
    uint8_t c01 = (bin_idx0 == bin_idx1) ? 1 : 0;
    bin_indices[i][1] = bin_idx1 | (c01 * uint64_t(-1));
    uint8_t c02 = (bin_idx0 == bin_idx2 || bin_idx1 == bin_idx2) ? 1 : 0;
    bin_indices[i][2] = bin_idx2 | (c02 * uint64_t(-1));
    if (c01 == 1) {
      uint8_t* encode_pos =
          encode_buf.data() +
          (input_permute[i] * kkrt_psi_options.cuckoo_hash_num + 1) *
              encode_size;
      prg.Fill(absl::MakeSpan(encode_pos, encode_size));
    }
    if (c02 == 1) {
      uint8_t* encode_pos =
          encode_buf.data() +
          (input_permute[i] * kkrt_psi_options.cuckoo_hash_num + 2) *
              encode_size;
      prg.Fill(absl::MakeSpan(encode_pos, encode_size));
    }
  }

  uint64_t t = 0;
  uint64_t r = 0;
  // while not all the corrections have been recieved, try to encode any that
  // we can
  // TODO(shuyan.ycf): this implementation wastes cpus if the networking is
  // slow (Due to spin logics). Better use synchoronization primitives.
  while (r != num_bins) {
    // process things in steps
    for (uint64_t j = 0; j < kkrtOtBatchSize; ++j) {
      // lets check a random item to see if it can be encoded. If so,
      // we will write this item's encodings in the myMaskBuff at position i.
      auto input_idx = input_permute[t];

      // for each hash function, try to encode the item.
      for (uint64_t h = 0; h < kkrt_psi_options.cuckoo_hash_num; ++h) {
        uint64_t b_idx = bin_indices[input_idx][h];

        // if the bin index is less than r, then we have recieved
        // the correction and can encode it
        if (b_idx < r) {
          // write the encoding into encode_buf at position  t, h
          uint8_t* encoding =
              encode_buf.data() +
              (t * kkrt_psi_options.cuckoo_hash_num + h) * encode_size;
          sender.Encode(b_idx, items[input_idx], encoding, encode_size);

          // make this location as already been encoded
          bin_indices[input_idx][h] = -1;
        }
      }

      // wrap around the input looking for items that we can encode
      t = (t + 1) % run_ctx->self_size;
    }

    // after stepSize attempts to encode items, lets see if more
    // corrections have arrived.
    r = recv_idx.load(std::memory_order_acquire);
  }

  // Join receiving thread and throw exceptions if any thing is wrong.
  f_recv_corrections.get();

  // add batch Send
  size_t encode_batch_num =
      (run_ctx->self_size + kkrt_psi_options.psi_batch_size - 1) /
      kkrt_psi_options.psi_batch_size;

  ThrottleControlSender throttle_control_sender(run_ctx->link0, run_ctx->link1,
                                                encode_batch_num);
  throttle_control_sender.StartRecvThread();

  for (size_t i = 0; i < run_ctx->self_size;) {
    size_t curr_step_item_num =
        std::min(kkrt_psi_options.psi_batch_size, run_ctx->self_size - i);
    size_t curr_step_encode_num =
        curr_step_item_num * kkrt_psi_options.cuckoo_hash_num;
    KkrtPsiBatch batch;

    batch.flatten_bytes.resize(encode_size * curr_step_encode_num);
    uint8_t* encoding = encode_buf.data() +
                        (i * kkrt_psi_options.cuckoo_hash_num) * encode_size;
    for (size_t j = 0; j < curr_step_item_num; ++j) {
      auto input_idx = input_permute[i + j];

      for (size_t k = 0; k < kkrt_psi_options.cuckoo_hash_num; k++) {
        uint64_t b_idx = bin_indices[input_idx][k];

        if (b_idx != uint64_t(-1)) {
          sender.Encode(b_idx, items[input_idx], encoding, encode_size);
        }
        encoding += encode_size;
      }
    }

    batch.item_num = curr_step_item_num;
    batch.is_last_batch = false;

    encoding = encode_buf.data() +
               (i * kkrt_psi_options.cuckoo_hash_num) * encode_size;
    memcpy(batch.flatten_bytes.data(), encoding,
           encode_size * curr_step_encode_num);

    i += curr_step_item_num;

    if (i == run_ctx->self_size) {
      batch.is_last_batch = true;
    }
    batch_count++;
    throttle_control_sender.SendAsync(
        batch.Serialize(),
        fmt::format("KKRT:PSI:SENDER OPRF:{}", curr_step_item_num));
  }
  throttle_control_sender.WaitRecvThread();

  run_ctx->finished_batch_count = batch_count;
}
}  // namespace

KkrtPsiOptions GetDefaultKkrtPsiOptions() {
  KkrtPsiOptions kkrt_psi_options;

  kkrt_psi_options.ot_batch_size = kKkrtOtBatchSize;
  kkrt_psi_options.psi_batch_size = kKkrtPsiBatchSize;

  kkrt_psi_options.stash_size = kStashSize;
  kkrt_psi_options.cuckoo_hash_num = kCuckooHashNum;
  kkrt_psi_options.stat_sec_param = kStatSecParam;

  return kkrt_psi_options;
}

// first use baseOT get  128 ots
// then, use iknpOtExtension get 512 ots
void GetKkrtOtSenderOptions(
    const std::shared_ptr<yasl::link::Context>& link_ctx, const size_t num_ot,
    yasl::BaseRecvOptions* recv_opts) {
  YASL_ENFORCE(recv_opts != nullptr);
  size_t base_ot_num = 128;

  // use baseot get 128 ots
  yasl::BaseSendOptions iknp_send_opts;

  iknp_send_opts.blocks.resize(base_ot_num);

  yasl::BaseOtSend(link_ctx, absl::MakeSpan(iknp_send_opts.blocks));

  (*recv_opts).choices = yasl::CreateRandomChoices(num_ot);

  (*recv_opts).blocks.resize(num_ot);
  (*recv_opts).choices.resize(num_ot);

  std::vector<uint128_t> choices =
      yasl::CreateRandomChoiceBits<uint128_t>(num_ot);

  yasl::IknpOtExtRecv(link_ctx, iknp_send_opts, absl::MakeConstSpan(choices),
                      absl::MakeSpan((*recv_opts).blocks));

  for (size_t i = 0; i < num_ot; ++i) {
    (*recv_opts).choices[i] = GetBit(choices, i);
  }
}

void GetKkrtOtReceiverOptions(
    const std::shared_ptr<yasl::link::Context>& link_ctx, const size_t num_ot,
    yasl::BaseSendOptions* send_opts) {
  YASL_ENFORCE(send_opts != nullptr);
  size_t base_ot_num = 128;

  // use baseot get 128 ots
  yasl::BaseRecvOptions iknp_recv_opts;

  iknp_recv_opts.choices = yasl::CreateRandomChoices(base_ot_num);

  iknp_recv_opts.blocks.resize(base_ot_num);

  yasl::BaseOtRecv(link_ctx, iknp_recv_opts.choices,
                   absl::MakeSpan(iknp_recv_opts.blocks));

  (*send_opts).blocks.resize(num_ot);

  std::vector<uint128_t> choices =
      yasl::CreateRandomChoiceBits<uint128_t>(num_ot);

  IknpOtExtSend(link_ctx, iknp_recv_opts, absl::MakeSpan((*send_opts).blocks));
}

void KkrtPsiSend(const std::shared_ptr<yasl::link::Context>& link_ctx,
                 const KkrtPsiOptions& kkrt_psi_options,
                 const yasl::BaseRecvOptions& base_options,
                 const std::vector<uint128_t>& items_hash) {
  YASL_ENFORCE((kkrt_psi_options.cuckoo_hash_num == 3) &&
                   (kkrt_psi_options.stash_size == 0),
               "now only support cuckoo HashNum = 3 , stash size = 0");

  YASL_ENFORCE((base_options.blocks.size() == 512) &&
                   (base_options.choices.size() == 512),
               "now only support baseRecvOption block size 512");

  size_t peer_set_size = ExchangeSetSize(link_ctx, items_hash.size());

  YASL_ENFORCE((peer_set_size > 0) && (!items_hash.empty()),
               "item size need not zero, mine={}, peer={}", items_hash.size(),
               peer_set_size);

  auto ctx = CreateSenderRunContext(link_ctx, items_hash);
  ctx->peer_size = peer_set_size;

  CuckooIndex::Options option =
      CuckooIndex::SelectParams(ctx->peer_size, kkrt_psi_options.stash_size,
                                kkrt_psi_options.cuckoo_hash_num);

  ctx->kkrt_ot_num = option.NumBins();

  yasl::KkrtOtExtSender sender;

  sender.Init(link_ctx, base_options, ctx->kkrt_ot_num);

  SendOprfEncode(ctx, kkrt_psi_options, items_hash, sender);

  const char* finish_str = "kkrt finish";

  ctx->link0->SendAsync(ctx->link0->NextRank(), finish_str,
                        fmt::format("KKRT:PSI:Finished"));
}

std::vector<std::size_t> KkrtPsiRecv(
    const std::shared_ptr<yasl::link::Context>& link_ctx,
    const KkrtPsiOptions& kkrt_psi_options,
    const yasl::BaseSendOptions& base_options,
    const std::vector<uint128_t>& items_hash) {
  YASL_ENFORCE((kkrt_psi_options.cuckoo_hash_num == 3) &&
                   (kkrt_psi_options.stash_size == 0),
               "now only support cuckoo HashNum = 3 , stash size = 0");

  YASL_ENFORCE(base_options.blocks.size() == 512,
               "now only support yasl::BaseSendOptions block size 512");

  std::vector<std::size_t> ret_intersection;

  size_t peer_set_size = ExchangeSetSize(link_ctx, items_hash.size());

  YASL_ENFORCE((peer_set_size > 0) && (!items_hash.empty()),
               "item size need not zero, mine={}, peer={}", items_hash.size(),
               peer_set_size);

  CuckooIndex::Options option =
      CuckooIndex::SelectParams(items_hash.size(), kkrt_psi_options.stash_size,
                                kkrt_psi_options.cuckoo_hash_num);
  CuckooIndex cuckoo_index(option);

  auto ctx = CreateReceiverRunContext(link_ctx, items_hash, &cuckoo_index);

  ctx->peer_size = peer_set_size;

  std::vector<uint128_t> kkrt_ot_recv_encode(ctx->kkrt_ot_num);

  yasl::KkrtOtExtReceiver receiver;

  receiver.Init(link_ctx, base_options, ctx->kkrt_ot_num);

  receiver.SetBatchSize(kkrt_psi_options.ot_batch_size);
  uint64_t kkrt_ot_batch_size = receiver.GetBatchSize();

  std::array<std::unordered_map<std::string, size_t>, kCuckooHashNum>
      oprf_encode_map;
  for (size_t i = 0; i < kCuckooHashNum; i++) {
    oprf_encode_map[i].reserve(ctx->kkrt_ot_num);
  }

  uint64_t encode_size =
      KkrtEncodeSize(kkrt_psi_options.stat_sec_param, ctx->self_size,
                     ctx->peer_size);  // by byte

  auto ck_bins = cuckoo_index.bins();

  std::string encode_str(encode_size, '\0');

  const size_t ot_num_batch =
      (ctx->kkrt_ot_num + kkrt_ot_batch_size - 1) / kkrt_ot_batch_size;

  ThrottleControlSender throttle_control_sender(ctx->link0, ctx->link1,
                                                ot_num_batch);

  throttle_control_sender.StartRecvThread();

  for (size_t batch_idx = 0; batch_idx < ot_num_batch; ++batch_idx) {
    const size_t num_this_batch = std::min<size_t>(
        ctx->kkrt_ot_num - batch_idx * kkrt_ot_batch_size, kkrt_ot_batch_size);

    size_t batch_start = batch_idx * kkrt_ot_batch_size;
    for (size_t i = 0; i < num_this_batch; ++i) {
      size_t current_idx = batch_start + i;
      if (ck_bins[current_idx].IsEmpty()) {
        receiver.ZeroEncode(current_idx);
      } else {
        uint128_t input_item = items_hash[ck_bins[current_idx].InputIdx()];
        receiver.Encode(
            current_idx, input_item,
            absl::Span<uint8_t>(reinterpret_cast<uint8_t*>(&encode_str[0]),
                                encode_size));

        uint8_t min_hash_idx = cuckoo_index.MinCollidingHashIdx(current_idx);
        oprf_encode_map[min_hash_idx].emplace(encode_str,
                                              ck_bins[current_idx].InputIdx());
      }
    }

    const auto tag = fmt::format("KKRT_PSI:sendCorrection:{}", batch_idx);
    auto send_buf = receiver.ShiftCorrection(num_this_batch);
    // use throttle_control_sender
    throttle_control_sender.SendAsync(send_buf, tag);
  }
  throttle_control_sender.WaitRecvThread();

  size_t batch_count = 0;

  ThrottleControlReceiver throttle_control_receiver(ctx->link0, ctx->link1);
  while (true) {
    // Receive sendor prf encode.
    KkrtPsiBatch batch = KkrtPsiBatch::Deserialize(
        throttle_control_receiver.RecvMsgWithSendCounter());
    batch_count++;

    const bool is_last_batch = batch.is_last_batch;

    size_t curr_step_item_num = batch.item_num;
    size_t curr_step_encode_num =
        curr_step_item_num * kkrt_psi_options.cuckoo_hash_num;
    YASL_ENFORCE_EQ(batch.flatten_bytes.size(),
                    (curr_step_encode_num * encode_size));

    for (size_t i = 0; i < curr_step_item_num; ++i) {
      for (size_t j = 0; j < kkrt_psi_options.cuckoo_hash_num; ++j) {
        std::string encode_sub_str = batch.flatten_bytes.substr(
            (i * kkrt_psi_options.cuckoo_hash_num + j) * encode_size,
            encode_size);

        auto it = oprf_encode_map[j].find(encode_sub_str);
        if (it != oprf_encode_map[j].end()) {
          ret_intersection.emplace_back(it->second);
        }
      }
    }

    if (is_last_batch) {
      break;
    }
  }

  ctx->finished_batch_count = batch_count;
  ctx->link0->Recv(ctx->link0->NextRank(),
                   fmt::format("KKRT:PSI:Wait Sender Finished"));

  return ret_intersection;
}

void KkrtPsiSend(const std::shared_ptr<yasl::link::Context>& link_ctx,
                 const yasl::BaseRecvOptions& base_options,
                 const std::vector<uint128_t>& items_hash) {
  KkrtPsiOptions kkrt_psi_options = GetDefaultKkrtPsiOptions();

  return KkrtPsiSend(link_ctx, kkrt_psi_options, base_options, items_hash);
}

std::vector<std::size_t> KkrtPsiRecv(
    const std::shared_ptr<yasl::link::Context>& link_ctx,
    const yasl::BaseSendOptions& base_options,
    const std::vector<uint128_t>& items_hash) {
  KkrtPsiOptions kkrt_psi_options = GetDefaultKkrtPsiOptions();

  return KkrtPsiRecv(link_ctx, kkrt_psi_options, base_options, items_hash);
}

}  // namespace spu::psi
