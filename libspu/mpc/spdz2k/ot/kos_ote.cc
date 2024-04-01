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

#include "libspu/mpc/spdz2k/ot/kos_ote.h"

#include <random>

#include "emp-tool/utils/block.h"
#include "emp-tool/utils/f2k.h"
#include "yacl/crypto/hash/hash_utils.h"
#include "yacl/crypto/tools/crhash.h"
#include "yacl/crypto/tools/prg.h"
#include "yacl/crypto/tools/ro.h"
#include "yacl/crypto/tools/rp.h"
#include "yacl/link/link.h"
#include "yacl/utils/matrix_utils.h"
#include "yacl/utils/serialize.h"

#include "libspu/core/prelude.h"

// KOS have already implemented in sz/YACL
// However, TinyOT need an active secure delta-OT
namespace spu::mpc::spdz2k {

namespace {
constexpr size_t kKappa = 128;
constexpr size_t kS = 64;
constexpr size_t kBatchSize = 128;

// Binary Irreducible Polynomials
// ref: https://www.hpl.hp.com/techreports/98/HPL-98-135.pdf
constexpr uint64_t kGfMod64 = (1 << 4) | (1 << 3) | (1 << 1) | 1;

#define U128_LHF(Value) absl::Int128High64(static_cast<absl::int128>(Value))
#define U128_RHF(Value) absl::Int128Low64(static_cast<absl::int128>(Value))

inline uint64_t Block_Lhf(const emp::block& x) {
  return static_cast<uint64_t>(_mm_extract_epi64(x, 1));
}

inline uint64_t Block_Rhf(const emp::block& x) {
  return static_cast<uint64_t>(_mm_extract_epi64(x, 0));
}

inline uint128_t Mul64(uint64_t x, uint64_t y) {
  emp::block mblock = emp::zero_block;
  emp::block empty = emp::zero_block;
  emp::mul128(emp::makeBlock(0, x), emp::makeBlock(0, y), &mblock, &empty);
  return yacl::MakeUint128(Block_Lhf(mblock), Block_Rhf(mblock));
}

inline uint64_t Reduce64(uint128_t x) {
  emp::block xb = emp::makeBlock(U128_LHF(x), U128_RHF(x));
  emp::block tb = emp::zero_block;
  emp::block empty = emp::zero_block;  // useless
  // high 64 of xb
  emp::mul128(emp::makeBlock(0, U128_LHF(x)), emp::makeBlock(0, kGfMod64), &tb,
              &empty);
  xb ^= tb;
  // high 64 of mb
  emp::mul128(emp::makeBlock(0, Block_Lhf(tb)), emp::makeBlock(0, kGfMod64),
              &tb, &empty);
  xb ^= tb;
  return Block_Rhf(xb);
}

struct CheckMsg {
  uint64_t x;
  std::array<uint64_t, kKappa> t;

  // Constructor
  CheckMsg() {
    x = 0;
    std::fill(t.begin(), t.end(), 0);
  }

  yacl::Buffer Pack() {
    std::vector<uint64_t> res;
    res.push_back(x);
    res.insert(res.end(), t.begin(), t.end());
    return {res.data(), res.size() * sizeof(uint64_t)};
  }

  void Unpack(yacl::ByteContainerView buf) {
    std::memcpy(&x, buf.data(), sizeof(uint64_t));
    std::memcpy(t.data(), buf.data() + sizeof(uint64_t),
                kKappa * sizeof(uint64_t));
  }
};

inline uint64_t GenKosSharedSeed(
    const std::shared_ptr<yacl::link::Context>& ctx) {
  SPU_ENFORCE(ctx->WorldSize() == 2);
  if (ctx->Rank() == 0) {
    std::random_device rd;
    uint64_t seed = static_cast<uint64_t>(rd()) << 32 | rd();
    ctx->SendAsync(ctx->NextRank(),
                   yacl::ByteContainerView(&seed, sizeof(uint64_t)),
                   fmt::format("KOS-Seed"));
    return seed;
  } else {
    uint64_t seed = 0;
    auto buf = ctx->Recv(ctx->NextRank(), fmt::format("KOS-Seed"));
    std::memcpy(&seed, buf.data(), sizeof(uint64_t));
    return seed;
  }
}

inline auto ExtendBaseOt(
    const std::shared_ptr<yacl::crypto::OtSendStore>& base_ot,
    const size_t block_num) {
  std::array<std::vector<uint128_t>, kKappa> base_ot_ext0;
  std::array<std::vector<uint128_t>, kKappa> base_ot_ext1;
  for (size_t k = 0; k < base_ot->Size(); ++k) {
    base_ot_ext0[k] = std::vector<uint128_t>(block_num);
    base_ot_ext1[k] = std::vector<uint128_t>(block_num);
    yacl::crypto::Prg<uint128_t> prg0(base_ot->GetBlock(k, 0));
    yacl::crypto::Prg<uint128_t> prg1(base_ot->GetBlock(k, 1));

    prg0.Fill(absl::MakeSpan(base_ot_ext0[k]));
    prg1.Fill(absl::MakeSpan(base_ot_ext1[k]));
  }
  return std::make_pair(base_ot_ext0, base_ot_ext1);
}

inline auto ExtendBaseOt(
    const std::shared_ptr<yacl::crypto::OtRecvStore>& base_ot,
    const size_t block_num) {
  std::array<std::vector<uint128_t>, kKappa> base_ot_ext;
  for (size_t k = 0; k < base_ot->Size(); ++k) {
    base_ot_ext[k] = std::vector<uint128_t>(block_num);
    yacl::crypto::Prg<uint128_t> prg(base_ot->GetBlock(k));
    prg.Fill(absl::MakeSpan(base_ot_ext[k]));
  }
  return base_ot_ext;
}

inline auto ExtendChoice(const std::vector<bool>& choices,
                         const size_t final_size) {
  // Extend choices to batch_num * kBlockNum bits
  // 1st part (valid_ot_num bits): original ot choices
  // 2nd part (verify_ot_num bits): rand bits used for checking
  // 3rd party (the rest bits): padding 1;
  std::vector<bool> choices_ext = choices;

  // 2nd part Extension
  yacl::crypto::Prg<bool> gen;
  for (size_t i = 0; i < kS; i++) {
    choices_ext.push_back(gen());
  }
  // 3rd part Extension
  choices_ext.resize(final_size);
  return choices_ext;
}

}  // namespace

// KOS based delta-OTE
void KosOtExtSend(const std::shared_ptr<yacl::link::Context>& ctx,
                  const std::shared_ptr<yacl::crypto::OtRecvStore>& base_ot,
                  absl::Span<uint128_t> send_blocks, uint128_t& delta) {
  SPU_ENFORCE(ctx->WorldSize() == 2);
  SPU_ENFORCE(base_ot->Size() == kKappa);
  SPU_ENFORCE(!send_blocks.empty());
  SPU_ENFORCE(
      kS == 64,
      "Currently, KOS only support statistical security = 64 bit, but get {}",
      kS);

  const size_t ot_num_valid = send_blocks.size();
  const size_t ot_num_ext = ot_num_valid + kS;  // without batch padding
  const size_t batch_num = (ot_num_ext + kBatchSize - 1) / kBatchSize;
  const size_t block_num = batch_num * kBatchSize / 128;

  // Prepare for batched computation
  std::vector<uint128_t> q_ext(ot_num_ext);
  auto ot_ext = ExtendBaseOt(base_ot, block_num);

  // Prepare for consistency check
  std::array<uint64_t, kKappa> q_check;
  std::fill(q_check.begin(), q_check.end(), 0);
  auto seed = GenKosSharedSeed(ctx);

  auto rand_samples = std::vector<uint64_t>(batch_num * 2);
  yacl::crypto::Prg<uint64_t> prg(seed);
  prg.Fill(absl::MakeSpan(rand_samples));

  // Note the following is identical to the IKNP protocol without the final hash
  // code partially copied from yacl/crypto-primitives/ot/extension/kkrt_ote.cc
  // For every batch
  for (size_t i = 0; i < batch_num; ++i) {
    std::array<uint128_t, kBatchSize> recv_msg;
    const size_t offset = i * kBatchSize / 128;  // block offsets

    auto buf = ctx->Recv(ctx->NextRank(), fmt::format("KOS:{}", i));
    std::memcpy(recv_msg.data(), buf.data(), buf.size());

    // Q = (u & s) ^ G(K_s) = ((G(K_0) ^ G(K_1) ^ r)) & s) ^ G(K_s)
    // Q = G(K_0) when s is 0
    // Q = G(K_0) ^ r when s is 1
    // Hence we get the wanted behavior in IKNP, that is:
    //  s == 0, the sender receives T = G(K_0)
    //  s == 1, the sender receives U = G(K_0) ^ r = T ^ r
    for (size_t k = 0; k < kKappa; ++k) {
      const auto& ot_slice = ot_ext[k][offset];

      if (base_ot->GetChoice(k)) {
        recv_msg[k] ^= ot_slice;
      } else {
        recv_msg[k] = ot_slice;
      }

      // ******************* CONSISTENCY CHECK *******************
      // q_check[k] ^= U128_LHF(recv_msg[k]) & rand_samples[2 * i];
      // q_check[k] ^= U128_RHF(recv_msg[k]) & rand_samples[2 * i + 1];
      uint128_t ret = Mul64(U128_LHF(recv_msg[k]), rand_samples[2 * i]);
      ret ^= Mul64(U128_RHF(recv_msg[k]), rand_samples[2 * i + 1]);
      q_check[k] ^= Reduce64(ret);
      // ******************* CONSISTENCY CHECK *******************
    }

    // Transpose.
    yacl::SseTranspose128(&recv_msg);

    // Finalize (without crhash)
    const size_t limit = std::min(kBatchSize, ot_num_ext - i * kBatchSize);
    for (size_t j = 0; j < limit; ++j) {
      q_ext[i * kBatchSize + j] = recv_msg[j];
    }
  }

  delta = 0;
  for (size_t k = 0; k < 128; ++k) {
    if (base_ot->GetChoice(k)) {
      delta |= (uint128_t)1 << k;
    }
  }

  // ******************* CONSISTENCY CHECK *******************
  CheckMsg check_msgs;
  check_msgs.Unpack(ctx->Recv(ctx->NextRank(), fmt::format("KOS-CHECK")));

  for (size_t k = 0; k < kKappa; ++k) {
    uint128_t result = 0;
    if (base_ot->GetChoice(k)) {
      result = check_msgs.t[k] ^ (check_msgs.x);
    } else {
      result = check_msgs.t[k];
    }
    SPU_ENFORCE(result == q_check[k]);
  }
  // ******************* CONSISTENCY CHECK *******************

  q_ext.resize(ot_num_valid);
  for (size_t i = 0; i < ot_num_valid; i++) {
    send_blocks[i] = q_ext[i];
  }
}

// KOS based delta-OTE
void KosOtExtRecv(const std::shared_ptr<yacl::link::Context>& ctx,
                  const std::shared_ptr<yacl::crypto::OtSendStore>& base_ot,
                  const std::vector<bool>& choices,
                  absl::Span<uint128_t> recv_blocks) {
  SPU_ENFORCE(ctx->WorldSize() == 2);      // Check OT has two parties
  SPU_ENFORCE(base_ot->Size() == kKappa);  // Check base OT size
  SPU_ENFORCE(recv_blocks.size() == choices.size());
  SPU_ENFORCE(!recv_blocks.empty());
  SPU_ENFORCE(
      kS == 64,
      "Currently, KOS only support statistical security = 64 bit, but get {}",
      kS);

  const size_t ot_num_valid = recv_blocks.size();
  const size_t ot_num_ext = ot_num_valid + kS;  // without batch padding
  const size_t batch_num = (ot_num_ext + kBatchSize - 1) / kBatchSize;
  const size_t block_num = batch_num * kBatchSize / 128;

  // Prepare for batched computation
  std::vector<uint128_t> t_ext(ot_num_ext);
  auto choice_ext = ExtendChoice(choices, batch_num * kBatchSize);
  auto ot_ext = ExtendBaseOt(base_ot, block_num);

  // Prepare for consistency check
  CheckMsg check_msgs;
  auto seed = GenKosSharedSeed(ctx);

  auto rand_samples = std::vector<uint64_t>(batch_num * 2);
  yacl::crypto::Prg<uint64_t> prg(seed);
  prg.Fill(absl::MakeSpan(rand_samples));

  // Note the following is identical to the IKNP protocol without the final hash
  // code partially copied from yacl/crypto-primitives/ot/extension/kkrt_ote.cc
  // For a task of generating 129 OTs, we actually generates 128 * 2 = 256
  // OTs.
  for (size_t i = 0; i < batch_num; ++i) {
    std::array<uint128_t, kKappa> send_msg;
    std::array<uint128_t, kKappa> t;

    const size_t offset = i * kBatchSize / 128;  // block offsets
    // uint128_t choice_slice = *(choice_ext.data() + offset);
    uint128_t choice_slice = 0;
    for (size_t k = 0; k < kBatchSize; ++k) {
      if (choice_ext[i * kBatchSize + k]) {
        choice_slice |= (uint128_t)1 << k;
      }
    }

    // ******************* CONSISTENCY CHECK *******************
    // check_msgs.x ^= U128_LHF(choice_slice) & rand_samples[2 * i];
    // check_msgs.x ^= U128_RHF(choice_slice) & rand_samples[2 * i + 1];
    uint128_t ret = Mul64(U128_LHF(choice_slice), rand_samples[2 * i]);
    ret ^= Mul64(U128_RHF(choice_slice), rand_samples[2 * i + 1]);
    check_msgs.x ^= Reduce64(ret);
    // ******************* CONSISTENCY CHECK *******************

    for (size_t k = 0; k < kKappa; ++k) {
      const auto& ot_slice0 = ot_ext.first[k][offset];
      const auto& ot_slice1 = ot_ext.second[k][offset];
      send_msg[k] = ot_slice0 ^ ot_slice1 ^ choice_slice;
      t[k] = ot_slice0;

      // ******************* CONSISTENCY CHECK *******************
      // check_msgs.t[k] ^= U128_LHF(t[k]) & rand_samples[2 * i];
      // check_msgs.t[k] ^= U128_RHF(t[k]) & rand_samples[2 * i + 1];
      uint128_t ret = Mul64(U128_LHF(t[k]), rand_samples[2 * i]);
      ret ^= Mul64(U128_RHF(t[k]), rand_samples[2 * i + 1]);
      check_msgs.t[k] ^= Reduce64(ret);
      // ******************* CONSISTENCY CHECK *******************
    }

    ctx->SendAsync(ctx->NextRank(),
                   yacl::ByteContainerView(send_msg.data(),
                                           send_msg.size() * sizeof(uint128_t)),
                   fmt::format("KOS:{}", i));

    // Transpose.
    yacl::SseTranspose128(&t);

    // Finalize (without crhash)
    const size_t limit = std::min(kBatchSize, ot_num_ext - i * kBatchSize);
    for (size_t j = 0; j < limit; ++j) {
      t_ext[i * kBatchSize + j] = t[j];
    }
  }

  // ******************* CONSISTENCY CHECK *******************
  // check_msgs.Print();
  auto buf = check_msgs.Pack();
  ctx->SendAsync(ctx->NextRank(), buf, fmt::format("KOS-CHECK"));
  // ******************* CONSISTENCY CHECK *******************

  t_ext.resize(ot_num_valid);

  for (size_t i = 0; i < ot_num_valid; ++i) {
    recv_blocks[i] = t_ext[i];
  }
}
// END of KOS
};  // namespace spu::mpc::spdz2k