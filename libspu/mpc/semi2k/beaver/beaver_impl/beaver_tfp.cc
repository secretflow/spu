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

#include "libspu/mpc/semi2k/beaver/beaver_impl/beaver_tfp.h"

#include <algorithm>
#include <utility>

#include "yacl/crypto/rand/rand.h"
#include "yacl/link/algorithm/gather.h"
#include "yacl/utils/serialize.h"

#include "libspu/mpc/common/prg_tensor.h"
#include "libspu/mpc/semi2k/beaver/beaver_impl/trusted_party/trusted_party.h"
#include "libspu/mpc/utils/gfmp_ops.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::semi2k {

namespace {

inline size_t CeilDiv(size_t a, size_t b) { return (a + b - 1) / b; }

void FillReplayDesc(Beaver::ReplayDesc* desc, FieldType field, int64_t size,
                    const std::vector<Beaver::PrgSeedBuff>& encrypted_seeds,
                    PrgCounter counter, PrgSeed self_seed,
                    ElementType eltype = ElementType::kRing) {
  if (desc == nullptr || desc->status != Beaver::Init) {
    return;
  }
  desc->size = size;
  desc->field = field;
  desc->prg_counter = counter;
  desc->encrypted_seeds = encrypted_seeds;
  desc->seed = self_seed;
  desc->eltype = eltype;
}

}  // namespace

BeaverTfpUnsafe::BeaverTfpUnsafe(std::shared_ptr<yacl::link::Context> lctx)
    : lctx_(std::move(lctx)),
      seed_(yacl::crypto::SecureRandSeed()),
      counter_(0) {
  auto buf = yacl::SerializeUint128(seed_);
  std::vector<yacl::Buffer> all_bufs =
      yacl::link::Gather(lctx_, buf, 0, "BEAVER_TFP:SYNC_SEEDS");

  if (lctx_->Rank() == 0) {
    // Collects seeds from all parties.
    for (size_t rank = 0; rank < lctx_->WorldSize(); ++rank) {
      PrgSeed seed = yacl::DeserializeUint128(all_bufs[rank]);
      seeds_.push_back(seed);
      seeds_buff_.emplace_back(reinterpret_cast<void*>(&seed), sizeof(PrgSeed));
    }
  }
}

BeaverTfpUnsafe::Triple BeaverTfpUnsafe::Mul(FieldType field, int64_t size,
                                             ReplayDesc* x_desc,
                                             ReplayDesc* y_desc,
                                             ElementType eltype) {
  std::vector<TrustedParty::Operand> ops(3);
  Shape shape({size, 1});
  std::vector<std::vector<PrgSeed>> replay_seeds(3);

  auto if_replay = [&](const ReplayDesc* replay_desc, size_t idx) {
    if (replay_desc == nullptr || replay_desc->status != Beaver::Replay) {
      ops[idx].seeds = seeds_;
      // enforce the eltypes in ops
      ops[idx].desc.eltype = eltype;
      return prgCreateArray(field, shape, seed_, &counter_, &ops[idx].desc,
                            eltype);
    } else {
      SPU_ENFORCE(replay_desc->field == field);
      SPU_ENFORCE(replay_desc->eltype == eltype);
      SPU_ENFORCE(replay_desc->size == size);
      if (lctx_->Rank() == 0) {
        SPU_ENFORCE(replay_desc->encrypted_seeds.size() == lctx_->WorldSize());
        replay_seeds[idx].resize(replay_desc->encrypted_seeds.size());
        for (size_t i = 0; i < replay_seeds[idx].size(); i++) {
          SPU_ENFORCE(replay_desc->encrypted_seeds[i].size() ==
                      sizeof(PrgSeed));
          std::memcpy(&replay_seeds[idx][i],
                      replay_desc->encrypted_seeds[i].data(), sizeof(PrgSeed));
        }
        ops[idx].seeds = replay_seeds[idx];
        ops[idx].desc.field = field;
        ops[idx].desc.eltype = eltype;
        ops[idx].desc.shape = shape;
        ops[idx].desc.prg_counter = replay_desc->prg_counter;
      }
      PrgCounter tmp_counter = replay_desc->prg_counter;
      return prgCreateArray(field, shape, replay_desc->seed, &tmp_counter,
                            nullptr, eltype);
    }
  };

  FillReplayDesc(x_desc, field, size, seeds_buff_, counter_, seed_, eltype);
  auto a = if_replay(x_desc, 0);
  FillReplayDesc(y_desc, field, size, seeds_buff_, counter_, seed_, eltype);
  auto b = if_replay(y_desc, 1);
  auto c = prgCreateArray(field, shape, seed_, &counter_, &ops[2].desc, eltype);

  if (lctx_->Rank() == 0) {
    ops[2].seeds = seeds_;
    auto adjust = TrustedParty::adjustMul(absl::MakeSpan(ops));
    if (eltype == ElementType::kGfmp) {
      auto T = c.eltype();
      gfmp_add_mod_(c, adjust.as(T));
    } else {
      ring_add_(c, adjust);
    }
  }

  Triple ret;
  std::get<0>(ret) = std::move(*a.buf());
  std::get<1>(ret) = std::move(*b.buf());
  std::get<2>(ret) = std::move(*c.buf());

  return ret;
}

BeaverTfpUnsafe::Pair BeaverTfpUnsafe::MulPriv(FieldType field, int64_t size,
                                               ElementType eltype) {
  std::vector<TrustedParty::Operand> ops(2);
  Shape shape({size, 1});

  ops[0].seeds = seeds_;
  // enforce the eltypes in ops
  ops[0].desc.eltype = eltype;
  ops[1].desc.eltype = eltype;
  auto a_or_b =
      prgCreateArray(field, shape, seed_, &counter_, &ops[0].desc, eltype);
  auto c = prgCreateArray(field, shape, seed_, &counter_, &ops[1].desc, eltype);

  if (lctx_->Rank() == 0) {
    ops[1].seeds = seeds_;
    auto adjust = TrustedParty::adjustMulPriv(absl::MakeSpan(ops));
    if (eltype == ElementType::kGfmp) {
      auto T = c.eltype();
      gfmp_add_mod_(c, adjust.as(T));
    } else {
      ring_add_(c, adjust);
    }
  }

  Pair ret;
  std::get<0>(ret) = std::move(*a_or_b.buf());
  std::get<1>(ret) = std::move(*c.buf());

  return ret;
}

BeaverTfpUnsafe::Pair BeaverTfpUnsafe::Square(FieldType field, int64_t size,
                                              ReplayDesc* x_desc) {
  std::vector<TrustedParty::Operand> ops(2);
  Shape shape({size, 1});
  std::vector<std::vector<PrgSeed>> replay_seeds(2);

  auto if_replay = [&](const ReplayDesc* replay_desc, size_t idx) {
    if (replay_desc == nullptr || replay_desc->status != Beaver::Replay) {
      ops[idx].seeds = seeds_;
      return prgCreateArray(field, shape, seed_, &counter_, &ops[idx].desc);
    } else {
      SPU_ENFORCE(replay_desc->field == field);
      SPU_ENFORCE(replay_desc->size == size);
      if (lctx_->Rank() == 0) {
        SPU_ENFORCE(replay_desc->encrypted_seeds.size() == lctx_->WorldSize());
        replay_seeds[idx].resize(replay_desc->encrypted_seeds.size());
        for (size_t i = 0; i < replay_seeds[idx].size(); i++) {
          SPU_ENFORCE(replay_desc->encrypted_seeds[i].size() ==
                      sizeof(PrgSeed));
          std::memcpy(&replay_seeds[idx][i],
                      replay_desc->encrypted_seeds[i].data(), sizeof(PrgSeed));
        }
        ops[idx].seeds = replay_seeds[idx];
        ops[idx].desc.field = field;
        ops[idx].desc.shape = shape;
        ops[idx].desc.prg_counter = replay_desc->prg_counter;
      }
      PrgCounter tmp_counter = replay_desc->prg_counter;
      return prgCreateArray(field, shape, replay_desc->seed, &tmp_counter,
                            nullptr);
    }
  };

  FillReplayDesc(x_desc, field, size, seeds_buff_, counter_, seed_);
  auto a = if_replay(x_desc, 0);
  auto b = prgCreateArray(field, shape, seed_, &counter_, &ops[1].desc);

  if (lctx_->Rank() == 0) {
    ops[1].seeds = seeds_;
    auto adjust = TrustedParty::adjustSquare(absl::MakeSpan(ops));
    ring_add_(b, adjust);
  }

  Pair ret;
  std::get<0>(ret) = std::move(*a.buf());
  std::get<1>(ret) = std::move(*b.buf());

  return ret;
}

BeaverTfpUnsafe::Triple BeaverTfpUnsafe::Dot(FieldType field, int64_t m,
                                             int64_t n, int64_t k,
                                             ReplayDesc* x_desc,
                                             ReplayDesc* y_desc) {
  std::vector<TrustedParty::Operand> ops(3);
  std::vector<std::vector<PrgSeed>> replay_seeds(3);
  std::vector<Shape> shapes(3);
  shapes[0] = {m, k};
  shapes[1] = {k, n};
  shapes[2] = {m, n};

  auto if_replay = [&](const ReplayDesc* replay_desc, size_t idx) {
    if (replay_desc == nullptr) {
      ops[idx].seeds = seeds_;
      return prgCreateArray(field, shapes[idx], seed_, &counter_,
                            &ops[idx].desc);
    } else {
      SPU_ENFORCE(replay_desc->field == field);
      if (replay_desc->status == Beaver::TransposeReplay) {
        std::reverse(shapes[idx].begin(), shapes[idx].end());
      }
      if (lctx_->Rank() == 0) {
        SPU_ENFORCE(replay_desc->encrypted_seeds.size() == lctx_->WorldSize());
        replay_seeds[idx].resize(replay_desc->encrypted_seeds.size());
        for (size_t i = 0; i < replay_seeds[idx].size(); i++) {
          SPU_ENFORCE(replay_desc->encrypted_seeds[i].size() ==
                      sizeof(PrgSeed));
          std::memcpy(&replay_seeds[idx][i],
                      replay_desc->encrypted_seeds[i].data(), sizeof(PrgSeed));
        }
        ops[idx].seeds = replay_seeds[idx];
        ops[idx].desc.field = field;
        ops[idx].desc.shape = shapes[idx];
        ops[idx].desc.prg_counter = replay_desc->prg_counter;
        ops[idx].transpose = replay_desc->status == Beaver::TransposeReplay;
      }
      PrgCounter tmp_counter = replay_desc->prg_counter;
      auto ret = prgCreateArray(field, shapes[idx], replay_desc->seed,
                                &tmp_counter, nullptr);
      if (replay_desc->status == Beaver::TransposeReplay) {
        ret = ret.transpose().clone();
      }
      return ret;
    }
  };

  FillReplayDesc(x_desc, field, m * k, seeds_buff_, counter_, seed_);
  auto a = if_replay(x_desc, 0);
  FillReplayDesc(y_desc, field, n * k, seeds_buff_, counter_, seed_);
  auto b = if_replay(y_desc, 1);
  auto c = prgCreateArray(field, {m, n}, seed_, &counter_, &ops[2].desc);

  if (lctx_->Rank() == 0) {
    ops[2].seeds = seeds_;
    auto adjust = TrustedParty::adjustDot(absl::MakeSpan(ops));
    ring_add_(c, adjust);
  }

  Triple ret;
  std::get<0>(ret) = std::move(*a.buf());
  std::get<1>(ret) = std::move(*b.buf());
  std::get<2>(ret) = std::move(*c.buf());

  return ret;
}

BeaverTfpUnsafe::Triple BeaverTfpUnsafe::And(int64_t size) {
  std::vector<TrustedParty::Operand> ops(3);
  // inside beaver, use max field for efficiency
  auto field = FieldType::FM128;
  int64_t elsize = CeilDiv(size, SizeOf(field));
  Shape shape({elsize, 1});

  auto a = prgCreateArray(field, shape, seed_, &counter_, &ops[0].desc);
  auto b = prgCreateArray(field, shape, seed_, &counter_, &ops[1].desc);
  auto c = prgCreateArray(field, shape, seed_, &counter_, &ops[2].desc);

  if (lctx_->Rank() == 0) {
    for (auto& op : ops) {
      op.seeds = seeds_;
    }
    auto adjust = TrustedParty::adjustAnd(absl::MakeSpan(ops));
    ring_xor_(c, adjust);
  }

  Triple ret;
  std::get<0>(ret) = std::move(*a.buf());
  std::get<1>(ret) = std::move(*b.buf());
  std::get<2>(ret) = std::move(*c.buf());
  std::get<0>(ret).resize(size);
  std::get<1>(ret).resize(size);
  std::get<2>(ret).resize(size);

  return ret;
}

BeaverTfpUnsafe::Pair BeaverTfpUnsafe::Trunc(FieldType field, int64_t size,
                                             size_t bits) {
  std::vector<TrustedParty::Operand> ops(2);
  Shape shape({size, 1});

  auto a = prgCreateArray(field, shape, seed_, &counter_, &ops[0].desc);
  auto b = prgCreateArray(field, shape, seed_, &counter_, &ops[1].desc);
  if (lctx_->Rank() == 0) {
    for (auto& op : ops) {
      op.seeds = seeds_;
    }
    auto adjust = TrustedParty::adjustTrunc(absl::MakeSpan(ops), bits);
    ring_add_(b, adjust);
  }

  Pair ret;
  ret.first = std::move(*a.buf());
  ret.second = std::move(*b.buf());

  return ret;
}

BeaverTfpUnsafe::Triple BeaverTfpUnsafe::TruncPr(FieldType field, int64_t size,
                                                 size_t bits) {
  std::vector<TrustedParty::Operand> ops(3);
  Shape shape({size, 1});

  auto r = prgCreateArray(field, shape, seed_, &counter_, &ops[0].desc);
  auto rc = prgCreateArray(field, shape, seed_, &counter_, &ops[1].desc);
  auto rb = prgCreateArray(field, shape, seed_, &counter_, &ops[2].desc);

  if (lctx_->Rank() == 0) {
    for (auto& op : ops) {
      op.seeds = seeds_;
    }
    auto adjusts = TrustedParty::adjustTruncPr(absl::MakeSpan(ops), bits);
    ring_add_(rc, std::get<0>(adjusts));
    ring_add_(rb, std::get<1>(adjusts));
  }

  Triple ret;
  std::get<0>(ret) = std::move(*r.buf());
  std::get<1>(ret) = std::move(*rc.buf());
  std::get<2>(ret) = std::move(*rb.buf());

  return ret;
}

BeaverTfpUnsafe::Array BeaverTfpUnsafe::RandBit(FieldType field, int64_t size) {
  std::vector<TrustedParty::Operand> ops(1);
  Shape shape({size, 1});

  auto a = prgCreateArray(field, shape, seed_, &counter_, &ops[0].desc);
  if (lctx_->Rank() == 0) {
    for (auto& op : ops) {
      op.seeds = seeds_;
    }
    auto adjust = TrustedParty::adjustRandBit(absl::MakeSpan(ops));
    ring_add_(a, adjust);
  }

  return std::move(*a.buf());
}

BeaverTfpUnsafe::Pair BeaverTfpUnsafe::PermPair(
    FieldType field, int64_t size, size_t perm_rank,
    absl::Span<const int64_t> perm_vec) {
  constexpr char kTag[] = "BEAVER_TFP:PERM";

  std::vector<TrustedParty::Operand> ops(2);
  Shape shape({size});

  auto a = prgCreateArray(field, shape, seed_, &counter_, &ops[0].desc);
  auto b = prgCreateArray(field, shape, seed_, &counter_, &ops[1].desc);

  if (lctx_->Rank() == 0) {
    for (auto& op : ops) {
      op.seeds = seeds_;
    }
    if (perm_rank != lctx_->Rank()) {
      auto pv_buf = lctx_->Recv(perm_rank, kTag);

      ring_add_(b, TrustedParty::adjustPerm(
                       absl::MakeSpan(ops),
                       absl::MakeSpan(pv_buf.data<int64_t>(),
                                      pv_buf.size() / sizeof(int64_t))));
    } else {
      ring_add_(b, TrustedParty::adjustPerm(absl::MakeSpan(ops), perm_vec));
    }
  } else if (perm_rank == lctx_->Rank()) {
    lctx_->SendAsync(
        0, yacl::Buffer(perm_vec.data(), perm_vec.size() * sizeof(int64_t)),
        kTag);
  }

  Pair ret;
  ret.first = std::move(*a.buf());
  ret.second = std::move(*b.buf());

  return ret;
}

std::unique_ptr<Beaver> BeaverTfpUnsafe::Spawn() {
  return std::make_unique<BeaverTfpUnsafe>(lctx_->Spawn());
}

BeaverTfpUnsafe::Pair BeaverTfpUnsafe::Eqz(FieldType field, int64_t size) {
  std::vector<TrustedParty::Operand> ops(2);
  Shape shape({size, 1});

  auto a = prgCreateArray(field, shape, seed_, &counter_, &ops[0].desc);
  auto b = prgCreateArray(field, shape, seed_, &counter_, &ops[1].desc);
  if (lctx_->Rank() == 0) {
    for (auto& op : ops) {
      op.seeds = seeds_;
    }
    auto adjust = TrustedParty::adjustEqz(absl::MakeSpan(ops));
    ring_xor_(b, adjust);
  }

  Pair ret;
  ret.first = std::move(*a.buf());
  ret.second = std::move(*b.buf());

  return ret;
}

}  // namespace spu::mpc::semi2k
