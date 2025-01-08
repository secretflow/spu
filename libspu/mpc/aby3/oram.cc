// Copyright 2024 Ant Group Co., Ltd.
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

#include "libspu/mpc/aby3/oram.h"

#include <future>

#include "yacl/crypto/rand/rand.h"

#include "libspu/mpc/ab_api.h"
#include "libspu/mpc/aby3/type.h"
#include "libspu/mpc/aby3/value.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::aby3 {

// [TODO] divide into blocks when s is large
// generate 3 * 2pc-dpf, e0 e1 e2
// p0 holds(e01, e10), p1 holds(e11, e20), p2 holds(e21, e00)
NdArrayRef OramOneHotAA::proc(KernelEvalContext *ctx, const NdArrayRef &in,
                              int64_t s) const {
  auto *comm = ctx->getState<Communicator>();
  const auto eltype = in.eltype();
  const auto field = eltype.as<AShrTy>()->field();
  NdArrayRef out(makeType<OShrTy>(field), {s});

  DISPATCH_ALL_FIELDS(field, [&]() {
    using el_t = ring2k_t;
    using shr_t = std::array<el_t, 2>;
    NdArrayView<shr_t> out_(out);

    auto in_b = UnwrapValue(a2b(ctx->sctx(), WrapValue(in)));
    NdArrayView<shr_t> target_idxs_(in_b);

    // generate aeskey for dpf
    auto [self_aes_keys, next_aes_keys] = oram::genAesKey(ctx, 1);

    auto octx = oram::OramContext<el_t>(s);

    for (int64_t j = 0; j < 3; j++) {
      // in round (rank - 1), as helper
      if ((j + 1) % 3 == static_cast<int64_t>(comm->getRank())) {
        // beaver for dpf gen
        oram::genOramBeaverHelper<oram::DpfKeyT>(ctx, Log2Ceil(s) * 2,
                                                 oram::OpKind::And);
        // beaver for B2A convert
        oram::genOramBeaverHelper<el_t>(ctx, 1, oram::OpKind::Mul);
      } else {
        auto dpf_rank = comm->getRank() == static_cast<size_t>(j);
        auto aes_key = dpf_rank ? self_aes_keys[0] : next_aes_keys[0];
        auto target_point = dpf_rank ? target_idxs_[0][0] ^ target_idxs_[0][1]
                                     : target_idxs_[0][0];
        // dpf gen
        octx.genDpf(ctx, static_cast<oram::DpfGenCtrl>(j), aes_key,
                    target_point);
        // B2A
        octx.onehotB2A(ctx, static_cast<oram::DpfGenCtrl>(j));
      }
    }

    pforeach(0, s, [&](int64_t k) {
      for (int64_t j = 0; j < 2; j++) {
        out_[k][j] = octx.dpf_e[j][k];
      }
    });
  });

  return out;
};

// generate 1 * 2pc-dpf, e
// p0 holds(e0), p1 holds(e1)
NdArrayRef OramOneHotAP::proc(KernelEvalContext *ctx, const NdArrayRef &in,
                              int64_t s) const {
  auto *comm = ctx->getState<Communicator>();
  const auto eltype = in.eltype();
  const auto field = eltype.as<AShrTy>()->field();
  const auto numel = in.numel();
  NdArrayRef out(makeType<OPShrTy>(field), {s});

  DISPATCH_ALL_FIELDS(field, [&]() {
    using el_t = ring2k_t;
    using shr_t = std::array<el_t, 2>;
    NdArrayView<el_t> out_(out);

    auto in_b = UnwrapValue(a2b(ctx->sctx(), WrapValue(in)));

    if (comm->getRank() == 2) {
      oram::genOramBeaverHelper<oram::DpfKeyT>(ctx, Log2Ceil(s) * 2,
                                               oram::OpKind::And);
      oram::genOramBeaverHelper<el_t>(ctx, 1, oram::OpKind::Mul);
    } else {
      auto dst_rank = comm->getRank() == 0 ? 1 : 0;
      // 3->2
      NdArrayView<shr_t> in_(in_b);
      NdArrayRef target_point_2pc(makeType<RingTy>(field), in.shape());
      NdArrayView<el_t> target_point_2pc_(target_point_2pc);
      // reblind
      if (comm->getRank() == 0) {
        pforeach(0, numel,
                 [&](int64_t idx) { target_point_2pc_[idx] = in_[idx][0]; });
      } else {
        pforeach(0, numel, [&](int64_t idx) {
          target_point_2pc_[idx] = in_[idx][0] ^ in_[idx][1];
        });
      }

      // generate aeskey for dpf
      auto aes_key = yacl::crypto::SecureRandSeed();

      comm->sendAsync<uint128_t>(dst_rank, {aes_key}, "aes_key");
      aes_key += comm->recv<uint128_t>(dst_rank, "aes_key")[0];

      auto octx = oram::OramContext<el_t>(s);

      // dpf gen
      octx.genDpf(ctx, static_cast<oram::DpfGenCtrl>(1), aes_key,
                  target_point_2pc_[0]);
      // B2A
      octx.onehotB2A(ctx, static_cast<oram::DpfGenCtrl>(1));

      int64_t j = comm->getRank() == 0 ? 1 : 0;
      pforeach(0, s, [&](int64_t k) { out_[k] = octx.dpf_e[j][k]; });
    }
  });

  return out;
};

NdArrayRef OramReadOA::proc(KernelEvalContext *ctx, const NdArrayRef &onehot,
                            const NdArrayRef &db, int64_t offset) const {
  auto *comm = ctx->getState<Communicator>();
  auto *prg = ctx->getState<PrgState>();

  const auto field = db.eltype().as<AShrTy>()->field();
  int64_t index_times = db.shape()[1];
  int64_t db_numel = onehot.numel();

  NdArrayRef out(makeType<AShrTy>(field), {1, index_times});

  DISPATCH_ALL_FIELDS(field, [&]() {
    using el_t = ring2k_t;
    using shr_t = std::array<el_t, 2>;

    auto r = std::async([&] {
      auto [r0, r1] = prg->genPrssPair(field, {1, index_times},
                                       PrgState::GenPrssCtrl::Both);
      return ring_sub(r0, r1);
    });

    NdArrayView<shr_t> onehot_(onehot);

    NdArrayRef shifted_onehot(makeType<OShrTy>(field), onehot.shape());
    NdArrayView<shr_t> shifted_onehot_(shifted_onehot);
    if (offset != 0) {
      pforeach(0, db_numel, [&](int64_t idx) {
        shifted_onehot_[idx] = onehot_[(idx - offset + db_numel) % db_numel];
      });
    } else {
      shifted_onehot = onehot;
    }

    // [TODO]: accelerate matmul with GPU
    auto db0 = getFirstShare(db);
    auto db1 = getSecondShare(db);
    auto onehot0 = getFirstShare(shifted_onehot);
    auto onehot1 = getSecondShare(shifted_onehot);

    auto res0 = std::async(ring_mmul, onehot0, db0);
    auto res1 = ring_mmul(onehot1, db1);

    auto o1 = getFirstShare(out);
    auto o2 = getSecondShare(out);
    auto z1 = ring_sum({res0.get(), res1, r.get()});

    auto f = std::async([&] { ring_assign(o1, z1); });
    // reshare
    ring_assign(o2, comm->rotate(z1, kBindName()));
    f.get();
  });

  return out;
}

NdArrayRef OramReadOP::proc(KernelEvalContext *ctx, const NdArrayRef &onehot,
                            const NdArrayRef &db, int64_t offset) const {
  auto *comm = ctx->getState<Communicator>();
  auto *prg = ctx->getState<PrgState>();
  const auto field = onehot.eltype().as<OPShrTy>()->field();
  int64_t index_times = 1;
  if (db.shape().size() == 2) {
    index_times = db.shape()[1];
  }
  NdArrayRef out(makeType<AShrTy>(field), {1, index_times});
  auto o1 = getFirstShare(out);
  auto o2 = getSecondShare(out);
  int64_t db_numel = onehot.numel();

  DISPATCH_ALL_FIELDS(field, [&]() {
    using el_t = ring2k_t;
    using shr_t = std::array<el_t, 2>;

    NdArrayView<shr_t> out_(out);
    NdArrayRef out2pc(makeType<RingTy>(field), {1, index_times});
    NdArrayView<el_t> out2pc_(out2pc);

    auto r = std::async([&] {
      auto [r0, r1] = prg->genPrssPair(field, {1, index_times},
                                       PrgState::GenPrssCtrl::Both);
      return ring_sub(r0, r1);
    });

    if (comm->getRank() == 2) {
      pforeach(0, index_times, [&](int64_t idx) { out2pc_[idx] = 0; });
    } else {
      NdArrayView<el_t> onehot_(onehot);
      NdArrayRef shifted_onehot(makeType<OPShrTy>(field), onehot.shape());
      NdArrayView<el_t> shifted_onehot_(shifted_onehot);
      if (offset != 0) {
        pforeach(0, db_numel, [&](int64_t idx) {
          shifted_onehot_[idx] = onehot_[(idx - offset + db_numel) % db_numel];
        });
      } else {
        shifted_onehot = onehot;
      }

      // [TODO]: accelerate matmul with GPU
      out2pc = ring_mmul(shifted_onehot, db);
    }

    ring_add_(out2pc, r.get());

    auto f = std::async([&] { ring_assign(o1, out2pc); });
    ring_assign(o2, comm->rotate(out2pc, kBindName()));
    f.get();
  });

  return out;
}
}  // namespace spu::mpc::aby3

namespace spu::mpc::oram {

template <typename T>
using Triple = std::tuple<T, T, T>;

// set lsb of x target_bit
uint128_t setLsb(uint128_t x, uint8_t target_bit) {
  target_bit &= 1;
  x = (x & (static_cast<uint128_t>(-1) - 1)) | target_bit;
  return x;
};

uint8_t getLsb(uint128_t x) { return static_cast<uint8_t>(x & 1); };

// decompose x into bits
// each bit of x is uint128 with all 1/ all 0
std::vector<DpfKeyT> bitDecomposeToDpfKeyT(uint128_t x, int64_t max_num) {
  std::vector<DpfKeyT> res(max_num);
  for (int i = 0; i < max_num; i++) {
    auto temp = (x >> i) & 1;
    // set all bits 0 / 1
    res[i] = temp == 0 ? 0 : -1;
  }
  std::reverse(res.begin(), res.end());
  return res;
};

template <typename T>
Triple<std::vector<T>> genOramBeaverPrim(KernelEvalContext *ctx, int64_t num,
                                         OpKind op, size_t adjust_rank) {
  auto *comm = ctx->getState<Communicator>();
  auto *prg = ctx->getState<PrgState>();
  std::vector<T> beaver_triple(num * 3);

  if (comm->getRank() == adjust_rank) {
    prg->fillPrssPair<T>(nullptr, beaver_triple.data(), num * 3,
                         PrgState::GenPrssCtrl::Second);
  } else {
    prg->fillPrssPair<T>(beaver_triple.data(), nullptr, num * 3,
                         PrgState::GenPrssCtrl::First);
  }

  std::vector<T> a(beaver_triple.begin(), beaver_triple.begin() + num);
  std::vector<T> b(beaver_triple.begin() + num,
                   beaver_triple.begin() + num * 2);
  std::vector<T> c(beaver_triple.begin() + num * 2, beaver_triple.end());

  // adjust
  if (comm->getRank() == adjust_rank) {
    auto adjust_c = comm->recv<T>(comm->nextRank(), "adjusted_c");
    if (op == OpKind::And) {
      pforeach(0, num, [&](int64_t i) { c[i] ^= adjust_c[i]; });
    } else {
      pforeach(0, num, [&](int64_t i) { c[i] += adjust_c[i]; });
    }
  }

  return std::make_tuple(a, b, c);
};

template <typename T>
void genOramBeaverHelper(KernelEvalContext *ctx, int64_t num, OpKind op) {
  auto *comm = ctx->getState<Communicator>();
  auto *prg = ctx->getState<PrgState>();
  size_t adjust_rank = comm->prevRank();

  // beaver_triple = [a, b, c]
  std::vector<T> beaver_triple0(num * 3);
  std::vector<T> beaver_triple1(num * 3);
  prg->fillPrssPair<T>(nullptr, beaver_triple0.data(), num * 3,
                       PrgState::GenPrssCtrl::Second);
  prg->fillPrssPair<T>(beaver_triple1.data(), nullptr, num * 3,
                       PrgState::GenPrssCtrl::First);

  // adjust c
  if (op == OpKind::And) {
    pforeach(0, num, [&](int64_t idx) {
      beaver_triple0[num * 2 + idx] =
          ((beaver_triple0[idx] ^ beaver_triple1[idx]) &
           (beaver_triple0[num + idx] ^ beaver_triple1[num + idx])) ^
          (beaver_triple0[num * 2 + idx] ^ beaver_triple1[num * 2 + idx]);
    });
  } else {
    pforeach(0, num, [&](int64_t idx) {
      beaver_triple0[num * 2 + idx] =
          ((beaver_triple0[idx] + beaver_triple1[idx]) *
           (beaver_triple0[num + idx] + beaver_triple1[num + idx])) -
          (beaver_triple0[num * 2 + idx] + beaver_triple1[num * 2 + idx]);
    });
  }
  std::vector<T> adjusted_c(beaver_triple0.begin() + num * 2,
                            beaver_triple0.end());

  comm->sendAsync<T>(adjust_rank, absl::MakeSpan(adjusted_c), "adjusted_c");
};

DpfKeyT computecw(KernelEvalContext *ctx, DpfKeyT target_bit, DpfKeyT suml,
                  DpfKeyT sumr, std::array<DpfKeyT, 3> &oram_and_beaver_l,
                  std::array<DpfKeyT, 3> &oram_and_beaver_r, DpfGenCtrl ctrl) {
  auto *comm = ctx->getState<Communicator>();
  auto dpf_rank = comm->getRank() == static_cast<size_t>(ctrl);
  size_t dst_rank = dpf_rank ? comm->prevRank() : comm->nextRank();

  std::array<DpfKeyT, 4> mask;
  mask[0] = target_bit ^ oram_and_beaver_l[0];
  mask[1] = suml ^ oram_and_beaver_l[1];
  mask[2] = dpf_rank ? target_bit ^ -1 ^ oram_and_beaver_r[0]
                     : target_bit ^ oram_and_beaver_r[0];
  mask[3] = sumr ^ oram_and_beaver_r[1];

  comm->sendAsync<DpfKeyT>(dst_rank, absl::MakeSpan(mask), "open(x^a,y^b)");
  auto temp = comm->recv<DpfKeyT>(dst_rank, "open(x^a,y^b)");
  for (uint64_t i = 0; i < mask.size(); i++) {
    mask[i] ^= temp[i];
  }

  std::vector<DpfKeyT> z(2);
  z[0] = oram_and_beaver_l[2];
  z[0] ^= mask[0] & oram_and_beaver_l[1];
  z[0] ^= mask[1] & oram_and_beaver_l[0];

  z[1] = oram_and_beaver_r[2];
  z[1] ^= mask[2] & oram_and_beaver_r[1];
  z[1] ^= mask[3] & oram_and_beaver_r[0];
  if (dpf_rank) {
    z[0] ^= mask[0] & mask[1];
    z[1] ^= mask[2] & mask[3];
  }

  return z[0] ^ z[1];
};

template <typename T>
std::vector<T> mul2pc(KernelEvalContext *ctx, absl::Span<T const> x,
                      absl::Span<T const> y, size_t adjust_rank) {
  auto *comm = ctx->getState<Communicator>();
  size_t dst_rank =
      comm->getRank() == adjust_rank ? comm->prevRank() : comm->nextRank();

  auto numel = x.size();
  const auto &prim = genOramBeaverPrim<T>(ctx, numel, OpKind::Mul, adjust_rank);
  const auto &a = std::get<0>(prim);
  const auto &b = std::get<1>(prim);
  const auto &c = std::get<2>(prim);
  std::vector<T> eu(numel * 2);
  absl::Span<T> e(eu.data(), numel);
  absl::Span<T> u(eu.data() + numel, numel);
  std::vector<T> z(numel);

  pforeach(0, numel, [&](int64_t idx) {
    e[idx] = x[idx] - a[idx];  // e = x - a;
    u[idx] = y[idx] - b[idx];  // u = y - b;
  });

  comm->sendAsync<T>(dst_rank, absl::MakeSpan(eu), "open(x-a, y-b)");

  auto temp_eu = comm->recv<T>(dst_rank, "open(x-a, y-b)");
  pforeach(0, numel * 2, [&](int64_t idx) { eu[idx] += temp_eu[idx]; });

  // Zi = Ci + (X - A) * Bi + (Y - B) * Ai + <(X - A) * (Y - B)>
  pforeach(0, a.size(), [&](int64_t idx) {
    z[idx] = c[idx] + e[idx] * b[idx] + u[idx] * a[idx];
    if (comm->getRank() == adjust_rank) {
      // z += (X-A) * (Y-B);
      z[idx] += e[idx] * u[idx];
    }
  });
  return z;
};

template <typename T>
void OramContext<T>::onehotB2A(KernelEvalContext *ctx, DpfGenCtrl ctrl) {
  auto *comm = ctx->getState<Communicator>();
  auto *prg = ctx->getState<PrgState>();

  auto dpf_rank = comm->getRank() == static_cast<size_t>(ctrl);
  size_t dst_rank = dpf_rank ? comm->prevRank() : comm->nextRank();
  int64_t dpf_idx = comm->getRank() == static_cast<size_t>(ctrl) ? 0 : 1;

  T pm = 0;
  T F = 0;
  std::vector<T> r(1);
  prg->fillPriv(absl::MakeSpan(r));

  const std::vector<T> e = dpf_e[dpf_idx];
  const std::vector<T> v = convert_help_v[dpf_idx];
  std::for_each(e.begin(), e.end(), [&](T ele) { pm += ele; });
  std::for_each(v.begin(), v.end(), [&](T ele) { F -= ele; });
  T blinded_pm = pm + r[0];

  // open blinded_pm
  comm->sendAsync<T>(dst_rank, {blinded_pm}, "open(blinded_pm)");
  blinded_pm += comm->recv<T>(dst_rank, "open(blinded_pm)")[0];

  auto pm_mul_F = mul2pc<T>(ctx, {pm}, {F}, static_cast<size_t>(ctrl));
  T blinded_F = pm_mul_F[0] + r[0];

  // open blinded_F
  comm->sendAsync<T>(dst_rank, {blinded_F}, "open(blinded_F)");
  blinded_F += comm->recv<T>(dst_rank, "open(blinded_F)")[0];

  std::vector<T> e_a(dpf_size_);
  pforeach(0, dpf_size_, [&](int64_t idx) {
    e_a[idx] = e[idx] * blinded_pm - v[idx] - e[idx] * blinded_F;
  });

  dpf_e[dpf_idx] = e_a;
};

std::pair<std::vector<uint128_t>, std::vector<uint128_t>> genAesKey(
    KernelEvalContext *ctx, int64_t index_times) {
  auto *comm = ctx->getState<Communicator>();

  std::vector<uint128_t> self_aes_keys(index_times);
  pforeach(0, index_times, [&](int64_t idx) {
    self_aes_keys[idx] = yacl::crypto::SecureRandSeed();
  });

  auto next_aes_keys =
      comm->rotate<uint128_t>(absl::MakeSpan(self_aes_keys), "aes_key");

  return std::make_pair(self_aes_keys, next_aes_keys);
}

template <typename T>
void OramContext<T>::genDpf(KernelEvalContext *ctx, DpfGenCtrl ctrl,
                            uint128_t aes_key, uint128_t target_point) {
  auto *comm = ctx->getState<Communicator>();

  auto odpf = OramDpf(dpf_size_, yacl::crypto::SecureRandU128(), aes_key,
                      static_cast<uint128_t>(target_point));
  odpf.gen(ctx, ctrl);

  auto dpf_rank = comm->getRank() == static_cast<size_t>(ctrl);
  int64_t dpf_idx = dpf_rank ? 0 : 1;
  T neg_flag = dpf_rank ? -1 : 1;

  // cast e and v to T type and convert v to arith
  // leave convert e outside
  std::transform(odpf.final_e.begin(), odpf.final_e.begin() + dpf_size_,
                 dpf_e[dpf_idx].begin(),
                 [&](uint8_t x) { return neg_flag * static_cast<T>(x); });
  std::transform(odpf.final_v.begin(), odpf.final_v.begin() + dpf_size_,
                 convert_help_v[dpf_idx].begin(),
                 [&](uint128_t x) { return neg_flag * static_cast<T>(x); });
};

std::vector<DpfKeyT> OramDpf::lengthDoubling(
    const std::vector<DpfKeyT> &input) {
  std::vector<DpfKeyT> plain_text(input.size() * 2);
  std::vector<DpfKeyT> cipher_text(input.size() * 2);
  pforeach(0, input.size(), [&](int64_t idx) {
    plain_text[idx * 2] = input[idx];
    plain_text[idx * 2 + 1] = input[idx] ^ 1;
  });

  aes_crypto_.Encrypt(absl::MakeConstSpan(plain_text),
                      absl::MakeSpan(cipher_text));

  pforeach(0, input.size(), [&](int64_t idx) {
    cipher_text[idx * 2] ^= plain_text[idx * 2];
    cipher_text[idx * 2 + 1] ^= plain_text[idx * 2] ^ 1;
  });

  return cipher_text;
};

void OramDpf::gen(KernelEvalContext *ctx, DpfGenCtrl ctrl) {
  auto *comm = ctx->getState<Communicator>();
  auto dpf_rank = comm->getRank() == static_cast<size_t>(ctrl);
  size_t dst_rank = dpf_rank ? comm->prevRank() : comm->nextRank();

  // generate 2*depth beaver triple
  auto [a, b, c] = genOramBeaverPrim<DpfKeyT>(ctx, depth_ * 2, OpKind::And,
                                              static_cast<size_t>(ctrl));
  // set lsb of root seed
  root_seed_ = setLsb(root_seed_, dpf_rank ? 0 : 1);
  // break target point into bit vectors
  std::vector<DpfKeyT> target_point_bits =
      bitDecomposeToDpfKeyT(target_point_, depth_);
  std::vector<DpfKeyT> prev_v = {root_seed_};
  std::vector<CorrectionFlagT> prev_e = {
      static_cast<uint8_t>(dpf_rank ? 0 : 1)};
  int64_t half_layer_numel = 1;

  for (int64_t l = 0; l < depth_; l++) {
    // last layer, reduce keynum
    if (l == depth_ - 1) {
      half_layer_numel = numel_ / 2 + static_cast<int64_t>(numel_ % 2 != 0);
      prev_v.resize(half_layer_numel);
    }
    // generate key on ith level, [2*i] for left child, [2*i+1] for right child
    std::vector<DpfKeyT> cur_v(half_layer_numel * 2);
    std::vector<CorrectionFlagT> cur_e(half_layer_numel * 2);

    DpfKeyT sumL = 0;
    DpfKeyT sumR = 0;

    cur_v = lengthDoubling(prev_v);

    for (int64_t i = 0; i < half_layer_numel; i++) {
      sumL ^= cur_v[2 * i];
      sumR ^= cur_v[2 * i + 1];
    }

    // compute (target_point_bits[i] & L) ^ (1 ^ target_point_bits[i] & R)
    std::array<DpfKeyT, 3> oram_and_beaver_l = {a[l * 2], b[l * 2], c[l * 2]};
    std::array<DpfKeyT, 3> oram_and_beaver_r = {a[l * 2 + 1], b[l * 2 + 1],
                                                c[l * 2 + 1]};

    cw[l] = computecw(ctx, target_point_bits[l], sumL, sumR, oram_and_beaver_l,
                      oram_and_beaver_r, ctrl);

    std::vector<DpfKeyT> exchanged_cw = {cw[l]};
    comm->sendAsync<DpfKeyT>(dst_rank, absl::MakeSpan(exchanged_cw), "open_cw");
    exchanged_cw = comm->recv<DpfKeyT>(dst_rank, "open_cw");
    cw[l] ^= exchanged_cw[0];

    cwt[l][0] = getLsb(sumL) ^ getLsb(target_point_bits[l]);
    cwt[l][1] = getLsb(sumR) ^ getLsb(target_point_bits[l]);

    comm->sendAsync<CorrectionFlagT>(dst_rank, absl::MakeSpan(cwt[l]),
                                     "open_cwt");
    auto exchanged_cwt = comm->recv<CorrectionFlagT>(dst_rank, "open_cwt");

    cwt[l][0] ^= exchanged_cwt[0] ^ 1;
    cwt[l][1] ^= exchanged_cwt[1];

    pforeach(0, half_layer_numel, [&](int64_t i) {
      cur_e[i * 2] = getLsb(cur_v[i * 2]) ^ (prev_e[i] & cwt[l][0]);
      cur_e[i * 2 + 1] = getLsb(cur_v[i * 2 + 1]) ^ (prev_e[i] & cwt[l][1]);
      DpfKeyT extended_e = prev_e[i] == 0 ? 0 : -1;
      cur_v[i * 2] ^= extended_e & cw[l];
      cur_v[i * 2 + 1] ^= extended_e & cw[l];
    });

    half_layer_numel *= 2;
    prev_e.assign(cur_e.begin(), cur_e.end());
    prev_v.assign(cur_v.begin(), cur_v.end());
  }

  std::copy(prev_e.begin(), prev_e.begin() + numel_, final_e.begin());
  // use v for conversion, instead of spliting to (int64, int64) in DUORAM
  std::copy(prev_v.begin(), prev_v.begin() + numel_, final_v.begin());
};

}  // namespace spu::mpc::oram
