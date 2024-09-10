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

#include "libspu/mpc/albo/oram.h"

#include <future>

#include "yacl/crypto/rand/rand.h"

#include "libspu/mpc/ab_api.h"
#include "libspu/mpc/albo/type.h"
#include "libspu/mpc/albo/value.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::albo {

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
    ring_assign(o2, comm->rotate(z1, kBindName));
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
    ring_assign(o2, comm->rotate(out2pc, kBindName));
    f.get();
  });

  return out;
}
}  // namespace spu::mpc::albo
