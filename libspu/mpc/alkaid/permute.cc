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

#include "libspu/mpc/alkaid/permute.h"

#include "libspu/mpc/alkaid/type.h"
#include "libspu/mpc/alkaid/value.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/utils/permute.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::alkaid {

namespace {

PermVector ring2pv(const NdArrayRef& x) {
  SPU_ENFORCE(x.eltype().isa<Ring2k>(), "must be ring2k_type, got={}",
              x.eltype());
  const auto field = x.eltype().as<Ring2k>()->field();
  PermVector pv(x.numel());
  DISPATCH_ALL_FIELDS(field, "_", [&]() {
    NdArrayView<ring2k_t> _x(x);
    pforeach(0, x.numel(), [&](int64_t idx) { pv[idx] = int64_t(_x[idx]); });
  });
  return pv;
}

}  // namespace

NdArrayRef RandPermM::proc(KernelEvalContext* ctx, const Shape& shape) const {
  NdArrayRef out(makeType<PShrTy>(), shape);

  // generate a RandU64 pair as permutation seeds
  auto* prg_state = ctx->getState<PrgState>();
  const auto [seed_self, seed_next] =
      prg_state->genPrssPair(FieldType::FM64, {1}, PrgState::GenPrssCtrl::Both);
  NdArrayView<uint64_t> _seed_self(seed_self);
  NdArrayView<uint64_t> _seed_next(seed_next);
  const auto pv_self = genRandomPerm(out.numel(), _seed_self[0]);
  const auto pv_next = genRandomPerm(out.numel(), _seed_next[0]);

  const auto field = out.eltype().as<PShrTy>()->field();
  auto out1 = getFirstShare(out);
  auto out2 = getSecondShare(out);
  DISPATCH_ALL_FIELDS(field, "_", [&]() {
    NdArrayView<ring2k_t> _out1(out1);
    NdArrayView<ring2k_t> _out2(out2);
    pforeach(0, out.numel(), [&](int64_t idx) {
      _out1[idx] = ring2k_t(pv_self[idx]);
      _out2[idx] = ring2k_t(pv_next[idx]);
    });
  });
  return out;
}

// Ref: https://eprint.iacr.org/2019/695.pdf
// Algorithm 9: Optimized shuffling protocol
NdArrayRef PermAM::proc(KernelEvalContext* ctx, const NdArrayRef& in,
                        const NdArrayRef& perm) const {
  auto* comm = ctx->getState<Communicator>();
  const auto numel = in.numel();
  const auto field = in.eltype().as<AShrTy>()->field();
  auto* prg_state = ctx->getState<PrgState>();

  PermVector pv_self = ring2pv(getFirstShare(perm));
  PermVector pv_next = ring2pv(getSecondShare(perm));

  NdArrayRef out(in.eltype(), in.shape());
  DISPATCH_ALL_FIELDS(field, "_", [&]() {
    using el_t = ring2k_t;
    using shr_t = std::array<el_t, 2>;

    NdArrayView<shr_t> _out(out);
    NdArrayView<shr_t> _in(in);

    std::vector<el_t> a0(numel);
    std::vector<el_t> a1(numel);
    prg_state->fillPrssPair(a0.data(), a1.data(), a0.size(),
                            PrgState::GenPrssCtrl::Both);

    if (comm->getRank() == 0) {
      std::vector<el_t> tmp(numel);
      std::vector<el_t> delta(numel);
      pforeach(0, numel, [&](int64_t idx) {
        tmp[idx] = _in[pv_self[idx]][0] + _in[pv_self[idx]][1] - a0[idx];
      });
      pforeach(0, numel,
               [&](int64_t idx) { delta[idx] = tmp[pv_next[idx]] - a1[idx]; });
      comm->sendAsync<el_t>(2, delta, "delta");

      // 2to3 re-share
      std::vector<el_t> r0(numel);
      std::vector<el_t> r1(numel);
      prg_state->fillPrssPair(r0.data(), r1.data(), r1.size(),
                              PrgState::GenPrssCtrl::Both);
      pforeach(0, numel, [&](int64_t idx) {
        _out[idx][0] = r0[idx];
        _out[idx][1] = r1[idx];
      });

    } else if (comm->getRank() == 1) {
      auto gama = comm->recv<el_t>(2, "gama");
      std::vector<el_t> tmp(numel);
      std::vector<el_t> beta(numel);

      pforeach(0, numel,
               [&](int64_t idx) { tmp[idx] = gama[pv_self[idx]] + a0[idx]; });
      pforeach(0, numel, [&](int64_t idx) { beta[idx] = tmp[pv_next[idx]]; });

      // 2to3 re-share
      std::vector<el_t> r0(numel);
      prg_state->fillPrssPair<el_t>(r0.data(), nullptr, r0.size(),
                                    PrgState::GenPrssCtrl::First);
      pforeach(0, numel, [&](int64_t idx) { beta[idx] -= r0[idx]; });

      comm->sendAsync<el_t>(2, beta, "2to3");
      tmp = comm->recv<el_t>(2, "2to3");

      pforeach(0, numel, [&](int64_t idx) {
        _out[idx][0] = r0[idx];
        _out[idx][1] = beta[idx] + tmp[idx];
      });

    } else if (comm->getRank() == 2) {
      std::vector<el_t> gama(numel);
      std::vector<el_t> beta(numel);
      pforeach(0, numel, [&](int64_t idx) {
        gama[idx] = _in[pv_next[idx]][0] + a1[idx];
      });
      comm->sendAsync<el_t>(1, gama, "gama");
      auto delta = comm->recv<el_t>(0, "delta");
      pforeach(0, numel, [&](int64_t idx) { beta[idx] = delta[pv_self[idx]]; });

      // 2to3 re-share
      std::vector<el_t> r1(numel);
      prg_state->fillPrssPair<el_t>(nullptr, r1.data(), r1.size(),
                                    PrgState::GenPrssCtrl::Second);
      pforeach(0, numel, [&](int64_t idx) {  //
        beta[idx] -= r1[idx];
      });
      comm->sendAsync<el_t>(1, beta, "2to3");
      auto tmp = comm->recv<el_t>(1, "2to3");

      // rebuild the final result.
      pforeach(0, numel, [&](int64_t idx) {
        _out[idx][0] = beta[idx] + tmp[idx];
        _out[idx][1] = r1[idx];
      });
    } else {
      SPU_THROW("Party number exceeds 3!");
    }
  });
  return out;
}

NdArrayRef PermAP::proc(KernelEvalContext* ctx, const NdArrayRef& in,
                        const NdArrayRef& perm) const {
  NdArrayRef out(in.eltype(), in.shape());

  if (out.numel() != 0) {
    PermVector pv = ring2pv(perm);
    const auto& in1 = getFirstShare(in);
    const auto& in2 = getSecondShare(in);
    auto perm1 = applyPerm(in1, pv);
    auto perm2 = applyPerm(in2, pv);

    auto out1 = getFirstShare(out);
    auto out2 = getSecondShare(out);

    ring_assign(out1, perm1);
    ring_assign(out2, perm2);
  }
  return out;
}

// Ref: https://eprint.iacr.org/2019/695.pdf
// Algorithm 17: Optimized unshuffling protocol
NdArrayRef InvPermAM::proc(KernelEvalContext* ctx, const NdArrayRef& in,
                           const NdArrayRef& perm) const {
  auto* comm = ctx->getState<Communicator>();
  const auto numel = in.numel();
  const auto field = in.eltype().as<AShrTy>()->field();
  auto* prg_state = ctx->getState<PrgState>();

  PermVector pv_self = ring2pv(getFirstShare(perm));
  PermVector pv_next = ring2pv(getSecondShare(perm));

  NdArrayRef out(in.eltype(), in.shape());
  DISPATCH_ALL_FIELDS(field, "_", [&]() {
    using el_t = ring2k_t;
    using shr_t = std::array<el_t, 2>;

    NdArrayView<shr_t> _out(out);
    NdArrayView<shr_t> _in(in);

    std::vector<el_t> a0(numel);
    std::vector<el_t> a1(numel);
    prg_state->fillPrssPair(a0.data(), a1.data(), a0.size(),
                            PrgState::GenPrssCtrl::Both);

    if (comm->getRank() == 0) {
      std::vector<el_t> beta(numel);
      std::vector<el_t> tmp(numel);
      auto gama = comm->recv<el_t>(2, "gama");

      pforeach(0, numel, [&](int64_t idx) {
        tmp[pv_next[idx]] = gama[idx] + a1[pv_next[idx]];
      });
      pforeach(0, numel, [&](int64_t idx) { beta[pv_self[idx]] = tmp[idx]; });

      // 2to3 re-share
      std::vector<el_t> r1(numel);
      prg_state->fillPrssPair<el_t>(nullptr, r1.data(), r1.size(),
                                    PrgState::GenPrssCtrl::Second);
      pforeach(0, numel, [&](int64_t idx) {  //
        beta[idx] -= r1[idx];
      });
      comm->sendAsync<el_t>(2, beta, "2to3");
      tmp = comm->recv<el_t>(2, "2to3");

      // rebuild the final result.
      pforeach(0, numel, [&](int64_t idx) {
        _out[idx][0] = beta[idx] + tmp[idx];
        _out[idx][1] = r1[idx];
      });
    } else if (comm->getRank() == 1) {
      std::vector<el_t> tmp(numel);
      std::vector<el_t> delta(numel);

      pforeach(0, numel, [&](int64_t idx) {
        tmp[pv_next[idx]] = _in[idx][0] + _in[idx][1] - a1[pv_next[idx]];
      });
      pforeach(0, numel, [&](int64_t idx) {
        delta[pv_self[idx]] = tmp[idx] - a0[pv_self[idx]];
      });
      comm->sendAsync<el_t>(2, delta, "delta");

      // 2to3 re-share
      std::vector<el_t> r0(numel);
      std::vector<el_t> r1(numel);
      prg_state->fillPrssPair(r0.data(), r1.data(), r1.size(),
                              PrgState::GenPrssCtrl::Both);
      pforeach(0, numel, [&](int64_t idx) {
        _out[idx][0] = r0[idx];
        _out[idx][1] = r1[idx];
      });

    } else if (comm->getRank() == 2) {
      std::vector<el_t> gama(numel);
      std::vector<el_t> beta(numel);
      pforeach(0, numel, [&](int64_t idx) {
        gama[pv_self[idx]] = _in[idx][1] + a0[pv_self[idx]];
      });
      comm->sendAsync<el_t>(0, gama, "gama");
      auto delta = comm->recv<el_t>(1, "delta");
      pforeach(0, numel, [&](int64_t idx) { beta[pv_next[idx]] = delta[idx]; });

      // 2to3 re-share
      std::vector<el_t> r0(numel);
      prg_state->fillPrssPair<el_t>(r0.data(), nullptr, r0.size(),
                                    PrgState::GenPrssCtrl::First);
      pforeach(0, numel, [&](int64_t idx) {  //
        beta[idx] -= r0[idx];
      });

      comm->sendAsync<el_t>(0, beta, "2to3");
      auto tmp = comm->recv<el_t>(0, "2to3");

      pforeach(0, numel, [&](int64_t idx) {
        _out[idx][0] = r0[idx];
        _out[idx][1] = beta[idx] + tmp[idx];
      });

    } else {
      SPU_THROW("Party number exceeds 3!");
    }
  });
  return out;
}

NdArrayRef InvPermAP::proc(KernelEvalContext* ctx, const NdArrayRef& in,
                           const NdArrayRef& perm) const {
  NdArrayRef out(in.eltype(), in.shape());

  if (out.numel() != 0) {
    PermVector pv = ring2pv(perm);
    const auto& in1 = getFirstShare(in);
    const auto& in2 = getSecondShare(in);

    auto perm1 = applyInvPerm(in1, pv);
    auto perm2 = applyInvPerm(in2, pv);

    auto out1 = getFirstShare(out);
    auto out2 = getSecondShare(out);

    ring_assign(out1, perm1);
    ring_assign(out2, perm2);
  }
  return out;
}

}  // namespace spu::mpc::alkaid
