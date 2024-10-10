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

#include "libspu/mpc/aby3/permute.h"

#include "libspu/mpc/aby3/type.h"
#include "libspu/mpc/aby3/value.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/utils/permute.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/utils.h"

namespace spu::mpc::aby3 {

MemRef RandPermM::proc(KernelEvalContext* ctx, const Shape& shape) const {
  MemRef out(makeType<PermShareTy>(), shape);

  auto* prg_state = ctx->getState<PrgState>();
  const auto& pvs = prg_state->genPrssPermPair(out.numel());
  const auto& pv_self = pvs.first;
  const auto& pv_next = pvs.second;

  auto out1 = getFirstShare(out);
  auto out2 = getSecondShare(out);
  DISPATCH_ALL_STORAGE_TYPES(out.eltype().storage_type(), [&]() {
    MemRefView<ScalarT> _out1(out1);
    MemRefView<ScalarT> _out2(out2);
    pforeach(0, out.numel(), [&](int64_t idx) {
      _out1[idx] = ScalarT(pv_self[idx]);
      _out2[idx] = ScalarT(pv_next[idx]);
    });
  });
  return out;
}

// Ref: https://eprint.iacr.org/2019/695.pdf
// Algorithm 9: Optimized shuffling protocol
MemRef PermAM::proc(KernelEvalContext* ctx, const MemRef& in,
                    const MemRef& perm) const {
  auto* comm = ctx->getState<Communicator>();
  const auto numel = in.numel();
  auto* prg_state = ctx->getState<PrgState>();

  auto pv_self = getFirstShare(perm);
  auto pv_next = getSecondShare(perm);

  MemRef out(in.eltype(), in.shape());
  DISPATCH_ALL_STORAGE_TYPES(in.eltype().storage_type(), [&]() {
    using el_t = ScalarT;
    using shr_t = std::array<el_t, 2>;

    MemRefView<shr_t> _out(out);
    MemRefView<shr_t> _in(in);

    std::vector<el_t> a0(numel);
    std::vector<el_t> a1(numel);
    prg_state->fillPrssPair(a0.data(), a1.data(), GetVectorNumBytes(a0));

    DISPATCH_ALL_STORAGE_TYPES(pv_next.eltype().storage_type(), [&]() {
      using pv_t = ScalarT;
      MemRefView<pv_t> _pv_self(pv_self);
      MemRefView<pv_t> _pv_next(pv_next);
      if (comm->getRank() == 0) {
        std::vector<el_t> tmp(numel);
        std::vector<el_t> delta(numel);
        pforeach(0, numel, [&](int64_t idx) {
          tmp[idx] = _in[_pv_self[idx]][0] + _in[_pv_self[idx]][1] - a0[idx];
        });
        pforeach(0, numel, [&](int64_t idx) {
          delta[idx] = tmp[_pv_next[idx]] - a1[idx];
        });
        comm->sendAsync<el_t>(2, delta, "delta");

        // 2to3 re-share
        std::vector<el_t> r0(numel);
        std::vector<el_t> r1(numel);
        prg_state->fillPrssPair(r0.data(), r1.data(), GetVectorNumBytes(r1));
        pforeach(0, numel, [&](int64_t idx) {
          _out[idx][0] = r0[idx];
          _out[idx][1] = r1[idx];
        });

      } else if (comm->getRank() == 1) {
        auto gama = comm->recv<el_t>(2, "gama");
        std::vector<el_t> tmp(numel);
        std::vector<el_t> beta(numel);

        pforeach(0, numel, [&](int64_t idx) {
          tmp[idx] = gama[_pv_self[idx]] + a0[idx];
        });
        pforeach(0, numel,
                 [&](int64_t idx) { beta[idx] = tmp[_pv_next[idx]]; });

        // 2to3 re-share
        std::vector<el_t> r0(numel);
        prg_state->fillPrssPair(r0.data(), nullptr, GetVectorNumBytes(r0));
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
          gama[idx] = _in[_pv_next[idx]][0] + a1[idx];
        });
        comm->sendAsync<el_t>(1, gama, "gama");
        auto delta = comm->recv<el_t>(0, "delta");
        pforeach(0, numel,
                 [&](int64_t idx) { beta[idx] = delta[_pv_self[idx]]; });

        // 2to3 re-share
        std::vector<el_t> r1(numel);
        prg_state->fillPrssPair(nullptr, r1.data(), GetVectorNumBytes(r1));
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
  });
  return out;
}

MemRef PermAP::proc(KernelEvalContext* ctx, const MemRef& in,
                    const MemRef& perm) const {
  MemRef out(in.eltype(), in.shape());

  if (out.numel() != 0) {
    const auto in1 = getFirstShare(in);
    const auto in2 = getSecondShare(in);
    auto perm1 = applyPerm(in1, perm);
    auto perm2 = applyPerm(in2, perm);

    auto out1 = getFirstShare(out);
    auto out2 = getSecondShare(out);

    ring_assign(out1, perm1);
    ring_assign(out2, perm2);
  }
  return out;
}

// Ref: https://eprint.iacr.org/2019/695.pdf
// Algorithm 17: Optimized unshuffling protocol
MemRef InvPermAM::proc(KernelEvalContext* ctx, const MemRef& in,
                       const MemRef& perm) const {
  auto* comm = ctx->getState<Communicator>();
  const auto numel = in.numel();
  auto* prg_state = ctx->getState<PrgState>();

  auto pv_self = getFirstShare(perm);
  auto pv_next = getSecondShare(perm);

  MemRef out(in.eltype(), in.shape());
  DISPATCH_ALL_STORAGE_TYPES(in.eltype().storage_type(), [&]() {
    using el_t = ScalarT;
    using shr_t = std::array<el_t, 2>;

    MemRefView<shr_t> _out(out);
    MemRefView<shr_t> _in(in);

    std::vector<el_t> a0(numel);
    std::vector<el_t> a1(numel);
    prg_state->fillPrssPair(a0.data(), a1.data(), GetVectorNumBytes(a0));

    DISPATCH_ALL_STORAGE_TYPES(pv_self.eltype().storage_type(), [&]() {
      using pv_t = ScalarT;
      MemRefView<pv_t> _pv_self(pv_self);
      MemRefView<pv_t> _pv_next(pv_next);

      if (comm->getRank() == 0) {
        std::vector<el_t> beta(numel);
        std::vector<el_t> tmp(numel);
        auto gama = comm->recv<el_t>(2, "gama");

        pforeach(0, numel, [&](int64_t idx) {
          tmp[_pv_next[idx]] = gama[idx] + a1[_pv_next[idx]];
        });
        pforeach(0, numel,
                 [&](int64_t idx) { beta[_pv_self[idx]] = tmp[idx]; });

        // 2to3 re-share
        std::vector<el_t> r1(numel);
        prg_state->fillPrssPair(nullptr, r1.data(), GetVectorNumBytes(r1));
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
          tmp[_pv_next[idx]] = _in[idx][0] + _in[idx][1] - a1[_pv_next[idx]];
        });
        pforeach(0, numel, [&](int64_t idx) {
          delta[_pv_self[idx]] = tmp[idx] - a0[_pv_self[idx]];
        });
        comm->sendAsync<el_t>(2, delta, "delta");

        // 2to3 re-share
        std::vector<el_t> r0(numel);
        std::vector<el_t> r1(numel);
        prg_state->fillPrssPair(r0.data(), r1.data(), GetVectorNumBytes(r1));
        pforeach(0, numel, [&](int64_t idx) {
          _out[idx][0] = r0[idx];
          _out[idx][1] = r1[idx];
        });

      } else if (comm->getRank() == 2) {
        std::vector<el_t> gama(numel);
        std::vector<el_t> beta(numel);
        pforeach(0, numel, [&](int64_t idx) {
          gama[_pv_self[idx]] = _in[idx][1] + a0[_pv_self[idx]];
        });
        comm->sendAsync<el_t>(0, gama, "gama");
        auto delta = comm->recv<el_t>(1, "delta");
        pforeach(0, numel,
                 [&](int64_t idx) { beta[_pv_next[idx]] = delta[idx]; });

        // 2to3 re-share
        std::vector<el_t> r0(numel);
        prg_state->fillPrssPair(r0.data(), nullptr, GetVectorNumBytes(r0));
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
  });
  return out;
}

MemRef InvPermAP::proc(KernelEvalContext* ctx, const MemRef& in,
                       const MemRef& perm) const {
  MemRef out(in.eltype(), in.shape());

  if (out.numel() != 0) {
    const auto in1 = getFirstShare(in);
    const auto in2 = getSecondShare(in);

    auto perm1 = applyInvPerm(in1, perm);
    auto perm2 = applyInvPerm(in2, perm);

    auto out1 = getFirstShare(out);
    auto out2 = getSecondShare(out);

    ring_assign(out1, perm1);
    ring_assign(out2, perm2);
  }
  return out;
}

}  // namespace spu::mpc::aby3
