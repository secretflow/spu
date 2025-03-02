// Copyright 2024 Ant Group Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "libspu/mpc/cheetah/permute.h"

#include "libspu/mpc/ab_api.h"
#include "libspu/mpc/cheetah/type.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/utils/permute.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/waksman_net.h"

namespace spu::mpc::cheetah {

namespace {

inline int64_t getOwner(const NdArrayRef& x) {
  return x.eltype().as<Priv2kTy>()->owner();
}

// x is secret, y is private
NdArrayRef wrap_mulav(SPUContext* ctx, const NdArrayRef& x,
                      const NdArrayRef& y) {
  return UnwrapValue(mul_av(ctx, WrapValue(x), WrapValue(y)).value());
}

// MUX(flag, rhs, lhs)
NdArrayRef route_to_next_layer_top(SPUContext* ctx, const NdArrayRef& lhs,
                                   const NdArrayRef& rhs,
                                   const NdArrayRef& flag) {
  auto delta = ring_sub(rhs, lhs);
  // Indeed, flag is a private bool, so `mul_a1bv` can also be used.
  // Although `mul_a1bv` is friendly to communication amount (maybe 10+
  // times less), the rounds cost is much higher than `mul_av` (maybe 5+ times
  // more). As a result, `mul_av` is used here.
  return ring_add(lhs, wrap_mulav(ctx, delta, flag));
}

NdArrayRef get_private_flag(absl::Span<uint8_t> flag, int64_t flag_size,
                            size_t rank, const FieldType field) {
  // only perm Party will get the exactly switch flag.
  bool is_cur_rank = !flag.empty();
  auto const out_ty = makeType<Priv2kTy>(field, rank);

  if (!is_cur_rank) {
    return makeConstantArrayRef(out_ty, {flag_size});
  }

  return DISPATCH_ALL_FIELDS(field, [&]() {
    using T = ring2k_t;
    NdArrayRef out(out_ty, {flag_size});
    NdArrayView<T> _out(out);

    pforeach(0, flag_size,
             [&](int64_t idx) { _out[idx] = static_cast<T>(flag[idx]); });

    return out;
  });
}

NdArrayRef SecureInvPerm(KernelEvalContext* ctx, const NdArrayRef& x,
                         size_t perm_rank, const Index& pv) {
  const auto is_cur_rank = ctx->lctx()->Rank() == perm_rank;
  const auto field = x.eltype().as<RingTy>()->field();
  const auto num_packets = pv.size();
  SPU_ENFORCE(num_packets > 0, "permutation vector should not be empty.");
  // build graph structure
  const auto graph = generate_as_waksman_topology(num_packets);
  const auto width = graph.size();

  NdArrayRef ret = x;
  AsWaksmanRouting routing;
  if (is_cur_rank) {
    // only perm owner can generate routing
    routing = get_as_waksman_routing(pv);
  }

  // walk through the whole network, and re-arrange the element
  for (size_t column_idx = 0; column_idx < width; column_idx++) {
    Index lhs_indices;
    Index rhs_indices;
    Index top_routed;
    Index bottom_routed;

    // straight wires routed
    Index straight_routed_last_layer;
    Index straight_routed_cur_layer;
    // TODO: use dynamic bit set to save memory
    std::vector<uint8_t> flag;
    int64_t flag_size = 0;

    bool append_lhs = true;
    for (size_t i = 0; i < num_packets; i++) {
      // we make use of the special construction that the switch always comes
      // from the adjacent wires.
      if (graph[column_idx][i].first == graph[column_idx][i].second) {
        // no switch here, just skip.
        straight_routed_last_layer.push_back(i);
        straight_routed_cur_layer.push_back(graph[column_idx][i].first);
      } else {
        if (append_lhs) {
          // collect the current layer indices
          lhs_indices.push_back(i);
          // collect the next layer indices
          auto [upper, down] = graph[column_idx][i];
          top_routed.push_back(upper);
          bottom_routed.push_back(down);

          if (is_cur_rank) {
            // collect the switch flag
            auto it = routing[column_idx].find(i);
            flag.push_back(static_cast<uint8_t>(it->second));
          }
          flag_size++;

          append_lhs = false;
        } else {
          rhs_indices.push_back(i);
          append_lhs = true;
        }
      }
    }

    auto straight_last_value = ret.linear_gather(straight_routed_last_layer);
    auto lhs_value = ret.linear_gather(lhs_indices);
    auto rhs_value = ret.linear_gather(rhs_indices);
    auto flag_value =
        get_private_flag(absl::MakeSpan(flag), flag_size, perm_rank, field);

    // top
    auto p =
        route_to_next_layer_top(ctx->sctx(), lhs_value, rhs_value, flag_value);
    // bottom
    auto q = ring_sub(ring_add(lhs_value, rhs_value), p);

    // update ret for next layer
    ret.linear_scatter(p, top_routed);
    ret.linear_scatter(q, bottom_routed);
    ret.linear_scatter(straight_last_value, straight_routed_cur_layer);
  }

  return ret;
}
}  // namespace

///
/// The following implementations are mainly borrowed from semi2k/permute.cc
///
NdArrayRef RandPermM::proc(KernelEvalContext* ctx, const Shape& shape) const {
  NdArrayRef out(makeType<PShrTy>(), shape);

  auto* prg_state = ctx->getState<PrgState>();
  const auto perm_vector = prg_state->genPrivPerm(out.numel());

  const auto field = out.eltype().as<PShrTy>()->field();
  DISPATCH_ALL_FIELDS(field, [&]() {
    NdArrayView<ring2k_t> _out(out);
    pforeach(0, out.numel(),
             [&](int64_t idx) { _out[idx] = ring2k_t(perm_vector[idx]); });
  });

  return out;
}

NdArrayRef PermAM::proc(KernelEvalContext* ctx, const NdArrayRef& in,
                        const NdArrayRef& perm) const {
  auto* comm = ctx->getState<Communicator>();
  Index pv = ring2pv(perm);
  // copy here, to avoid inplace modification in SecureInvPerm
  NdArrayRef out = in.clone();

  for (size_t i = 0; i < comm->getWorldSize(); i++) {
    out = SecureInvPerm(ctx, out, i, pv);
  }

  return out;
}

NdArrayRef PermAP::proc(KernelEvalContext* ctx, const NdArrayRef& in,
                        const NdArrayRef& perm) const {
  return applyPerm(in, perm);
}

NdArrayRef InvPermAM::proc(KernelEvalContext* ctx, const NdArrayRef& in,
                           const NdArrayRef& perm) const {
  auto* comm = ctx->getState<Communicator>();
  // copy here, to avoid inplace modification in SecureInvPerm
  NdArrayRef out = in.clone();

  auto inv_perm = genInversePerm(perm);
  Index inv_pv = ring2pv(inv_perm);
  for (int i = comm->getWorldSize() - 1; i >= 0; --i) {
    out = SecureInvPerm(ctx, out, i, inv_pv);
  }
  return out;
}

NdArrayRef InvPermAP::proc(KernelEvalContext* ctx, const NdArrayRef& in,
                           const NdArrayRef& perm) const {
  return applyInvPerm(in, perm);
}

NdArrayRef InvPermAV::proc(KernelEvalContext* ctx, const NdArrayRef& in,
                           const NdArrayRef& perm) const {
  // copy here, to avoid inplace modification in SecureInvPerm
  NdArrayRef out = in.clone();

  return SecureInvPerm(ctx, out, getOwner(perm), ring2pv(perm));
}

}  // namespace spu::mpc::cheetah
