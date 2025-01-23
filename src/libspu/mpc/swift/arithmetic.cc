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

#include "libspu/mpc/swift/arithmetic.h"

#include <functional>
#include <iostream>
#include <string>

#include "libspu/core/type_util.h"
#include "libspu/core/vectorize.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/swift/commitment.h"
#include "libspu/mpc/swift/type.h"
#include "libspu/mpc/swift/value.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::swift {

NdArrayRef getOrCreateCompactArray(const NdArrayRef& in) {
  if (!in.isCompact()) {
    return in.clone();
  }
  return in;
}

NdArrayRef JointMessagePassing(KernelEvalContext* ctx, const NdArrayRef& msg,
                               size_t rank_send, size_t rank_hash,
                               size_t rank_recv, std::string_view tag) {
  auto const field = msg.eltype().as<RingTy>()->field();
  auto* comm = ctx->getState<Communicator>();
  auto ty = makeType<RingTy>(field);

  NdArrayRef res(ty, msg.shape());
  bool inconsistent_bit = false;

  auto rank = comm->getRank();

  if (rank == rank_send) {
    // send v to P_recv
    comm->sendAsync(rank_recv, msg, tag);

    res = msg;

    // malicious action 1 : P_send send wrong msg
    // comm->sendAsync(rank_recv, ring_neg(msg), tag);

    // recv inconsistent_bit from P_recv
    auto recv_b_from_pk = comm->recv<bool>(rank_recv, tag);
    inconsistent_bit = recv_b_from_pk[0];

    // exchange inconsistent bit between P_send and P_hash
    // reset inconsistent bit to b_send || b_hash
    std::array<bool, 1> send_b;
    send_b[0] = inconsistent_bit;

    // malicious action 2 : P_send send wrong inconsistent bit
    // send_b[0] = inconsistent_bit ^ true;

    auto recv_b_from_pj = comm->recv<bool>(rank_hash, tag);
    comm->sendAsync<bool>(rank_hash, absl::MakeSpan(send_b), tag);
    inconsistent_bit = recv_b_from_pk[0] || recv_b_from_pj[0];

    // broadcast Hash(v)
    // without considering the situation that some party is silent
    // which means the inconsistent bit of each party is all true or false
    if (inconsistent_bit) {
      std::string broadcast_msg(getOrCreateCompactArray(msg).data<char>(),
                                msg.numel() * msg.elsize());
      auto broadcast_msg_hash = commit(0, broadcast_msg, tag);
      yacl::ByteContainerView broadcast_msg_hash_bytes(
          reinterpret_cast<uint8_t const*>(broadcast_msg_hash.data()),
          broadcast_msg_hash.size());
      auto all_hash_bytes =
          yacl::link::AllGather(comm->lctx(), broadcast_msg_hash_bytes, tag);
      std::vector<std::string> all_hash(3);
      for (int i = 0; i < 3; i++) {
        all_hash[i] =
            std::string(reinterpret_cast<const char*>(all_hash_bytes[i].data()),
                        all_hash_bytes[i].size());
      }
      if (all_hash[rank_send] != all_hash[rank_hash]) {
        SPDLOG_INFO(
            "inconsistent check fail for tag {} from Party_{}, TTP = Party_{}",
            tag, rank_send, rank_recv);
      } else if (all_hash[rank_send] != all_hash[rank_recv]) {
        SPDLOG_INFO(
            "inconsistent check fail for tag {} from Party_{}, TTP = Party_{}",
            tag, rank_send, rank_hash);
      } else {
        SPDLOG_INFO(
            "inconsistent check fail for tag {} from Party_{}, TTP = Party_{}",
            tag, rank_send, rank_send);
      }
    }
  }
  if (rank == rank_hash) {
    res = msg;

    // send hash(v) to P_recv
    std::string msg_str(getOrCreateCompactArray(msg).data<char>(),
                        msg.numel() * msg.elsize());

    // malicious action 1 : P_hash send wrong hash
    // std::string msg_str(getOrCreateCompactArray(ring_neg(msg)).data<char>(),
    //                     msg.numel() * msg.elsize());

    auto msg_hash = commit(rank_hash, msg_str, tag);

    yacl::ByteContainerView msg_hash_bytes(
        reinterpret_cast<uint8_t const*>(msg_hash.data()), msg_hash.size());
    comm->sendAsync<std::uint8_t>(rank_recv, absl::MakeSpan(msg_hash_bytes),
                                  tag);

    // recv inconsistent_bit from P_recv
    auto recv_b_from_pk = comm->recv<bool>(rank_recv, tag);
    inconsistent_bit = recv_b_from_pk[0];

    // exchange inconsistent bit between P_send and P_hash
    // reset inconsistent bit to b_send || b_hash
    std::array<bool, 1> send_b;
    send_b[0] = inconsistent_bit;
    comm->sendAsync<bool>(rank_send, absl::MakeSpan(send_b), tag);
    auto recv_b_from_pi = comm->recv<bool>(rank_send, tag);
    inconsistent_bit = recv_b_from_pk[0] || recv_b_from_pi[0];

    // broadcast Hash(v)
    // without considering the situation that some party is silent
    // which means the inconsistent bit of each party is all true or false
    if (inconsistent_bit) {
      std::string broadcast_msg(getOrCreateCompactArray(msg).data<char>(),
                                msg.numel() * msg.elsize());
      auto broadcast_msg_hash = commit(0, broadcast_msg, tag);
      yacl::ByteContainerView broadcast_msg_hash_bytes(
          reinterpret_cast<uint8_t const*>(broadcast_msg_hash.data()),
          broadcast_msg_hash.size());
      auto all_hash_bytes =
          yacl::link::AllGather(comm->lctx(), broadcast_msg_hash_bytes, tag);
      std::vector<std::string> all_hash(3);
      for (int i = 0; i < 3; i++) {
        all_hash[i] =
            std::string(reinterpret_cast<const char*>(all_hash_bytes[i].data()),
                        all_hash_bytes[i].size());
      }
      if (all_hash[rank_send] != all_hash[rank_hash]) {
        SPDLOG_INFO(
            "inconsistent check fail for tag {} from Party_{}, TTP = Party_{}",
            tag, rank_hash, rank_recv);
      } else if (all_hash[rank_send] != all_hash[rank_recv]) {
        SPDLOG_INFO(
            "inconsistent check fail for tag {} from Party_{}, TTP = Party_{}",
            tag, rank_hash, rank_hash);
      } else {
        SPDLOG_INFO(
            "inconsistent check fail for tag {} from Party_{}, TTP = Party_{}",
            tag, rank_hash, rank_send);
      }
    }
  }
  if (rank == rank_recv) {
    // recv v and H_v from P_send and P_hash respectively
    auto res_v = comm->recv(rank_send, msg.eltype(), tag);
    res_v = res_v.reshape(msg.shape());

    auto recv_bytes = comm->recv<std::uint8_t>(rank_hash, tag);

    // check Hash(v) = H_v

    std::string recv_hash = std::string(
        reinterpret_cast<const char*>(recv_bytes.data()), recv_bytes.size());

    std::string recv_msg_str(getOrCreateCompactArray(res_v).data<char>(),
                             res_v.numel() * res_v.elsize());
    auto recv_msg_hash = commit(rank_hash, recv_msg_str, tag);

    if (recv_msg_hash != recv_hash) {
      inconsistent_bit = true;
    }

    if (!inconsistent_bit) {
      // send inconsistent_bit to P_send and P_hash
      std::array<bool, 1> send_b;
      send_b[0] = inconsistent_bit;

      comm->sendAsync<bool>(rank_hash, absl::MakeSpan(send_b), tag);
      comm->sendAsync<bool>(rank_send, absl::MakeSpan(send_b), tag);
    } else {
      SPDLOG_INFO("commit check fail for tag {}", tag);
      inconsistent_bit = true;

      // send inconsistent_bit to P_send and P_hash
      std::array<bool, 1> send_b;
      send_b[0] = inconsistent_bit;

      comm->sendAsync<bool>(rank_hash, absl::MakeSpan(send_b), tag);
      comm->sendAsync<bool>(rank_send, absl::MakeSpan(send_b), tag);

      // broadcast Hash(v)
      // without considering the situation that some party is silent
      // which means the inconsistent bit of each party is all true or false
      std::string broadcast_msg(getOrCreateCompactArray(res_v).data<char>(),
                                res_v.numel() * res_v.elsize());
      auto broadcast_msg_hash = commit(0, broadcast_msg, tag);
      yacl::ByteContainerView broadcast_msg_hash_bytes(
          reinterpret_cast<uint8_t const*>(broadcast_msg_hash.data()),
          broadcast_msg_hash.size());
      auto all_hash_bytes =
          yacl::link::AllGather(comm->lctx(), broadcast_msg_hash_bytes, tag);
      std::vector<std::string> all_hash(3);
      for (int i = 0; i < 3; i++) {
        all_hash[i] =
            std::string(reinterpret_cast<const char*>(all_hash_bytes[i].data()),
                        all_hash_bytes[i].size());
      }

      if (all_hash[rank_send] != all_hash[rank_hash]) {
        SPDLOG_INFO(
            "inconsistent check fail for tag {} from Party_{}, TTP = Party_{}",
            tag, rank_recv, rank_recv);
      } else if (all_hash[rank_send] != all_hash[rank_recv]) {
        SPDLOG_INFO(
            "inconsistent check fail for tag {} from Party_{}, TTP = Party_{}",
            tag, rank_recv, rank_hash);
      } else {
        SPDLOG_INFO(
            "inconsistent check fail for tag {} from Party_{}, TTP = Party_{}",
            tag, rank_recv, rank_send);
      }
    }
    res = res_v;
  }

  return res;
}

NdArrayRef Sharing(KernelEvalContext* ctx, const NdArrayRef& msg, size_t owner,
                   std::string_view tag) {
  auto const field = msg.eltype().as<RingTy>()->field();
  auto* comm = ctx->getState<Communicator>();
  auto* prg_state = ctx->getState<PrgState>();
  auto ty = makeType<RingTy>(field);
  auto out_ty = makeType<AShrTy>(field);
  auto rank = comm->getRank();

  NdArrayRef alpha1(ty, msg.shape());
  NdArrayRef alpha2(ty, msg.shape());
  NdArrayRef beta(ty, msg.shape());
  NdArrayRef gamma(ty, msg.shape());
  NdArrayRef beta_plus_gamma(ty, msg.shape());

  if (owner == 0) {
    // P0, Pj together sample random alpha_j
    auto [r0, r1] =
        prg_state->genPrssPair(field, msg.shape(), PrgState::GenPrssCtrl::Both);
    if (rank == 0) {
      alpha2 = r0;
      alpha1 = r1;
    }
    if (rank == 1) {
      alpha1 = r0;
    }
    if (rank == 2) {
      alpha2 = r1;
    }

    // parties sample random gamma
    auto r2 = prg_state->genPubl(field, msg.shape());
    gamma = r2;

    // P0 send beta = v + alpha to P1
    if (rank == 0) {
      beta = ring_add(msg, ring_add(alpha1, alpha2));
      comm->sendAsync(1, beta, "beta_01");
    }
    if (rank == 1) {
      beta = comm->recv(0, ty, "beta_01");
      beta = beta.reshape(msg.shape());
    }

    // P0 and P1 jmp-send beta to P2
    beta = JointMessagePassing(ctx, beta, 0, 1, 2, "beta_012");

    if (rank == 0) {
      beta_plus_gamma = ring_add(beta, gamma);
    }
  }
  if (owner == 1) {
    // P0, P1 together sample alpha1
    // P1, P2 together sample gamma
    auto [r0, r1] =
        prg_state->genPrssPair(field, msg.shape(), PrgState::GenPrssCtrl::Both);
    if (rank == 0) {
      alpha1 = r1;
    }
    if (rank == 1) {
      alpha1 = r0;
      gamma = r1;
    }
    if (rank == 2) {
      gamma = r0;
    }

    // parties sample random alpha2
    auto r2 = prg_state->genPubl(field, msg.shape());
    alpha2 = r2;

    // P1 send beta = v + alpha to P2
    if (rank == 1) {
      beta = ring_add(msg, ring_add(alpha1, alpha2));
      comm->sendAsync(2, beta, "beta_12");
    }
    if (rank == 2) {
      beta = comm->recv(1, ty, "beta_12");
      beta = beta.reshape(msg.shape());
    }

    // P1, P2 jmp-send beta + gamma to P0
    beta_plus_gamma = ring_add(beta, gamma);
    beta_plus_gamma = JointMessagePassing(ctx, beta_plus_gamma, 1, 2, 0,
                                          "beta_plus_gamma_120");
  }
  if (owner == 2) {
    // P0, P2 together sample alpha2
    // P1, P2 together sample gamma
    auto [r0, r1] =
        prg_state->genPrssPair(field, msg.shape(), PrgState::GenPrssCtrl::Both);
    if (rank == 0) {
      alpha2 = r0;
    }
    if (rank == 1) {
      gamma = r1;
    }
    if (rank == 2) {
      alpha2 = r1;
      gamma = r0;
    }

    // parties sample random alpha1
    auto r2 = prg_state->genPubl(field, msg.shape());
    alpha1 = r2;

    // P2 send beta = v + alpha to P1
    if (rank == 2) {
      beta = ring_add(msg, ring_add(alpha1, alpha2));
      comm->sendAsync(1, beta, "beta_12");
    }
    if (rank == 1) {
      beta = comm->recv(2, ty, "beta_12");
      beta = beta.reshape(msg.shape());
    }

    // P1, P2 jmp-send beta + gamma to P0
    beta_plus_gamma = ring_add(beta, gamma);
    beta_plus_gamma = JointMessagePassing(ctx, beta_plus_gamma, 2, 1, 0,
                                          "beta_plus_gamma_210");
  }

  return DISPATCH_ALL_FIELDS(field, [&]() {
    using ashr_el_t = ring2k_t;
    using ashr_t = std::array<ashr_el_t, 3>;

    NdArrayView<ashr_el_t> _alpha1(alpha1);
    NdArrayView<ashr_el_t> _alpha2(alpha2);
    NdArrayView<ashr_el_t> _beta(beta);
    NdArrayView<ashr_el_t> _gamma(gamma);
    NdArrayView<ashr_el_t> _beta_plus_gamma(beta_plus_gamma);

    NdArrayRef out(out_ty, msg.shape());
    NdArrayView<ashr_t> _out(out);

    if (rank == 0) {
      pforeach(0, msg.numel(), [&](int64_t idx) {
        _out[idx][0] = _alpha1[idx];
        _out[idx][1] = _alpha2[idx];
        _out[idx][2] = _beta_plus_gamma[idx];
      });
    }
    if (rank == 1) {
      pforeach(0, msg.numel(), [&](int64_t idx) {
        _out[idx][0] = _alpha1[idx];
        _out[idx][1] = _beta[idx];
        _out[idx][2] = _gamma[idx];
      });
    }
    if (rank == 2) {
      pforeach(0, msg.numel(), [&](int64_t idx) {
        _out[idx][0] = _alpha2[idx];
        _out[idx][1] = _beta[idx];
        _out[idx][2] = _gamma[idx];
      });
    }
    return out;
  });
}

NdArrayRef JointSharing(KernelEvalContext* ctx, const NdArrayRef& msg,
                        size_t rank_i, size_t rank_j, std::string_view tag) {
  auto const field = msg.eltype().as<RingTy>()->field();
  auto* comm = ctx->getState<Communicator>();
  auto* prg_state = ctx->getState<PrgState>();
  auto ty = makeType<RingTy>(field);
  auto out_ty = makeType<AShrTy>(field);
  auto rank = comm->getRank();

  NdArrayRef out(out_ty, msg.shape());

  return DISPATCH_ALL_FIELDS(field, [&]() {
    using ashr_el_t = ring2k_t;
    using ashr_t = std::array<ashr_el_t, 3>;

    NdArrayView<ashr_t> _out(out);
    NdArrayView<ashr_el_t> _msg(msg);

    if ((rank_i == 1 && rank_j == 2) || (rank_i == 2 && rank_j == 1)) {
      //  0   0   r
      //  0   v   r - v
      //  0   v   r - v
      auto r = prg_state->genPubl(field, msg.shape());
      auto r_v = ring_sub(r, msg);
      NdArrayView<ashr_el_t> _r_v(r_v);
      NdArrayView<ashr_el_t> _r(r);

      pforeach(0, msg.numel(), [&](int64_t idx) {
        _out[idx][0] = ring2k_t(0);
        _out[idx][1] = rank == 0 ? ring2k_t(0) : _msg[idx];
        _out[idx][2] = rank == 0 ? _r[idx] : _r_v[idx];
      });
    } else if ((rank_i == 1 && rank_j == 0) || (rank_i == 0 && rank_j == 1)) {
      //  -v   0   r
      //  -v   0   r
      //   0   0   r
      auto r = prg_state->genPubl(field, msg.shape());
      auto neg_msg = ring_neg(msg);
      NdArrayView<ashr_el_t> _neg_msg(neg_msg);
      NdArrayView<ashr_el_t> _r(r);

      pforeach(0, msg.numel(), [&](int64_t idx) {
        _out[idx][0] = rank == 2 ? ring2k_t(0) : _neg_msg[idx];
        _out[idx][1] = ring2k_t(0);
        _out[idx][2] = _r[idx];
      });
    } else if ((rank_i == 2 && rank_j == 0) || (rank_i == 0 && rank_j == 2)) {
      //   0   -v   r
      //   0    0   r
      //  -v    0   r
      auto r = prg_state->genPubl(field, msg.shape());
      auto neg_msg = ring_neg(msg);
      NdArrayView<ashr_el_t> _neg_msg(neg_msg);
      NdArrayView<ashr_el_t> _r(r);

      pforeach(0, msg.numel(), [&](int64_t idx) {
        _out[idx][0] = rank == 2 ? _neg_msg[idx] : ring2k_t(0);
        _out[idx][1] = rank == 0 ? _neg_msg[idx] : ring2k_t(0);
        _out[idx][2] = _r[idx];
      });
    } else {
      SPU_THROW("Party idx wrong in Joint Sharing");
    }
    return out;
  });
}

NdArrayRef P2A::proc(KernelEvalContext* ctx, const NdArrayRef& in) const {
  auto* comm = ctx->getState<Communicator>();

  const auto* in_ty = in.eltype().as<Pub2kTy>();
  const auto field = in_ty->field();

  auto rank = comm->getRank();

  return DISPATCH_ALL_FIELDS(field, [&]() {
    using ashr_el_t = ring2k_t;
    using ashr_t = std::array<ashr_el_t, 3>;
    using pshr_el_t = ring2k_t;

    NdArrayRef out(makeType<AShrTy>(field), in.shape());
    NdArrayView<ashr_t> _out(out);
    NdArrayView<pshr_el_t> _in(in);

    // 0, 0, v
    // 0, v, 0
    // 0, v, 0

    pforeach(0, in.numel(), [&](int64_t idx) {
      _out[idx][0] = ring2k_t(0);
      _out[idx][1] = rank == 0 ? ring2k_t(0) : _in[idx];
      _out[idx][2] = rank == 0 ? _in[idx] : ring2k_t(0);
    });

    return out;
  });
}

NdArrayRef A2P::proc(KernelEvalContext* ctx, const NdArrayRef& in) const {
  auto* comm = ctx->getState<Communicator>();
  const auto field = in.eltype().as<RingTy>()->field();
  auto numel = in.numel();
  auto rank = comm->getRank();
  auto ty = makeType<RingTy>(field);

  return DISPATCH_ALL_FIELDS(field, [&] {
    using pshr_el_t = ring2k_t;
    using ashr_el_t = ring2k_t;
    using ashr_t = std::array<ashr_el_t, 3>;

    NdArrayRef out(makeType<Pub2kTy>(field), in.shape());

    NdArrayView<pshr_el_t> _out(out);
    NdArrayView<ashr_t> _in(in);

    NdArrayRef alpha1(ty, in.shape());
    NdArrayRef alpha2(ty, in.shape());
    NdArrayRef beta(ty, in.shape());

    if (rank == 0) {
      alpha1 = getFirstShare(in);
      alpha2 = getSecondShare(in);
    }
    if (rank == 1) {
      alpha1 = getFirstShare(in);
      beta = getSecondShare(in);
    }
    if (rank == 2) {
      alpha2 = getFirstShare(in);
      beta = getSecondShare(in);
    }

    // P1, P2 -> P0 : beta
    // P0, P1 -> P2 : alpha1
    // P2, P0 -> P1 : alpha2
    beta = JointMessagePassing(ctx, beta, 1, 2, 0, "beta");
    alpha1 = JointMessagePassing(ctx, alpha1, 0, 1, 2, "alpha1");
    alpha2 = JointMessagePassing(ctx, alpha2, 2, 0, 1, "alpha2");

    NdArrayView<ashr_el_t> _alpha1(alpha1);
    NdArrayView<ashr_el_t> _alpha2(alpha2);
    NdArrayView<ashr_el_t> _beta(beta);

    pforeach(0, numel, [&](int64_t idx) {
      _out[idx] = _beta[idx] - _alpha1[idx] - _alpha2[idx];
    });
    return out;
  });
}

NdArrayRef A2V::proc(KernelEvalContext* ctx, const NdArrayRef& in,
                     size_t rank_dst) const {
  const auto field = in.eltype().as<AShrTy>()->field();
  auto* comm = ctx->getState<Communicator>();
  auto rank = comm->getRank();

  return DISPATCH_ALL_FIELDS(field, [&]() {
    using vshr_el_t = ring2k_t;
    using ashr_el_t = ring2k_t;
    using ashr_t = std::array<ashr_el_t, 3>;

    NdArrayView<ashr_t> _in(in);
    auto out_ty = makeType<Priv2kTy>(field, rank_dst);
    auto ty = makeType<RingTy>(field);

    NdArrayRef alpha1(ty, in.shape());
    NdArrayRef alpha2(ty, in.shape());
    NdArrayRef beta(ty, in.shape());

    if (rank == 0) {
      alpha1 = getFirstShare(in);
      alpha2 = getSecondShare(in);
    }
    if (rank == 1) {
      alpha1 = getFirstShare(in);
      beta = getSecondShare(in);
    }
    if (rank == 2) {
      alpha2 = getFirstShare(in);
      beta = getSecondShare(in);
    }

    if (rank_dst == 0) {
      beta = JointMessagePassing(ctx, beta, 1, 2, 0, "beta");
    }
    if (rank_dst == 1) {
      alpha2 = JointMessagePassing(ctx, alpha2, 2, 0, 1, "alpha2");
    }
    if (rank_dst == 2) {
      alpha1 = JointMessagePassing(ctx, alpha1, 0, 1, 2, "alpha1");
    }

    if (rank == rank_dst) {
      NdArrayView<ashr_el_t> _alpha1(alpha1);
      NdArrayView<ashr_el_t> _alpha2(alpha2);
      NdArrayView<ashr_el_t> _beta(beta);

      NdArrayRef out(out_ty, in.shape());
      NdArrayView<vshr_el_t> _out(out);

      pforeach(0, in.numel(), [&](int64_t idx) {
        _out[idx] = _beta[idx] - _alpha1[idx] - _alpha2[idx];
      });
      return out;
    } else {
      return makeConstantArrayRef(out_ty, in.shape());
    }
  });
}

NdArrayRef V2A::proc(KernelEvalContext* ctx, const NdArrayRef& in) const {
  const auto* in_ty = in.eltype().as<Priv2kTy>();
  const auto field = in_ty->field();

  size_t owner_rank = in_ty->owner();

  auto tmp = in.as(makeType<RingTy>(field));

  auto res = Sharing(ctx, tmp, owner_rank, "v2a");

  return res;
}

NdArrayRef NegateA::proc(KernelEvalContext* ctx, const NdArrayRef& in) const {
  const auto* in_ty = in.eltype().as<AShrTy>();
  const auto field = in_ty->field();

  return DISPATCH_ALL_FIELDS(field, [&]() {
    using el_t = std::make_unsigned_t<ring2k_t>;
    using shr_t = std::array<el_t, 3>;

    NdArrayRef out(makeType<AShrTy>(field), in.shape());
    NdArrayView<shr_t> _out(out);
    NdArrayView<shr_t> _in(in);

    pforeach(0, in.numel(), [&](int64_t idx) {
      _out[idx][0] = -_in[idx][0];
      _out[idx][1] = -_in[idx][1];
      _out[idx][2] = -_in[idx][2];
    });

    return out;
  });
}

NdArrayRef RandA::proc(KernelEvalContext* ctx, const Shape& shape) const {
  auto* prg_state = ctx->getState<PrgState>();
  auto* comm = ctx->getState<Communicator>();
  const auto field = ctx->getState<Z2kState>()->getDefaultField();
  auto ty = makeType<RingTy>(field);
  auto rank = comm->getRank();

  NdArrayRef alpha1(ty, shape);
  NdArrayRef alpha2(ty, shape);
  NdArrayRef beta(ty, shape);
  NdArrayRef gamma(ty, shape);

  NdArrayRef out(makeType<AShrTy>(field), shape);

  // Comparison only works for [-2^(k-2), 2^(k-2)]
  auto [r0, r1] =
      prg_state->genPrssPair(field, shape, PrgState::GenPrssCtrl::Both);
  auto [r2, r3] =
      prg_state->genPrssPair(field, shape, PrgState::GenPrssCtrl::Both);

  r0 = ring_rshift(r0, {2});
  r1 = ring_rshift(r1, {2});

  if (rank == 0) {
    alpha2 = r0;
    alpha1 = r1;
  }
  if (rank == 1) {
    alpha1 = r0;
    beta = r1;
    gamma = ring_rshift(r3, {2});
  }
  if (rank == 2) {
    alpha2 = r1;
    beta = r0;
    gamma = ring_rshift(r2, {2});
  }

  auto beta_plus_gamma = ring_add(beta, gamma);
  beta_plus_gamma =
      JointMessagePassing(ctx, beta_plus_gamma, 1, 2, 0, "beta_plus_gamma_120");

  DISPATCH_ALL_FIELDS(field, [&]() {
    using el_t = ring2k_t;
    using ashr_t = std::array<el_t, 3>;
    NdArrayView<el_t> _alpha1(alpha1);
    NdArrayView<el_t> _alpha2(alpha2);
    NdArrayView<el_t> _beta(beta);
    NdArrayView<el_t> _gamma(gamma);
    NdArrayView<el_t> _beta_plus_gamma(beta_plus_gamma);

    NdArrayView<ashr_t> _out(out);
    pforeach(0, out.numel(), [&](int64_t idx) {
      _out[idx][0] = rank == 2 ? _alpha2[idx] : _alpha1[idx];
      _out[idx][1] = rank == 0 ? _alpha2[idx] : _beta[idx];
      _out[idx][2] = rank == 0 ? _beta_plus_gamma[idx] : _gamma[idx];
    });
  });
  return out;
}

////////////////////////////////////////////////////////////////////
// add family
////////////////////////////////////////////////////////////////////
NdArrayRef AddAP::proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                       const NdArrayRef& rhs) const {
  auto* comm = ctx->getState<Communicator>();
  const auto* lhs_ty = lhs.eltype().as<AShrTy>();
  const auto* rhs_ty = rhs.eltype().as<Pub2kTy>();

  SPU_ENFORCE(lhs_ty->field() == rhs_ty->field());
  const auto field = lhs_ty->field();

  auto rank = comm->getRank();

  return DISPATCH_ALL_FIELDS(field, [&]() {
    using el_t = ring2k_t;
    using ashr_t = std::array<el_t, 3>;

    NdArrayRef out(makeType<AShrTy>(field), lhs.shape());
    NdArrayView<ashr_t> _out(out);
    NdArrayView<ashr_t> _lhs(lhs);
    NdArrayView<el_t> _rhs(rhs);

    pforeach(0, lhs.numel(), [&](int64_t idx) {
      _out[idx][0] = _lhs[idx][0];
      _out[idx][1] = _lhs[idx][1];
      _out[idx][2] = _lhs[idx][2];
      if (rank == 0) _out[idx][2] += _rhs[idx];
      if (rank == 1 || rank == 2) _out[idx][1] += _rhs[idx];
    });
    return out;
  });
}

NdArrayRef AddAA::proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                       const NdArrayRef& rhs) const {
  const auto* lhs_ty = lhs.eltype().as<AShrTy>();
  const auto* rhs_ty = rhs.eltype().as<AShrTy>();

  SPU_ENFORCE(lhs_ty->field() == rhs_ty->field());
  const auto field = lhs_ty->field();

  return DISPATCH_ALL_FIELDS(field, [&]() {
    using ashr_t = std::array<ring2k_t, 3>;

    NdArrayRef out(makeType<AShrTy>(field), lhs.shape());
    NdArrayView<ashr_t> _out(out);
    NdArrayView<ashr_t> _lhs(lhs);
    NdArrayView<ashr_t> _rhs(rhs);

    pforeach(0, lhs.numel(), [&](int64_t idx) {
      _out[idx][0] = _lhs[idx][0] + _rhs[idx][0];
      _out[idx][1] = _lhs[idx][1] + _rhs[idx][1];
      _out[idx][2] = _lhs[idx][2] + _rhs[idx][2];
    });
    return out;
  });
}

////////////////////////////////////////////////////////////////////
// multiply family
////////////////////////////////////////////////////////////////////
NdArrayRef MulAP::proc(KernelEvalContext*, const NdArrayRef& lhs,
                       const NdArrayRef& rhs) const {
  const auto* lhs_ty = lhs.eltype().as<AShrTy>();
  const auto* rhs_ty = rhs.eltype().as<Pub2kTy>();

  SPU_ENFORCE(lhs_ty->field() == rhs_ty->field());
  const auto field = lhs_ty->field();

  return DISPATCH_ALL_FIELDS(field, [&]() {
    using el_t = ring2k_t;
    using ashr_t = std::array<el_t, 3>;

    NdArrayRef out(makeType<AShrTy>(field), lhs.shape());
    NdArrayView<ashr_t> _out(out);
    NdArrayView<ashr_t> _lhs(lhs);
    NdArrayView<el_t> _rhs(rhs);

    pforeach(0, lhs.numel(), [&](int64_t idx) {
      _out[idx][0] = _lhs[idx][0] * _rhs[idx];
      _out[idx][1] = _lhs[idx][1] * _rhs[idx];
      _out[idx][2] = _lhs[idx][2] * _rhs[idx];
    });
    return out;
  });
}

NdArrayRef MulAA_semi::proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                            const NdArrayRef& rhs) const {
  // Debug only
  const auto field = lhs.eltype().as<RingTy>()->field();
  auto* comm = ctx->getState<Communicator>();
  auto* prg_state = ctx->getState<PrgState>();
  auto rank = comm->getRank();
  auto ty = makeType<RingTy>(field);
  auto shape = lhs.shape();

  NdArrayRef alpha1(ty, shape);
  NdArrayRef alpha2(ty, shape);
  NdArrayRef beta_z(ty, shape);
  NdArrayRef out(makeType<AShrTy>(field), shape);

  // P0, Pj together sample random alpha_j
  auto [r0, r1] =
      prg_state->genPrssPair(field, lhs.shape(), PrgState::GenPrssCtrl::Both);
  if (rank == 0) {
    alpha2 = r0;
    alpha1 = r1;
  }
  if (rank == 1) {
    alpha1 = r0;
  }
  if (rank == 2) {
    alpha2 = r1;
  }

  if (rank == 0) {
    auto alpha_x1 = getFirstShare(lhs);
    auto alpha_x2 = getSecondShare(rhs);
    auto alpha_x = ring_add(alpha_x1, alpha_x2);

    auto alpha_y1 = getFirstShare(lhs);
    auto alpha_y2 = getSecondShare(rhs);
    auto alpha_y = ring_add(alpha_y1, alpha_y2);
    auto Gamma = ring_mul(alpha_x, alpha_y);
    auto Gammas = ring_rand_additive_splits(Gamma, 2);
    comm->sendAsync(1, Gammas[0], "Gamma_i");
    comm->sendAsync(2, Gammas[1], "Gamma_i");
  }
  if (rank == 1 || rank == 2) {
    auto Gamma = comm->recv(0, ty, "Gamma_i");
    auto alpha_xi = getFirstShare(lhs);
    auto alpha_yi = getFirstShare(rhs);
    auto beta_x = getSecondShare(lhs);
    auto beta_y = getSecondShare(rhs);

    auto beta_zi =
        rank == 2 ? ring_mul(beta_x, beta_y) : ring_zeros(field, shape);
    ring_sub_(beta_zi, ring_mul(beta_x, alpha_yi));
    ring_sub_(beta_zi, ring_mul(beta_y, alpha_xi));
    ring_add_(beta_zi, Gamma);
    if (rank == 1) {
      ring_add_(beta_zi, alpha1);
    }
    if (rank == 2) {
      ring_add_(beta_zi, alpha2);
    }

    comm->sendAsync((3 - rank), beta_zi, "beta_zi");
    auto beta_zi_ = comm->recv((3 - rank), ty, "beta_zi");
    beta_z = ring_add(beta_zi, beta_zi_);
  }

  return DISPATCH_ALL_FIELDS(field, [&]() {
    using el_t = ring2k_t;
    using ashr_t = std::array<el_t, 3>;

    NdArrayView<ashr_t> _out(out);
    NdArrayView<el_t> _alpha1(alpha1);
    NdArrayView<el_t> _alpha2(alpha2);
    NdArrayView<el_t> _beta_z(beta_z);

    pforeach(0, rhs.numel(), [&](int64_t idx) {
      _out[idx][0] = rank == 2 ? _alpha2[idx] : _alpha1[idx];
      _out[idx][1] = rank == 0 ? _alpha2[idx] : _beta_z[idx];
    });

    return out;
  });
}

NdArrayRef RandA_RSS(KernelEvalContext* ctx, const Shape& shape,
                     const FieldType field) {
  // semi-honest RandA for RSS
  // store the shares like RSS
  // P0 : x0  x1  dummy
  // P1 : x1  x2  dummy
  // P2 : x2  x0  dummy
  auto* prg_state = ctx->getState<PrgState>();

  NdArrayRef out(makeType<AShrTy>(field), shape);

  DISPATCH_ALL_FIELDS(field, [&]() {
    using el_t = ring2k_t;

    std::vector<el_t> r0(shape.numel());
    std::vector<el_t> r1(shape.numel());
    prg_state->fillPrssPair(r0.data(), r1.data(), r0.size(),
                            PrgState::GenPrssCtrl::Both);

    NdArrayView<std::array<el_t, 3>> _out(out);

    pforeach(0, out.numel(), [&](int64_t idx) {
      // Comparison only works for [-2^(k-2), 2^(k-2)).
      // TODO: Move this constraint to upper layer, saturate it here.
      _out[idx][0] = r0[idx] >> 2;
      _out[idx][1] = r1[idx] >> 2;
      _out[idx][2] = el_t(0);
    });
  });

  return out;
}

NdArrayRef RSS_A2P(KernelEvalContext* ctx, const NdArrayRef& in,
                   std::string_view tag) {
  auto* comm = ctx->getState<Communicator>();
  const auto field = in.eltype().as<AShrTy>()->field();
  auto numel = in.numel();
  // auto rank = comm->getRank();
  // std::cout << "Party " << rank << " numel = " << numel << std::endl;

  return DISPATCH_ALL_FIELDS(field, [&]() {
    using pshr_el_t = ring2k_t;
    using ashr_el_t = ring2k_t;
    using ashr_t = std::array<ashr_el_t, 3>;

    NdArrayRef out(makeType<Pub2kTy>(field), in.shape());
    NdArrayView<pshr_el_t> _out(out);
    NdArrayView<ashr_t> _in(in);

    std::vector<ashr_el_t> x2(numel);

    pforeach(0, numel, [&](int64_t idx) { x2[idx] = _in[idx][1]; });

    auto x3 = comm->rotate<ashr_el_t>(x2, tag);  // comm => 1, k

    pforeach(0, numel, [&](int64_t idx) {
      _out[idx] = _in[idx][0] + _in[idx][1] + x3[idx];
    });

    return out;
  });
}

NdArrayRef RssMul_semi(KernelEvalContext* ctx, const NdArrayRef& lhs,
                       const NdArrayRef& rhs, std::string_view tag) {
  // semi-honest mult for RSS
  // store the shares like RSS
  // P0 : x0  x1  dummy
  // P1 : x1  x2  dummy
  // P2 : x2  x0  dummy
  const auto field = lhs.eltype().as<Ring2k>()->field();
  auto* comm = ctx->getState<Communicator>();
  auto* prg_state = ctx->getState<PrgState>();

  return DISPATCH_ALL_FIELDS(field, [&]() {
    using el_t = ring2k_t;
    using shr_t = std::array<el_t, 3>;

    std::vector<el_t> r0(lhs.numel());
    std::vector<el_t> r1(lhs.numel());

    prg_state->fillPrssPair(r0.data(), r1.data(), r0.size(),
                            PrgState::GenPrssCtrl::Both);

    NdArrayView<shr_t> _lhs(lhs);
    NdArrayView<shr_t> _rhs(rhs);

    // z1 = (x1 * y1) + (x1 * y2) + (x2 * y1) + (r0 - r1);
    pforeach(0, lhs.numel(), [&](int64_t idx) {
      r0[idx] = (_lhs[idx][0] * _rhs[idx][0]) + (_lhs[idx][0] * _rhs[idx][1]) +
                (_lhs[idx][1] * _rhs[idx][0]) + (r0[idx] - r1[idx]);
    });

    r1 = comm->rotate<el_t>(r0, tag);  // comm => 1, k

    NdArrayRef out(makeType<AShrTy>(field), lhs.shape());
    NdArrayView<shr_t> _out(out);

    pforeach(0, lhs.numel(), [&](int64_t idx) {
      _out[idx][0] = r0[idx];
      _out[idx][1] = r1[idx];
    });

    return out;
  });
}

// extend each element from FieldType_in to FieldType_out
// only for AShrTy
NdArrayRef RingChange(KernelEvalContext* ctx, const NdArrayRef& in,
                      FieldType fieldType_in, FieldType fieldType_out,
                      bool isAShrTy) {
  auto numel = in.numel();
  NdArrayRef res(makeType<AShrTy>(fieldType_out), in.shape());
  if (isAShrTy) {
    DISPATCH_ALL_FIELDS(fieldType_out, [&]() {
      using ashr_t = std::array<ring2k_t, 3>;
      NdArrayView<ashr_t> _res(res);
      if (fieldType_in == FieldType::FM32) {
        NdArrayView<std::array<uint32_t, 3>> _in(in);
        pforeach(0, numel, [&](int64_t idx) {
          _res[idx][0] = static_cast<ring2k_t>(_in[idx][0]);
          _res[idx][1] = static_cast<ring2k_t>(_in[idx][1]);
          _res[idx][2] = static_cast<ring2k_t>(_in[idx][2]);
        });
      } else if (fieldType_in == FieldType::FM64) {
        NdArrayView<std::array<uint64_t, 3>> _in(in);
        pforeach(0, numel, [&](int64_t idx) {
          _res[idx][0] = static_cast<ring2k_t>(_in[idx][0]);
          _res[idx][1] = static_cast<ring2k_t>(_in[idx][1]);
          _res[idx][2] = static_cast<ring2k_t>(_in[idx][2]);
        });
      } else if (fieldType_in == FieldType::FM128) {
        NdArrayView<std::array<uint128_t, 3>> _in(in);
        pforeach(0, numel, [&](int64_t idx) {
          _res[idx][0] = static_cast<ring2k_t>(_in[idx][0]);
          _res[idx][1] = static_cast<ring2k_t>(_in[idx][1]);
          _res[idx][2] = static_cast<ring2k_t>(_in[idx][2]);
        });
      } else {
        SPU_THROW("error FieldType");
      }
    });
  } else {
    // RingTy
    DISPATCH_ALL_FIELDS(fieldType_out, [&]() {
      using ashr_t = ring2k_t;
      NdArrayView<ashr_t> _res(res);
      if (fieldType_in == FieldType::FM32) {
        NdArrayView<uint32_t> _in(in);
        pforeach(0, numel, [&](int64_t idx) {
          _res[idx] = static_cast<ring2k_t>(_in[idx]);
        });
      } else if (fieldType_in == FieldType::FM64) {
        NdArrayView<uint64_t> _in(in);
        pforeach(0, numel, [&](int64_t idx) {
          _res[idx] = static_cast<ring2k_t>(_in[idx]);
        });
      } else if (fieldType_in == FieldType::FM128) {
        NdArrayView<uint128_t> _in(in);
        pforeach(0, numel, [&](int64_t idx) {
          _res[idx] = static_cast<ring2k_t>(_in[idx]);
        });
      } else {
        SPU_THROW("error FieldType");
      }
    });
  }

  return res;
}

// The functionality of MulPre is a malicious multiplication protocol for RSS
// refer to https://eprint.iacr.org/2023/1744
NdArrayRef MulPre(KernelEvalContext* ctx, const NdArrayRef& lhs,
                  const NdArrayRef& rhs) {
  const auto field = lhs.eltype().as<Ring2k>()->field();
  auto* prg_state = ctx->getState<PrgState>();
  auto numel = lhs.numel();
  // if (!(field == FieldType::FM32 || field == FieldType::FM64)) {
  //   SPU_THROW("malicious MulPre in MulAA only support FM32 and FM64");
  // }

  FieldType field_sigma_plus_k, field_sigma_mask;

  if (field == FieldType::FM32) {
    field_sigma_plus_k = FM64;
    field_sigma_mask = FM32;
  } else if (field == FieldType::FM64) {
    field_sigma_plus_k = FM128;
    field_sigma_mask = FM64;
  } else if (field == FieldType::FM128) {
    field_sigma_plus_k = FM128;
    field_sigma_mask = FM128;
  } else {
    SPU_THROW("error FieldType");
  }

  // 1. calculate lsh * rhs in the semi-honest setting
  auto res = RssMul_semi(ctx, lhs, rhs, "mul: lhs * rhs");

  // 2. generate potentially incorrect triple (a,b,c)
  SPU_ENFORCE(lhs.shape() == rhs.shape());
  auto a = RandA_RSS(ctx, lhs.shape(), field_sigma_plus_k);
  auto b = RingChange(ctx, rhs, field, field_sigma_plus_k, true);
  auto c = RssMul_semi(ctx, a, b, "mul: a * b");

  // 3. sample r in \mathbb{Z}_{2^{sigma}}
  auto r = prg_state->genPubl(field_sigma_mask, lhs.shape());
  r = RingChange(ctx, r, field_sigma_mask, field_sigma_plus_k, false);
  auto v = NdArrayRef(makeType<AShrTy>(field_sigma_plus_k), r.shape());
  auto w = NdArrayRef(makeType<AShrTy>(field_sigma_plus_k), r.shape());

  auto lhs_extend = RingChange(ctx, lhs, field, field_sigma_plus_k, true);
  auto res_extend = RingChange(ctx, res, field, field_sigma_plus_k, true);

  DISPATCH_ALL_FIELDS(field_sigma_plus_k, [&]() {
    using ashr_t = std::array<ring2k_t, 3>;

    NdArrayView<ring2k_t> _r(r);
    NdArrayView<ashr_t> _lhs_extend(lhs_extend);
    NdArrayView<ashr_t> _a(a);
    NdArrayView<ashr_t> _v(v);

    pforeach(0, numel, [&](int64_t idx) {
      _v[idx][0] = _r[idx] * _lhs_extend[idx][0] - _a[idx][0];
      _v[idx][1] = _r[idx] * _lhs_extend[idx][1] - _a[idx][1];
      _v[idx][2] = _r[idx] * _lhs_extend[idx][2] - _a[idx][2];
    });

    auto reconstruct_v = RSS_A2P(ctx, v, "reconstruct v in MulPre");

    NdArrayView<ring2k_t> _reconstruct_v(reconstruct_v);
    NdArrayView<ashr_t> _w(w);
    NdArrayView<ashr_t> _b(b);
    NdArrayView<ashr_t> _res_extend(res_extend);
    NdArrayView<ashr_t> _c(c);
    pforeach(0, numel, [&](int64_t idx) {
      _w[idx][0] = _reconstruct_v[idx] * _b[idx][0] -
                   _r[idx] * _res_extend[idx][0] + _c[idx][0];
      _w[idx][1] = _reconstruct_v[idx] * _b[idx][1] -
                   _r[idx] * _res_extend[idx][1] + _c[idx][1];
      _w[idx][2] = _reconstruct_v[idx] * _b[idx][2] -
                   _r[idx] * _res_extend[idx][2] + _c[idx][2];
    });

    auto reconstruct_w = RSS_A2P(ctx, w, "reconstruct w in MulPre");
    // We need to change w from field_sigma_plus_k to field,
    // otherwise, when the shares are overflow (for example: c = c0 + c1 + c2 -
    // 2^l) r*(c-a*b) = r*(c + 2^l - a*b) != (\hat{c} - \hat{a}*b) mod
    // 2^{l+sigma}
    reconstruct_w =
        RingChange(ctx, reconstruct_w, field_sigma_plus_k, field, false);
    auto zeros = ring_zeros(field, lhs.shape());
    SPU_ENFORCE(ring_all_equal(zeros, reconstruct_w), "malicious in MulPre");
  });

  return res;
}

NdArrayRef MulAA::proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                       const NdArrayRef& rhs) const {
  const auto field = lhs.eltype().as<RingTy>()->field();
  auto* comm = ctx->getState<Communicator>();
  auto* prg_state = ctx->getState<PrgState>();
  auto rank = comm->getRank();
  auto ty = makeType<RingTy>(field);
  auto shape = lhs.shape();
  auto numel = lhs.numel();

  NdArrayRef alpha_z1(ty, shape);
  NdArrayRef alpha_z2(ty, shape);
  NdArrayRef gamma_z(ty, shape);
  NdArrayRef out(makeType<AShrTy>(field), shape);
  NdArrayRef d(makeType<AShrTy>(field), shape);
  NdArrayRef e(makeType<AShrTy>(field), shape);

  NdArrayRef chi_1(ty, shape);
  NdArrayRef chi_2(ty, shape);
  NdArrayRef Phi(ty, shape);

  NdArrayRef beta_z1_start(ty, shape);
  NdArrayRef beta_z2_start(ty, shape);

  NdArrayRef beta_plus_gamma_z(ty, shape);

  // P0, Pj together sample random alpha_j
  auto [r0, r1] =
      prg_state->genPrssPair(field, lhs.shape(), PrgState::GenPrssCtrl::Both);
  if (rank == 0) {
    alpha_z2 = r0;
    alpha_z1 = r1;
  }
  if (rank == 1) {
    alpha_z1 = r0;
    gamma_z = r1;
  }
  if (rank == 2) {
    alpha_z2 = r1;
    gamma_z = r0;
  }

  return DISPATCH_ALL_FIELDS(field, [&]() {
    using el_t = ring2k_t;
    using shr_t = std::array<el_t, 3>;

    NdArrayView<shr_t> _d(d);
    NdArrayView<shr_t> _e(e);
    NdArrayView<shr_t> _out(out);
    NdArrayView<shr_t> _lhs(lhs);
    NdArrayView<shr_t> _rhs(rhs);
    NdArrayView<el_t> _alpha_z1(alpha_z1);
    NdArrayView<el_t> _alpha_z2(alpha_z2);
    NdArrayView<el_t> _gamma_z(gamma_z);

    // generate RSS of e, d
    // refer to Table 3 in Swift
    // and init out
    if (rank == 0) {
      pforeach(0, numel, [&](int64_t idx) {
        _d[idx][0] = _lhs[idx][1];
        _d[idx][1] = _lhs[idx][0];
        _e[idx][0] = _rhs[idx][1];
        _e[idx][1] = _rhs[idx][0];

        _out[idx][0] = _alpha_z1[idx];
        _out[idx][1] = _alpha_z2[idx];
      });
    } else if (rank == 1) {
      pforeach(0, numel, [&](int64_t idx) {
        _d[idx][0] = _lhs[idx][0];
        _d[idx][1] = _lhs[idx][2];
        _e[idx][0] = _rhs[idx][0];
        _e[idx][1] = _rhs[idx][2];

        _out[idx][0] = _alpha_z1[idx];
        _out[idx][2] = _gamma_z[idx];
      });
    } else if (rank == 2) {
      pforeach(0, numel, [&](int64_t idx) {
        _d[idx][0] = _lhs[idx][2];
        _d[idx][1] = _lhs[idx][0];
        _e[idx][0] = _rhs[idx][2];
        _e[idx][1] = _rhs[idx][0];

        _out[idx][0] = _alpha_z2[idx];
        _out[idx][2] = _gamma_z[idx];
      });
    }

    // p0, p1 : chi_1 = f1
    // p0, p2 : chi_2 = f0
    // p1, p2 : Phi = f2 - gamma_x * gamma_y
    auto f = MulPre(ctx, d, e);

    NdArrayView<shr_t> _f(f);
    NdArrayView<el_t> _chi_1(chi_1);
    NdArrayView<el_t> _chi_2(chi_2);
    NdArrayView<el_t> _Phi(Phi);
    if (rank == 0) {
      pforeach(0, numel, [&](int64_t idx) {
        _chi_1[idx] = _f[idx][1];
        _chi_2[idx] = _f[idx][0];
      });
    } else if (rank == 1) {
      pforeach(0, numel, [&](int64_t idx) {
        _chi_1[idx] = _f[idx][0];
        _Phi[idx] = _f[idx][1] - _lhs[idx][2] * _rhs[idx][2];
      });
    } else if (rank == 2) {
      pforeach(0, numel, [&](int64_t idx) {
        _chi_2[idx] = _f[idx][1];
        _Phi[idx] = _f[idx][0] - _lhs[idx][2] * _rhs[idx][2];
      });
    }

    NdArrayView<el_t> _beta_z1_start(beta_z1_start);
    NdArrayView<el_t> _beta_z2_start(beta_z2_start);
    // [beta*_z] = -(beta_x + gamma_x)[alpha_y] - (beta_y + gamma_y)[alpha_x]
    //             +[alpha_z] + [chi]
    if (rank == 0) {
      pforeach(0, numel, [&](int64_t idx) {
        _beta_z1_start[idx] = -_lhs[idx][2] * _rhs[idx][0] -
                              _rhs[idx][2] * _lhs[idx][0] + _alpha_z1[idx] +
                              _chi_1[idx];
        _beta_z2_start[idx] = -_lhs[idx][2] * _rhs[idx][1] -
                              _rhs[idx][2] * _lhs[idx][1] + _alpha_z2[idx] +
                              _chi_2[idx];
      });
    } else if (rank == 1) {
      pforeach(0, numel, [&](int64_t idx) {
        _beta_z1_start[idx] = -(_lhs[idx][1] + _lhs[idx][2]) * _rhs[idx][0] -
                              (_rhs[idx][1] + _rhs[idx][2]) * _lhs[idx][0] +
                              _alpha_z1[idx] + _chi_1[idx];
      });
    } else if (rank == 2) {
      pforeach(0, numel, [&](int64_t idx) {
        _beta_z2_start[idx] = -(_lhs[idx][1] + _lhs[idx][2]) * _rhs[idx][0] -
                              (_rhs[idx][1] + _rhs[idx][2]) * _lhs[idx][0] +
                              _alpha_z2[idx] + _chi_2[idx];
      });
    }

    beta_z1_start =
        JointMessagePassing(ctx, beta_z1_start, 0, 1, 2, "beta_z1_start");
    beta_z2_start =
        JointMessagePassing(ctx, beta_z2_start, 0, 2, 1, "beta_z2_start");
    auto beta_z_start = ring_add(beta_z1_start, beta_z2_start);

    NdArrayView<el_t> _beta_z_start(beta_z_start);
    NdArrayView<el_t> _beta_plus_gamma_z(beta_plus_gamma_z);
    if (rank == 1 || rank == 2) {
      pforeach(0, numel, [&](int64_t idx) {
        // beta_z = beta*_z + beta_x * beta_y + Phi
        _out[idx][1] =
            _beta_z_start[idx] + _lhs[idx][1] * _rhs[idx][1] + _Phi[idx];

        _beta_plus_gamma_z[idx] = _out[idx][1] + _out[idx][2];
      });
    }
    beta_plus_gamma_z = JointMessagePassing(ctx, beta_plus_gamma_z, 1, 2, 0,
                                            "beta_plus_gamma_z");
    if (rank == 0) {
      pforeach(0, numel,
               [&](int64_t idx) { _out[idx][2] = _beta_plus_gamma_z[idx]; });
    }

    return out;
  });
}

////////////////////////////////////////////////////////////////////
// matmul family
////////////////////////////////////////////////////////////////////
NdArrayRef MatMulAP::proc(KernelEvalContext*, const NdArrayRef& x,
                          const NdArrayRef& y) const {
  const auto field = x.eltype().as<Ring2k>()->field();

  NdArrayRef z(makeType<AShrTy>(field), {x.shape()[0], y.shape()[1]});

  auto x1 = getFirstShare(x);
  auto x2 = getSecondShare(x);
  auto x3 = getThirdShare(x);

  auto z1 = getFirstShare(z);
  auto z2 = getSecondShare(z);
  auto z3 = getThirdShare(z);

  ring_mmul_(z1, x1, y);
  ring_mmul_(z2, x2, y);
  ring_mmul_(z3, x3, y);

  return z;
}

NdArrayRef MatRssMul_semi(KernelEvalContext* ctx, const NdArrayRef& lhs,
                          const NdArrayRef& rhs, std::string_view tag) {
  // semi-honest mult based on RSS
  // store the shares like RSS
  // P0 : x0  x1  dummy
  // P1 : x1  x2  dummy
  // P2 : x2  x0  dummy
  const auto field = lhs.eltype().as<Ring2k>()->field();
  auto* comm = ctx->getState<Communicator>();
  auto* prg_state = ctx->getState<PrgState>();

  auto M = lhs.shape()[0];
  auto N = rhs.shape()[1];

  auto [r0, r1] =
      prg_state->genPrssPair(field, {M, N}, PrgState::GenPrssCtrl::Both);

  NdArrayRef out(makeType<AShrTy>(field), {M, N});
  auto o1 = getFirstShare(out);
  auto o2 = getSecondShare(out);

  auto x1 = getFirstShare(lhs);
  auto x2 = getSecondShare(lhs);

  auto y1 = getFirstShare(rhs);
  auto y2 = getSecondShare(rhs);

  // o2 = (x1 * y1) + (x1 * y2) + (x2 * y1) + (r0 - r1);
  auto t1 = ring_mmul(x1, y1);
  auto t2 = ring_mmul(x1, y2);
  auto t3 = ring_mmul(x2, y1);
  auto tmp1 = ring_sum({t1, t2, t3});

  auto tmp2 = comm->rotate(tmp1, tag);

  ring_assign(o1, tmp1);
  ring_assign(o2, tmp2);

  return out;
}

// matrix version of MulPre
NdArrayRef MatMulPre(KernelEvalContext* ctx, const NdArrayRef& lhs,
                     const NdArrayRef& rhs) {
  const auto field = lhs.eltype().as<Ring2k>()->field();
  auto* prg_state = ctx->getState<PrgState>();
  auto M = lhs.shape()[0];
  auto N = rhs.shape()[1];
  // if (!(field == FieldType::FM32 || field == FieldType::FM64)) {
  //   SPU_THROW("malicious MulPre only support FM32 and FM64");
  // }

  FieldType field_sigma_plus_k, field_sigma_mask;

  if (field == FieldType::FM32) {
    field_sigma_plus_k = FM64;
    field_sigma_mask = FM32;
  } else if (field == FieldType::FM64) {
    field_sigma_plus_k = FM128;
    field_sigma_mask = FM64;
  } else if (field == FieldType::FM128) {
    field_sigma_plus_k = FM128;
    field_sigma_mask = FM128;
  } else {
    SPU_THROW("error FieldType");
  }

  // 1. calculate lsh * rhs in the semi-honest setting
  auto res = MatRssMul_semi(ctx, lhs, rhs, "Matmul of lsh * rhs");

  // 2. generate potentially incorrect triple (a,b,c)
  SPU_ENFORCE(lhs.shape()[1] == rhs.shape()[0]);
  auto a = RandA_RSS(ctx, lhs.shape(), field_sigma_plus_k);
  auto b = RingChange(ctx, rhs, field, field_sigma_plus_k, true);
  auto c = MatRssMul_semi(ctx, a, b, "Matmul of a * b");

  // 3. sample r in \mathbb{Z}_{2^{sigma}}
  auto r__ = prg_state->genPubl(field_sigma_mask, {1});
  auto r = static_cast<uint128_t>(123);
  auto v = NdArrayRef(makeType<AShrTy>(field_sigma_plus_k), lhs.shape());
  auto w = NdArrayRef(makeType<AShrTy>(field_sigma_plus_k), {M, N});

  auto lhs_extend = RingChange(ctx, lhs, field, field_sigma_plus_k, true);
  auto res_extend = RingChange(ctx, res, field, field_sigma_plus_k, true);

  auto lhs_extend1 = getFirstShare(lhs_extend);
  auto lhs_extend2 = getSecondShare(lhs_extend);
  auto a1 = getFirstShare(a);
  auto a2 = getSecondShare(a);
  auto v1 = getFirstShare(v);
  auto v2 = getSecondShare(v);

  auto tmp1 = ring_mul(lhs_extend1, r);
  auto tmp2 = ring_mul(lhs_extend2, r);

  // calculate v = r * lhs - a
  ring_assign(v1, ring_sub(tmp1, a1));
  ring_assign(v2, ring_sub(tmp2, a2));

  auto reconstruct_v = RSS_A2P(ctx, v, "reconstruct v in MatMulPre");

  auto b1 = getFirstShare(b);
  auto b2 = getSecondShare(b);
  auto res_extend1 = getFirstShare(res_extend);
  auto res_extend2 = getSecondShare(res_extend);
  auto c1 = getFirstShare(c);
  auto c2 = getSecondShare(c);
  auto w1 = getFirstShare(w);
  auto w2 = getSecondShare(w);

  // calculate w = v * b - r * res + c
  tmp1 = ring_mmul(reconstruct_v, b1);
  tmp2 = ring_mmul(reconstruct_v, b2);
  ring_sub_(tmp1, ring_mul(res_extend1, r));
  ring_sub_(tmp2, ring_mul(res_extend2, r));
  ring_add_(tmp1, c1);
  ring_add_(tmp2, c2);
  ring_assign(w1, tmp1);
  ring_assign(w2, tmp2);

  auto reconstruct_w = RSS_A2P(ctx, w, "reconstruct w in MatMulPre");

  reconstruct_w =
      RingChange(ctx, reconstruct_w, field_sigma_plus_k, field, false);
  auto zeros = ring_zeros(field, {M, N});
  SPU_ENFORCE(ring_all_equal(zeros, reconstruct_w), "malicious in MulPre");

  return res;
}

NdArrayRef MatMulAA::proc(KernelEvalContext* ctx, const NdArrayRef& x,
                          const NdArrayRef& y) const {
  auto* prg_state = ctx->getState<PrgState>();
  const auto field = x.eltype().as<Ring2k>()->field();
  auto* comm = ctx->getState<Communicator>();
  auto rank = comm->getRank();
  auto ty = makeType<RingTy>(field);
  auto M = x.shape()[0];
  auto N = y.shape()[1];

  NdArrayRef out(makeType<AShrTy>(field), {M, N});
  NdArrayRef d(makeType<AShrTy>(field), x.shape());
  NdArrayRef e(makeType<AShrTy>(field), y.shape());

  NdArrayRef chi_1(ty, {M, N});
  NdArrayRef chi_2(ty, {M, N});
  NdArrayRef Phi(ty, {M, N});

  NdArrayRef beta_z1_start(ty, {M, N});
  NdArrayRef beta_z2_start(ty, {M, N});

  NdArrayRef beta_plus_gamma_z(ty, {M, N});

  // P0, Pj together sample random alpha_j
  auto [r0, r1] =
      prg_state->genPrssPair(field, {M, N}, PrgState::GenPrssCtrl::Both);

  auto d0 = getFirstShare(d);
  auto d1 = getSecondShare(d);

  auto e0 = getFirstShare(e);
  auto e1 = getSecondShare(e);

  auto x0 = getFirstShare(x);
  auto x1 = getSecondShare(x);
  auto x2 = getThirdShare(x);

  auto y0 = getFirstShare(y);
  auto y1 = getSecondShare(y);
  auto y2 = getThirdShare(y);

  auto z0 = getFirstShare(out);
  auto z1 = getSecondShare(out);
  auto z2 = getThirdShare(out);

  if (rank == 0) {
    ring_assign(d0, x1);
    ring_assign(d1, x0);
    ring_assign(e0, y1);
    ring_assign(e1, y0);

    ring_assign(z0, r1);
    ring_assign(z1, r0);
  }
  if (rank == 1) {
    ring_assign(d0, x0);
    ring_assign(d1, x2);
    ring_assign(e0, y0);
    ring_assign(e1, y2);

    ring_assign(z0, r0);
    ring_assign(z2, r1);
  }
  if (rank == 2) {
    ring_assign(d0, x2);
    ring_assign(d1, x0);
    ring_assign(e0, y2);
    ring_assign(e1, y0);

    ring_assign(z0, r1);
    ring_assign(z2, r0);
  }

  // p0, p1 : chi_1 = f1
  // p0, p2 : chi_2 = f0
  // p1, p2 : Phi = f2 - gamma_x * gamma_y
  auto f = MatMulPre(ctx, d, e);

  auto f0 = getFirstShare(f);
  auto f1 = getSecondShare(f);
  auto f2 = getThirdShare(f);

  if (rank == 0) {
    ring_assign(chi_1, f1);
    ring_assign(chi_2, f0);
  }
  if (rank == 1) {
    ring_assign(chi_1, f0);
    auto tmp1 = ring_sub(f1, ring_mmul(x2, y2));
    ring_assign(Phi, tmp1);
  }
  if (rank == 2) {
    ring_assign(chi_2, f1);
    auto tmp1 = ring_sub(f0, ring_mmul(x2, y2));
    ring_assign(Phi, tmp1);
  }

  // [beta*_z] = -(beta_x + gamma_x)[alpha_y] - (beta_y + gamma_y)[alpha_x]
  //             +[alpha_z] + [chi]
  if (rank == 0) {
    beta_z1_start = ring_sum(
        {ring_neg(ring_mmul(x2, y0)), ring_neg(ring_mmul(x0, y2)), z0, chi_1});
    beta_z2_start = ring_sum(
        {ring_neg(ring_mmul(x2, y1)), ring_neg(ring_mmul(x1, y2)), z1, chi_2});
  }
  if (rank == 1) {
    auto tmp2 = ring_neg(ring_add(x1, x2));
    auto tmp3 = ring_neg(ring_add(y1, y2));
    tmp2 = ring_mmul(tmp2, y0);
    tmp3 = ring_mmul(x0, tmp3);
    beta_z1_start = ring_sum({tmp2, tmp3, z0, chi_1});
  }
  if (rank == 2) {
    auto tmp2 = ring_neg(ring_add(x1, x2));
    auto tmp3 = ring_neg(ring_add(y1, y2));
    tmp2 = ring_mmul(tmp2, y0);
    tmp3 = ring_mmul(x0, tmp3);
    beta_z2_start = ring_sum({tmp2, tmp3, z0, chi_2});
  }
  beta_z1_start =
      JointMessagePassing(ctx, beta_z1_start, 0, 1, 2, "beta_z1_start");
  beta_z2_start =
      JointMessagePassing(ctx, beta_z2_start, 0, 2, 1, "beta_z2_start");
  auto beta_z_start = ring_add(beta_z1_start, beta_z2_start);

  if (rank == 1 || rank == 2) {
    // beta_z = beta*_z + beta_x * beta_y + Phi
    ring_assign(z1, ring_sum({beta_z_start, ring_mmul(x1, y1), Phi}));
    ring_assign(beta_plus_gamma_z, ring_add(z1, z2));
  }
  beta_plus_gamma_z =
      JointMessagePassing(ctx, beta_plus_gamma_z, 1, 2, 0, "beta_plus_gamma_z");
  if (rank == 0) {
    ring_assign(z2, beta_plus_gamma_z);
  }

  return out;
}

NdArrayRef LShiftA::proc(KernelEvalContext*, const NdArrayRef& in,
                         const Sizes& bits) const {
  const auto* in_ty = in.eltype().as<AShrTy>();
  const auto field = in_ty->field();
  bool is_splat = bits.size() == 1;
  return DISPATCH_ALL_FIELDS(field, [&]() {
    using shr_t = std::array<ring2k_t, 3>;

    NdArrayRef out(makeType<AShrTy>(field), in.shape());
    NdArrayView<shr_t> _out(out);
    NdArrayView<shr_t> _in(in);
    pforeach(0, in.numel(), [&](int64_t idx) {
      auto shift_bit = is_splat ? bits[0] : bits[idx];
      _out[idx][0] = _in[idx][0] << shift_bit;
      _out[idx][1] = _in[idx][1] << shift_bit;
      _out[idx][2] = _in[idx][2] << shift_bit;
    });

    return out;
  });
}

std::pair<NdArrayRef, NdArrayRef> TruncA::Trgen(KernelEvalContext* ctx,
                                                int64_t bits, FieldType field,
                                                int64_t numel) const {
  auto* prg_state = ctx->getState<PrgState>();
  auto* comm = ctx->getState<Communicator>();
  auto ty_ring = makeType<RingTy>(field);
  auto ashrty = makeType<AShrTy>(field);
  auto rank = comm->getRank();
  auto shape = {numel};
  const int64_t k = SizeOf(field) * 8;

  auto dotp = MatMulAA();
  auto matmul = MatMulAA();
  auto a2p = A2P();

  NdArrayRef r(ashrty, shape);
  NdArrayRef r1(ty_ring, shape);
  NdArrayRef r2(ty_ring, shape);
  NdArrayRef rd(ashrty, shape);

  NdArrayRef public_const1(ty_ring, {k - bits});
  NdArrayRef public_const2(ty_ring, {k});
  NdArrayRef X(ashrty, {numel, k - bits});
  NdArrayRef Y(ashrty, {numel, k - bits});
  NdArrayRef P(ashrty, {numel, k});
  NdArrayRef Q(ashrty, {numel, k});
  NdArrayRef tmp(ashrty, {numel, numel});
  NdArrayRef tmp2(ashrty, {numel, numel});
  NdArrayRef A(ashrty, shape);
  NdArrayRef B(ashrty, shape);

  // pack bits together
  NdArrayRef r1_bits(ty_ring, {numel * k});
  NdArrayRef r2_bits(ty_ring, {numel * k});
  NdArrayRef r1_bits_share(ashrty, {numel * k});
  NdArrayRef r2_bits_share(ashrty, {numel * k});

  // P_0 and P_j generate r_j by PRG
  // P0.prg_r0 = P2.prg_r1 = r2
  // P0.prg_r1 = P1.prg_r0 = r1
  auto [prg_r0, prg_r1] =
      prg_state->genPrssPair(field, shape, PrgState::GenPrssCtrl::Both);

  // actuall, for the trunc pair: r, rd
  // they should satisfy: rd = arshift(r, d)
  // but in swift, which generate [[·]] share of each bit
  // and use the following expression to calculate r and rd
  // r  = \Sigma_{i=0}^{k-1} (2^i * r[i])
  // rd = \Sigma_{i=d}^{k-1} (2^{i-d} * r[i])
  // so in swift : r = rshift(r, d)
  // which cause the truncation result to be wrong
  // so we need to guarantee the msb of r is 0,
  // so that arshift(r, d) = rshift(r, d)
  ring_rshift_(prg_r0, {static_cast<int64_t>(1)});
  ring_rshift_(prg_r1, {static_cast<int64_t>(1)});
  if (rank == 0) {
    r1 = prg_r1;
    r2 = prg_r0;
  }
  if (rank == 1) {
    r1 = prg_r0;
  }
  if (rank == 2) {
    r2 = prg_r1;
  }

  DISPATCH_ALL_FIELDS(field, [&]() {
    using el_t = ring2k_t;
    using shr_t = std::array<el_t, 3>;

    NdArrayView<el_t> _r1(r1);
    NdArrayView<el_t> _r2(r2);
    NdArrayView<el_t> _r1_bits(r1_bits);
    NdArrayView<el_t> _r2_bits(r2_bits);

    // bit decompose r1 and r2
    pforeach(0, numel, [&](int64_t idx) {
      for (int64_t i = 0; i < k; i++) {
        _r1_bits[idx * k + i] = static_cast<ring2k_t>((_r1[idx] >> i) & 0x1);
        _r2_bits[idx * k + i] = static_cast<ring2k_t>((_r2[idx] >> i) & 0x1);
      }
    });

    // joint share r1_bits and r2_bits
    r1_bits_share = JointSharing(ctx, r1_bits, 0, 1, "r1_bits share");
    r2_bits_share = JointSharing(ctx, r2_bits, 0, 2, "r2_bits share");

    // for each r in batch:
    // A = x \cdot y
    // B = p \cdot q
    NdArrayView<shr_t> _X(X);
    NdArrayView<shr_t> _Y(Y);
    NdArrayView<shr_t> _P(P);
    NdArrayView<shr_t> _Q(Q);
    NdArrayView<shr_t> _r1_bits_share(r1_bits_share);
    NdArrayView<shr_t> _r2_bits_share(r2_bits_share);

    // public_const1 = 2 ^ {i - bits + 1} for i \in {d, ..., k - 1}
    // public_const2 = 2 ^ {i + 1} for i \in {0, 1, ..., k - 1}
    NdArrayView<el_t> _public_const1(public_const1);
    NdArrayView<el_t> _public_const2(public_const2);
    for (int64_t i = bits; i < k; i++) {
      _public_const1[i - bits] = (static_cast<ring2k_t>(1) << (i - bits + 1));
    }
    for (int64_t i = 0; i < k; i++) {
      _public_const2[i] = (static_cast<ring2k_t>(1) << (i + 1));
    }

    NdArrayView<shr_t> _A(A);
    NdArrayView<shr_t> _B(B);
    pforeach(0, numel, [&](int64_t idx) {
      for (int64_t i = bits; i < k; i++) {
        // MulAP
        _X[idx * (k - bits) + (i - bits)][0] =
            (ring2k_t(1) << (i - bits + 1)) * _r1_bits_share[idx * k + i][0];
        _X[idx * (k - bits) + (i - bits)][1] =
            (ring2k_t(1) << (i - bits + 1)) * _r1_bits_share[idx * k + i][1];
        _X[idx * (k - bits) + (i - bits)][2] =
            (ring2k_t(1) << (i - bits + 1)) * _r1_bits_share[idx * k + i][2];

        _Y[idx * (k - bits) + (i - bits)][0] = _r2_bits_share[idx * k + i][0];
        _Y[idx * (k - bits) + (i - bits)][1] = _r2_bits_share[idx * k + i][1];
        _Y[idx * (k - bits) + (i - bits)][2] = _r2_bits_share[idx * k + i][2];
      }
      for (int64_t i = 0; i < k; i++) {
        // MulAP
        _P[idx * k + i][0] =
            (ring2k_t(1) << (i + 1)) * _r1_bits_share[idx * k + i][0];
        _P[idx * k + i][1] =
            (ring2k_t(1) << (i + 1)) * _r1_bits_share[idx * k + i][1];
        _P[idx * k + i][2] =
            (ring2k_t(1) << (i + 1)) * _r1_bits_share[idx * k + i][2];

        _Q[idx * k + i][0] = _r2_bits_share[idx * k + i][0];
        _Q[idx * k + i][1] = _r2_bits_share[idx * k + i][1];
        _Q[idx * k + i][2] = _r2_bits_share[idx * k + i][2];
      }
    });
    Y = Y.transpose();
    Q = Q.transpose();

    tmp = matmul.proc(ctx, X, Y);
    tmp2 = matmul.proc(ctx, P, Q);
    NdArrayView<shr_t> _tmp(tmp);
    NdArrayView<shr_t> _tmp2(tmp2);
    SPU_ENFORCE(tmp.numel() == numel * numel);
    SPU_ENFORCE(tmp2.numel() == numel * numel);

    pforeach(0, numel, [&](int64_t idx) {
      _A[idx][0] = _tmp[idx * numel + idx][0];
      _A[idx][1] = _tmp[idx * numel + idx][1];
      _A[idx][2] = _tmp[idx * numel + idx][2];

      _B[idx][0] = _tmp2[idx * numel + idx][0];
      _B[idx][1] = _tmp2[idx * numel + idx][1];
      _B[idx][2] = _tmp2[idx * numel + idx][2];
    });

    NdArrayView<shr_t> _r(r);
    NdArrayView<shr_t> _rd(rd);

    pforeach(0, numel, [&](int64_t idx) {
      // use tmp as sum
      _rd[idx][0] = (ring2k_t)0;
      _rd[idx][1] = (ring2k_t)0;
      _rd[idx][2] = (ring2k_t)0;
      for (int64_t i = bits; i < k; i++) {
        _rd[idx][0] +=
            (((ring2k_t)1 << (i - bits)) *
             (_r1_bits_share[idx * k + i][0] + _r2_bits_share[idx * k + i][0]));
        _rd[idx][1] +=
            (((ring2k_t)1 << (i - bits)) *
             (_r1_bits_share[idx * k + i][1] + _r2_bits_share[idx * k + i][1]));
        _rd[idx][2] +=
            (((ring2k_t)1 << (i - bits)) *
             (_r1_bits_share[idx * k + i][2] + _r2_bits_share[idx * k + i][2]));
      }
      _rd[idx][0] -= _A[idx][0];
      _rd[idx][1] -= _A[idx][1];
      _rd[idx][2] -= _A[idx][2];

      _r[idx][0] = (ring2k_t)0;
      _r[idx][1] = (ring2k_t)0;
      _r[idx][2] = (ring2k_t)0;
      for (int64_t i = 0; i < k; i++) {
        _r[idx][0] += (((ring2k_t)1 << (i)) * (_r1_bits_share[idx * k + i][0] +
                                               _r2_bits_share[idx * k + i][0]));
        _r[idx][1] += (((ring2k_t)1 << (i)) * (_r1_bits_share[idx * k + i][1] +
                                               _r2_bits_share[idx * k + i][1]));
        _r[idx][2] += (((ring2k_t)1 << (i)) * (_r1_bits_share[idx * k + i][2] +
                                               _r2_bits_share[idx * k + i][2]));
      }
      _r[idx][0] -= _B[idx][0];
      _r[idx][1] -= _B[idx][1];
      _r[idx][2] -= _B[idx][2];
    });
  });
  return std::make_pair(r, rd);
}

NdArrayRef TruncA::proc(KernelEvalContext* ctx, const NdArrayRef& x,
                        size_t bits, SignType sign) const {
  const auto field = x.eltype().as<Ring2k>()->field();
  auto* comm = ctx->getState<Communicator>();
  auto rank = comm->getRank();

  NdArrayRef out(makeType<AShrTy>(field), x.shape());
  auto numel = x.numel();
  auto a2p = A2P();

  auto [r, rd] = TruncA::Trgen(ctx, static_cast<int64_t>(bits), field, numel);

  r.reshape(x.shape());
  rd.reshape(x.shape());

  NdArrayRef x_minux_r_share(makeType<AShrTy>(field), x.shape());
  DISPATCH_ALL_FIELDS(field, [&]() {
    using el_t = ring2k_t;
    using shr_t = std::array<el_t, 3>;

    NdArrayView<shr_t> _x_minux_r_share(x_minux_r_share);
    NdArrayView<shr_t> _r(r);
    NdArrayView<shr_t> _x(x);
    pforeach(0, numel, [&](int64_t idx) {
      _x_minux_r_share[idx][0] = _x[idx][0] - _r[idx][0];
      _x_minux_r_share[idx][1] = _x[idx][1] - _r[idx][1];
      _x_minux_r_share[idx][2] = _x[idx][2] - _r[idx][2];
    });

    auto x_minus_r = a2p.proc(ctx, x_minux_r_share);

    auto x_minus_r_d = ring_arshift(x_minus_r, {static_cast<int64_t>(bits)});

    NdArrayView<shr_t> _out(out);
    NdArrayView<shr_t> _rd(rd);
    NdArrayView<el_t> _x_minus_r_d(x_minus_r_d);
    pforeach(0, numel, [&](int64_t idx) {
      _out[idx][0] = _rd[idx][0];
      _out[idx][1] = _rd[idx][1];
      _out[idx][2] = _rd[idx][2];
      if (rank == 0) _out[idx][2] += _x_minus_r_d[idx];
      if (rank == 1 || rank == 2) _out[idx][1] += _x_minus_r_d[idx];
    });
  });

  return out.as(x.eltype());
}

}  // namespace spu::mpc::swift
