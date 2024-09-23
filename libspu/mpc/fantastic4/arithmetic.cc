#include "libspu/mpc/fantastic4/arithmetic.h"
#include <future>
#include "libspu/mpc/fantastic4/type.h"
#include "libspu/mpc/fantastic4/value.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/ab_api.h"

namespace spu::mpc::fantastic4 {

namespace {
  
  static NdArrayRef wrap_mul_aa(SPUContext* ctx, const NdArrayRef& x,
                              const NdArrayRef& y) {
    SPU_ENFORCE(x.shape() == y.shape());
    return UnwrapValue(mul_aa(ctx, WrapValue(x), WrapValue(y)));
  }

  size_t PrevRankA(size_t rank, size_t world_size){
    return (rank + world_size -1) % world_size;
  }

  size_t OffsetRankA(size_t myrank, size_t other, size_t world_size){
    size_t offset = (myrank + world_size -other) % world_size;
    if(offset == 3){
      offset = 1;
    }
    return offset;
  }


  // Sender and Receiver jointly input a X
  template <typename el_t>
  void JointInputArith(KernelEvalContext* ctx, std::vector<el_t>& input, NdArrayRef& output, size_t sender, size_t backup, size_t receiver, size_t outsider){
    auto* comm = ctx->getState<Communicator>();
    size_t world_size =  comm->getWorldSize();
    auto* prg_state = ctx->getState<PrgState>();
    auto myrank = comm->getRank();

    using shr_t = std::array<el_t, 3>;
    NdArrayView<shr_t> _out(output);
    
    // Receiver's Previous Party Rank
    // The mask corresponds to the prev party of receiver, receiver doesn't have the correpsonding PRG of its prev party
    size_t receiver_prev_rank = PrevRankA(receiver, world_size);

    // My offset from the receiver_prev_rank. 
    // 0- i'm the receiver_prev_rank
    // 1- i'm prev/next party of receiver_prev_rank
    // 2- next next
    size_t offset_from_receiver_prev = OffsetRankA(myrank, receiver_prev_rank, world_size);
    size_t offset_from_outsider_prev = OffsetRankA(myrank, (outsider + 4 - 1)%4 , world_size);

    if(myrank != receiver){
      // Non-Interactive Random Masks Generation.
      std::vector<el_t> r(output.numel());

      if(offset_from_receiver_prev == 0){
          // should use PRG[0]
          prg_state->fillPrssTuple<el_t>(r.data(), nullptr, nullptr , r.size(),
                              PrgState::GenPrssCtrl::First);
      }
      if(offset_from_receiver_prev == 1){
          // should use PRG[1]
          prg_state->fillPrssTuple<el_t>(nullptr, r.data(), nullptr , r.size(),
                              PrgState::GenPrssCtrl::Second);
      }
      if(offset_from_receiver_prev == 2){
          // should use PRG[2]
          prg_state->fillPrssTuple<el_t>(nullptr, nullptr, r.data(), r.size(),
                              PrgState::GenPrssCtrl::Third);
      }

      // For sender,backup,outsider
      // the corresponding share is set to r

      pforeach(0, output.numel(), [&](int64_t idx) {
          _out[idx][offset_from_receiver_prev] += r[idx];
      }); 

      if(myrank != outsider){

        std::vector<el_t> input_minus_r(output.numel());

        // For sender, backup
        // compute and set masked input x-r
        pforeach(0, output.numel(), [&](int64_t idx) {
          input_minus_r[idx] = (input[idx] - r[idx]);
          _out[idx][offset_from_outsider_prev] +=  input_minus_r[idx];
          
          }); 

        // Sender send x-r to receiver
        if(myrank == sender) {
          comm->sendAsync<el_t>(receiver, input_minus_r, "Joint Input");
        }

        // Backup update x-r for sender-to-receiver channel
        if(myrank == backup) {
          // Todo:
          // MAC update input_minus_r
        }
      }
    }

    if (myrank == receiver) {
      auto input_minus_r = comm->recv<el_t>(sender, "Joint Input");
      pforeach(0, output.numel(), [&](int64_t idx) {
          _out[idx][offset_from_outsider_prev] += input_minus_r[idx];
      }); 

      // Todo: 
      // Mac update sender-backup channel
    }
  }

}

NdArrayRef RandA::proc(KernelEvalContext* ctx, const Shape& shape) const {
  auto* prg_state = ctx->getState<PrgState>();
  const auto field = ctx->getState<Z2kState>()->getDefaultField();

  NdArrayRef out(makeType<AShrTy>(field), shape);

  DISPATCH_ALL_FIELDS(field, [&]() {
    using el_t = ring2k_t;

    std::vector<el_t> r0(shape.numel());
    std::vector<el_t> r1(shape.numel());
    std::vector<el_t> r2(shape.numel());

    prg_state->fillPrssTuple<el_t>(r0.data(), nullptr, nullptr, r0.size(),
                              PrgState::GenPrssCtrl::First);
    prg_state->fillPrssTuple<el_t>(nullptr, r1.data(), nullptr, r1.size(),
                              PrgState::GenPrssCtrl::Second); 
    prg_state->fillPrssTuple<el_t>(nullptr, nullptr, r2.data(), r2.size(),
                              PrgState::GenPrssCtrl::Third);   

    NdArrayView<std::array<el_t, 3>> _out(out);

    pforeach(0, out.numel(), [&](int64_t idx) {
      // Comparison only works for [-2^(k-2), 2^(k-2)).
      // TODO: Move this constraint to upper layer, saturate it here.
      _out[idx][0] = r0[idx] >> 2;
      _out[idx][1] = r1[idx] >> 2;
      _out[idx][2] = r2[idx] >> 2;
    });
  });

  return out;
}

NdArrayRef A2P::proc(KernelEvalContext* ctx, const NdArrayRef& in) const {
  auto* comm = ctx->getState<Communicator>();
  const auto field = in.eltype().as<AShrTy>()->field();
  auto numel = in.numel();

  return DISPATCH_ALL_FIELDS(field, [&]() {
    using pshr_el_t = ring2k_t;
    using ashr_el_t = ring2k_t;
    using ashr_t = std::array<ashr_el_t, 3>;

    NdArrayRef out(makeType<Pub2kTy>(field), in.shape());
    NdArrayView<pshr_el_t> _out(out);
    NdArrayView<ashr_t> _in(in);

    std::vector<ashr_el_t> x3(numel);

    pforeach(0, numel, [&](int64_t idx) { x3[idx] = _in[idx][2]; });
    
    // Pass the third share to previous party
    auto x4 = comm->rotate<ashr_el_t>(x3, "a2p");  // comm => 1, k

    pforeach(0, numel, [&](int64_t idx) {
      _out[idx] = _in[idx][0] + _in[idx][1] + _in[idx][2] + x4[idx];
    });

    return out;
  });
}

// x1 = x
// x2 = x3 = x4 = 0

NdArrayRef P2A::proc(KernelEvalContext* ctx, const NdArrayRef& in) const {
  auto* comm = ctx->getState<Communicator>();

  const auto* in_ty = in.eltype().as<Pub2kTy>();
  const auto field = in_ty->field();

  auto rank = comm->getRank();

  return DISPATCH_ALL_FIELDS(field, [&]() {
    using pshr_el_t = ring2k_t;
    using ashr_el_t = ring2k_t;
    using ashr_t = std::array<ashr_el_t, 3>;

    NdArrayRef out(makeType<AShrTy>(field), in.shape());
    NdArrayView<ashr_t> _out(out);
    NdArrayView<pshr_el_t> _in(in);

    pforeach(0, in.numel(), [&](int64_t idx) {
      _out[idx][0] = rank == 0 ? _in[idx] : 0;
      _out[idx][1] = rank == 3 ? _in[idx] : 0;
      _out[idx][2] = rank == 2 ? _in[idx] : 0;
    });

// for debug purpose, randomize the inputs to avoid corner cases.
#ifdef ENABLE_MASK_DURING_FANTASTIC4_P2A
    std::vector<ashr_el_t> r0(in.numel());
    std::vector<ashr_el_t> r1(in.numel());
    std::vector<ashr_el_t> r2(in.numel());

    std::vector<ashr_el_t> s0(in.numel());
    std::vector<ashr_el_t> s1(in.numel());
    std::vector<ashr_el_t> s2(in.numel());

    auto* prg_state = ctx->getState<PrgState>();
    prg_state->fillPrssTuple<ashr_el_t>(r0.data(), nullptr, nullptr, r0.size(),
                              PrgState::GenPrssCtrl::First);
    prg_state->fillPrssTuple<ashr_el_t>(nullptr, r1.data(), nullptr, r1.size(),
                              PrgState::GenPrssCtrl::Second); 
    prg_state->fillPrssTuple<ashr_el_t>(nullptr, nullptr, r2.data(), r2.size(),
                              PrgState::GenPrssCtrl::Third);                                                   

    for (int64_t idx = 0; idx < in.numel(); idx++) {
      s0[idx] = r0[idx] - r1[idx];
      s1[idx] = r1[idx] - r2[idx];
      }

    s2 = comm->rotate<ashr_el_t>(s1, "p2a.zero");

    for (int64_t idx = 0; idx < in.numel(); idx++) {
      // printf(" My rank = %zu, share = (%llu, %llu, %llu)", comm->getRank(), (unsigned long long)s0[idx], (unsigned long long)s1[idx], (unsigned long long)s2[idx]);
      // if(comm->getRank() == 0 && idx == 0){
      //   printf(" My rank = %zu, share = %llu\n", comm->getRank(), (unsigned long long)(~(s1[idx] + s2[idx])));
      // }
      // if(comm->getRank() == 2 && idx == 0){
      //   printf(" My rank = %zu, share = %llu\n", comm->getRank(), (unsigned long long)(-s1[idx] - s2[idx]));
      // }
      _out[idx][0] += s0[idx];
      _out[idx][1] += s1[idx];
      _out[idx][2] += s2[idx];
    }
#endif

    return out;
  });
}

NdArrayRef A2V::proc(KernelEvalContext* ctx, const NdArrayRef& in,
                     size_t rank) const {
  auto* comm = ctx->getState<Communicator>();
  const auto field = in.eltype().as<AShrTy>()->field();

  return DISPATCH_ALL_FIELDS(field, [&]() {
    using vshr_el_t = ring2k_t;
    using ashr_el_t = ring2k_t;
    using ashr_t = std::array<ashr_el_t, 3>;

    NdArrayView<ashr_t> _in(in);
    auto out_ty = makeType<Priv2kTy>(field, rank);

    if (comm->getRank() == rank) {
      auto x4 = comm->recv<ashr_el_t>(comm->nextRank(), "a2v");  // comm => 1, k
                                                                 //
      NdArrayRef out(out_ty, in.shape());
      NdArrayView<vshr_el_t> _out(out);

      pforeach(0, in.numel(), [&](int64_t idx) {
        _out[idx] = _in[idx][0] + _in[idx][1] + _in[idx][2] + x4[idx];
      });
      return out;

    } else if (comm->getRank() == (rank + 1) % 4) {
      std::vector<ashr_el_t> x3(in.numel());

      pforeach(0, in.numel(), [&](int64_t idx) { x3[idx] = _in[idx][2]; });

      comm->sendAsync<ashr_el_t>(comm->prevRank(), x3,
                                 "a2v");  // comm => 1, k
      return makeConstantArrayRef(out_ty, in.shape());
    } else {
      return makeConstantArrayRef(out_ty, in.shape());
    }
  });
}



// /////////////////////////////////////////////////
// V2A
// In aby3, no use of prg, the dealer just distribute shr1 and shr2, set shr3 = 0
// /////////////////////////////////////////////////
NdArrayRef V2A::proc(KernelEvalContext* ctx, const NdArrayRef& in) const {
  auto* comm = ctx->getState<Communicator>();

  const auto* in_ty = in.eltype().as<Priv2kTy>();
  const auto field = in_ty->field();

  size_t owner_rank = in_ty->owner();

  return DISPATCH_ALL_FIELDS(field, [&]() {
    using ashr_el_t = ring2k_t;
    using ashr_t = std::array<ashr_el_t, 3>;

    NdArrayRef out(makeType<AShrTy>(field), in.shape());
    NdArrayView<ashr_t> _out(out);

    if (comm->getRank() == owner_rank) {
      auto splits = ring_rand_additive_splits(in, 3);
      // send (shr2, shr3) to next party
      //      (shr3, shr1) to next next party
      //      (shr1, shr2) to prev party
      // shr4 = 0

      comm->sendAsync((owner_rank + 1) % 4, splits[1], "v2a 1");  // comm => 1, k
      comm->sendAsync((owner_rank + 1) % 4, splits[2], "v2a 2");  // comm => 1, k

      comm->sendAsync((owner_rank + 2) % 4, splits[2], "v2a 1");  // comm => 1, k
      comm->sendAsync((owner_rank + 2) % 4, splits[0], "v2a 2");  // comm => 1, k

      comm->sendAsync((owner_rank + 3) % 4, splits[0], "v2a 1");  // comm => 1, k
      comm->sendAsync((owner_rank + 3) % 4, splits[1], "v2a 2");  // comm => 1, k


      NdArrayView<ashr_el_t> _s0(splits[0]);
      NdArrayView<ashr_el_t> _s1(splits[1]);
      NdArrayView<ashr_el_t> _s2(splits[2]);

      pforeach(0, in.numel(), [&](int64_t idx) {
        _out[idx][0] = _s0[idx];
        _out[idx][1] = _s1[idx];
        _out[idx][1] = _s2[idx];
      });
    } 
    else if (comm->getRank() == (owner_rank + 1) % 4) {
      auto x1 = comm->recv<ashr_el_t>((comm->getRank() + 3) % 4, "v2a 1");  // comm => 1, k
      auto x2 = comm->recv<ashr_el_t>((comm->getRank() + 3) % 4, "v2a 2");  // comm => 1, k
      pforeach(0, in.numel(), [&](int64_t idx) {
        
        _out[idx][0] = x1[idx];
        _out[idx][1] = x2[idx];
        _out[idx][2] = 0;
      });
    } 
    else if (comm->getRank() == (owner_rank + 2) % 4) {
      auto x3 = comm->recv<ashr_el_t>((comm->getRank() + 2) % 4, "v2a 1");  // comm => 1, k
      auto x1 = comm->recv<ashr_el_t>((comm->getRank() + 2) % 4, "v2a 2");  // comm => 1, k
      pforeach(0, in.numel(), [&](int64_t idx) {
        _out[idx][0] = x3[idx];
        _out[idx][1] = 0;
        _out[idx][2] = x1[idx];
      });
    } else {
      auto x1 = comm->recv<ashr_el_t>((comm->getRank() + 1) % 4, "v2a 1");  // comm => 1, k
      auto x2 = comm->recv<ashr_el_t>((comm->getRank() + 1) % 4, "v2a 2");  // comm => 1, k
      pforeach(0, in.numel(), [&](int64_t idx) {
        _out[idx][0] = 0;
        _out[idx][1] = x1[idx];
        _out[idx][2] = x2[idx];
      });
    }

    return out;
  });
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
    using shr_t = std::array<el_t, 3>;

    NdArrayRef out(makeType<AShrTy>(field), lhs.shape());
    NdArrayView<shr_t> _out(out);
    NdArrayView<shr_t> _lhs(lhs);
    NdArrayView<el_t> _rhs(rhs);

    pforeach(0, lhs.numel(), [&](int64_t idx) {
      _out[idx][0] = _lhs[idx][0];
      _out[idx][1] = _lhs[idx][1];
      _out[idx][2] = _lhs[idx][2];
      if (rank == 0) {_out[idx][0] += _rhs[idx];}
      if (rank == 2) {_out[idx][2] += _rhs[idx];}
      if (rank == 3) {_out[idx][1] += _rhs[idx];}
    });
    return out;
  });
}

NdArrayRef AddAA::proc(KernelEvalContext*, const NdArrayRef& lhs,
                       const NdArrayRef& rhs) const {
  const auto* lhs_ty = lhs.eltype().as<AShrTy>();
  const auto* rhs_ty = rhs.eltype().as<AShrTy>();

  SPU_ENFORCE(lhs_ty->field() == rhs_ty->field());
  const auto field = lhs_ty->field();

  return DISPATCH_ALL_FIELDS(field, [&]() {
    using shr_t = std::array<ring2k_t, 3>;

    NdArrayRef out(makeType<AShrTy>(field), lhs.shape());
    NdArrayView<shr_t> _out(out);
    NdArrayView<shr_t> _lhs(lhs);
    NdArrayView<shr_t> _rhs(rhs);

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
    using shr_t = std::array<el_t, 3>;

    NdArrayRef out(makeType<AShrTy>(field), lhs.shape());
    NdArrayView<shr_t> _out(out);
    NdArrayView<shr_t> _lhs(lhs);
    NdArrayView<el_t> _rhs(rhs);

    pforeach(0, lhs.numel(), [&](int64_t idx) {
      _out[idx][0] = _lhs[idx][0] * _rhs[idx];
      _out[idx][1] = _lhs[idx][1] * _rhs[idx];
      _out[idx][2] = _lhs[idx][2] * _rhs[idx];
    });
    return out;
  });
}

NdArrayRef MulAA::proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                       const NdArrayRef& rhs) const {
  const auto field = lhs.eltype().as<Ring2k>()->field();
   auto* comm = ctx->getState<Communicator>();
  auto rank = comm->getRank();
  auto next_rank = (rank + 1) % 4;

  return DISPATCH_ALL_FIELDS(field, [&]() {
    using el_t = ring2k_t;
    using shr_t = std::array<el_t, 3>;

    NdArrayView<shr_t> _lhs(lhs);
    NdArrayView<shr_t> _rhs(rhs);
    NdArrayRef out(makeType<AShrTy>(field), lhs.shape());
    NdArrayView<shr_t> _out(out);
    pforeach(0, lhs.numel(), [&](int64_t idx) {
        for(auto i = 0; i < 3 ; i++ ){
          _out[idx][i] = 0;
        }
    });
    
    std::array<std::vector<el_t>, 5> a;

    for (auto& vec : a) {
        vec = std::vector<el_t>(lhs.numel());
    }
    pforeach(0, lhs.numel(), [&](int64_t idx) {
        for(auto i =0; i<5;i++){
          a[i][idx] = 0;
        }
    });

    pforeach(0, lhs.numel(), [&](int64_t idx) {
        a[rank][idx] = (_lhs[idx][0] + _lhs[idx][1]) * _rhs[idx][0] + _lhs[idx][0] * _rhs[idx][1]; // xi*yi + xi*yj + xj*yi
        a[next_rank][idx] = (_lhs[idx][1] + _lhs[idx][2]) * _rhs[idx][1] + _lhs[idx][1] * _rhs[idx][2];  // xj*yj + xj*yg + xg*yj
        a[4][idx] = _lhs[idx][0] * _rhs[idx][2] + _lhs[idx][2] * _rhs[idx][0];                    // xi*yg + xg*yi
    });

    JointInputArith<el_t>(ctx, a[1], out, 0, 1, 3, 2);
    JointInputArith<el_t>(ctx, a[2], out, 1, 2, 0, 3);
    JointInputArith<el_t>(ctx, a[3], out, 2, 3, 1, 0);
    JointInputArith<el_t>(ctx, a[0], out, 3, 0, 2, 1);
    JointInputArith<el_t>(ctx, a[4], out, 0, 2, 3, 1);
    JointInputArith<el_t>(ctx, a[4], out, 1, 3, 2, 0);

    return out;
  });
}

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

NdArrayRef MatMulAA::proc(KernelEvalContext* ctx, const NdArrayRef& x,
                          const NdArrayRef& y) const {

  const auto field = x.eltype().as<Ring2k>()->field();
   auto* comm = ctx->getState<Communicator>();
  auto rank = comm->getRank();
  auto next_rank = (rank + 1) % 4;

  
  auto M = x.shape()[0];
  auto K = x.shape()[1];
  auto N = y.shape()[1];
  return DISPATCH_ALL_FIELDS(field, [&]() {
    using el_t = ring2k_t;
    using shr_t = std::array<el_t, 3>;

    NdArrayRef out(makeType<AShrTy>(field), {M, N});

    NdArrayView<shr_t> _x(x);
    NdArrayView<shr_t> _y(y);
    NdArrayView<shr_t> _out(out);

    pforeach(0, M, [&](int64_t row) {
      for(int64_t col = 0; col < N ; col++ ){
        _out[row * N + col][0] = 0;
        _out[row * N + col][1] = 0;
        _out[row * N + col][2] = 0;
      }
    });

    std::array<std::vector<el_t>, 5> a;
    for (auto& vec : a) {
        vec = std::vector<el_t>(out.numel());
    }
    pforeach(0, out.numel(), [&](int64_t idx) {
        for(auto i =0; i<5;i++){
          a[i][idx] = 0;
        }
    });
    pforeach(0, M, [&](int64_t i) {
        for(int64_t j = 0; j < N; j++) {
          for(int64_t k = 0; k < K; k++) {
              // xi*yi + xi*yj + xj*yi
              a[rank][i * N + j] += (_x[i * K + k][0] + _x[i * K + k][1]) * _y[k * N + j][0] + _x[i * K + k][0] * _y[k * N + j][1]; 
              // xj*yj + xj*yg + xg*yj
              a[next_rank][i * N + j] += (_x[i * K + k][1] + _x[i * K + k][2]) * _y[k * N + j][1] + _x[i * K + k][1] * _y[k * N + j][2];
              // xi*yg + xg*yi 
              a[4][i * N + j] += _x[i * K + k][0] * _y[k * N + j][2] + _x[i * K + k][2] * _y[k * N + j][0]; 
          }
        }                
    });
    JointInputArith<el_t>(ctx, a[1], out, 0, 1, 3, 2);
    JointInputArith<el_t>(ctx, a[2], out, 1, 2, 0, 3);
    JointInputArith<el_t>(ctx, a[3], out, 2, 3, 1, 0);
    JointInputArith<el_t>(ctx, a[0], out, 3, 0, 2, 1);
    JointInputArith<el_t>(ctx, a[4], out, 0, 2, 3, 1);
    JointInputArith<el_t>(ctx, a[4], out, 1, 3, 2, 0);

    return out;


  });
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

void printBinary(unsigned long long x, size_t k) {
    for (int i = k - 1; i >= 0; --i) {
        unsigned long long bit = (x >> i) & 1ULL;
        printf("%llu", bit);
    }
}

NdArrayRef TruncAPr::proc(KernelEvalContext* ctx, const NdArrayRef& in, size_t bits,
                  SignType sign) const {
  (void)sign;  // TODO: optimize me.

  const auto field = in.eltype().as<Ring2k>()->field();
  const size_t k = SizeOf(field) * 8;
  auto* prg_state = ctx->getState<PrgState>();
  auto* comm = ctx->getState<Communicator>();
  auto rank = comm->getRank();



  return DISPATCH_ALL_FIELDS(field, [&]() {
    using el_t = ring2k_t;
    using shr_t = std::array<el_t, 3>;

    NdArrayRef out(makeType<AShrTy>(field), in.shape());
    NdArrayView<shr_t> _out(out);
    NdArrayView<shr_t> _in(in);

    NdArrayRef rb_shr(makeType<AShrTy>(field), in.shape());
    NdArrayView<shr_t> _rb_shr(rb_shr);

    NdArrayRef rc_shr(makeType<AShrTy>(field), in.shape());
    NdArrayView<shr_t> _rc_shr(rc_shr);

    NdArrayRef masked_input(makeType<AShrTy>(field), in.shape());
    NdArrayView<shr_t> _masked_input(masked_input);

    NdArrayRef sb_shr(makeType<AShrTy>(field), in.shape());
    NdArrayView<shr_t> _sb_shr(sb_shr);
    NdArrayRef sc_shr(makeType<AShrTy>(field), in.shape());
    NdArrayView<shr_t> _sc_shr(sc_shr);

    NdArrayRef overflow(makeType<AShrTy>(field), in.shape());
    NdArrayView<shr_t> _overflow(overflow);

    pforeach(0, out.numel(), [&](int64_t idx) {
          _out[idx][0] = 0;
          _out[idx][1] = 0;
          _out[idx][2] = 0;
          _rb_shr[idx][0] = 0;
          _rb_shr[idx][1] = 0;
          _rb_shr[idx][2] = 0;
          _rc_shr[idx][0] = 0;
          _rc_shr[idx][1] = 0;
          _rc_shr[idx][2] = 0;

          _sb_shr[idx][0] = 0;
          _sb_shr[idx][1] = 0;
          _sb_shr[idx][2] = 0;
          _sc_shr[idx][0] = 0;
          _sc_shr[idx][1] = 0;
          _sc_shr[idx][2] = 0;

    });
    

    if(rank == (size_t)0){ 
        // -------------------------------------
        // Step 1: Generate r and rb, rc
        // -------------------------------------
        // locally compute PRG[1] (unknown to P2), PRG[2] (unknown to P3)
        
        // std::vector<el_t> r0(output.numel());
        std::vector<el_t> r1(out.numel());
        std::vector<el_t> r2(out.numel());

        prg_state->fillPrssTuple<el_t>(nullptr, r1.data(), nullptr , r1.size(),
                                PrgState::GenPrssCtrl::Second);
        prg_state->fillPrssTuple<el_t>(nullptr, nullptr, r2.data() ,r2.size(),
                                PrgState::GenPrssCtrl::Third);

        std::vector<el_t> r(out.numel());
        std::vector<el_t> rb(out.numel());
        std::vector<el_t> rc(out.numel());
        
        pforeach(0, out.numel(), [&](int64_t idx) {
          // r = r_{k-1}......r_{0}
          r[idx] = r1[idx] + r2[idx];
          // rb = r >> k-1
          rb[idx] = r[idx] >> (k-1);
          // rc = r_{k-2}.....r_{m}
          rc[idx] = (r[idx] << 1) >> (bits + 1);
        });

        // -------------------------------------
        // Step 2: Generate the share of rb, rc
        // -------------------------------------
        JointInputArith(ctx, rb, rb_shr, 0, 1, 3, 2);
        JointInputArith(ctx, rc, rc_shr, 0, 1, 3, 2);

        // -------------------------------------
        // Step 3: compute [x] + [r]
        //          [r] = r0 + r1 + r2 + r3, only r1 and r2 are non-zero
        // -------------------------------------
        
        pforeach(0, out.numel(), [&](int64_t idx) {
          _masked_input[idx][0] = _in[idx][0]; // r0 = 0
          _masked_input[idx][1] = _in[idx][1] + r1[idx];
          _masked_input[idx][2] = _in[idx][2] + r2[idx];
        });

        // -------------------------------------
        // Step 4: Let P2 and P3 reconstruct s = x + r
        //         by P1 sends s1 to P2
        //            P2 sends s2 to P3
        // -------------------------------------


        // -------------------------------------
        // Step 5: compute sb = s{k-1} and sc = s{k-2}.....s{m}
        // -------------------------------------
        std::vector<el_t> sb(out.numel());
        std::vector<el_t> sc(out.numel());
        JointInputArith(ctx, sb, sb_shr, 2, 3, 0, 1);
        JointInputArith(ctx, sc, sc_shr, 2, 3, 0, 1);

        // -------------------------------------
        // Step 6: compute sb = s{k-1} and sc = s{k-2}.....s{m}
        // -------------------------------------
        auto sb_mul_rb = wrap_mul_aa(ctx->sctx(), sb_shr, rb_shr);
        NdArrayView<shr_t> _sb_mul_rb(sb_mul_rb);
        pforeach(0, out.numel(), [&](int64_t idx) {
          _overflow[idx][0] = _rb_shr[idx][0] + _sb_shr[idx][0] - 2*_sb_mul_rb[idx][0];
          _overflow[idx][1] = _rb_shr[idx][1] + _sb_shr[idx][1] - 2*_sb_mul_rb[idx][1];
          _overflow[idx][2] = _rb_shr[idx][2] + _sb_shr[idx][2] - 2*_sb_mul_rb[idx][2];

          _out[idx][0] = _sc_shr[idx][0] - _rc_shr[idx][0] + (_overflow[idx][0] << (k - bits - 1));
          _out[idx][1] = _sc_shr[idx][1] - _rc_shr[idx][1] + (_overflow[idx][1] << (k - bits - 1));
          _out[idx][2] = _sc_shr[idx][2] - _rc_shr[idx][2] + (_overflow[idx][2] << (k - bits - 1));

        });
    }

    if(rank == (size_t)1){
        // -------------------------------------
        // Step 1: Generate r and rb, rc
        // -------------------------------------
        std::vector<el_t> r1(out.numel());
        std::vector<el_t> r2(out.numel());
        // std::vector<el_t> r3(output.numel());
        prg_state->fillPrssTuple<el_t>(r1.data(), nullptr, nullptr , r1.size(),
                                PrgState::GenPrssCtrl::First);
        prg_state->fillPrssTuple<el_t>(nullptr, r2.data(), nullptr, r2.size(),
                                PrgState::GenPrssCtrl::Second);

        std::vector<el_t> r(out.numel());
        std::vector<el_t> rb(out.numel());
        std::vector<el_t> rc(out.numel());
        
        pforeach(0, out.numel(), [&](int64_t idx) {
          // r = r_{k-1}......r_{0}
          r[idx] = r1[idx] + r2[idx];
          // rb = r >> k-1
          rb[idx] = r[idx] >> (k-1);
          // rc = r_{k-2}.....r_{m}
          rc[idx] = (r[idx] << 1) >> (bits + 1);
        });

        // -------------------------------------
        // Step 2: Generate the share of rb, rc
        // -------------------------------------
        JointInputArith(ctx, rb, rb_shr, 0, 1, 3, 2);
        JointInputArith(ctx, rc, rc_shr, 0, 1, 3, 2);

        // -------------------------------------
        // Step 3: compute [x] + [r]
        //          [r] = r0 + r1 + r2 + r3, only r1 and r2 are non-zero
        // -------------------------------------
        std::vector<el_t> masked_input_shr_1(out.numel());
        pforeach(0, out.numel(), [&](int64_t idx) {
          _masked_input[idx][0] = _in[idx][0] + r1[idx];
          _masked_input[idx][1] = _in[idx][1] + r2[idx];
          _masked_input[idx][2] = _in[idx][2];
          masked_input_shr_1[idx] = _masked_input[idx][0];
        });

        // -------------------------------------
        // Step 4: Let P2 and P3 reconstruct s = x + r
        //         by P1 sends s1 to P2
        //            P2 sends s2 to P3
        // -------------------------------------
        comm->sendAsync<el_t>(2, masked_input_shr_1, "masked shr 1"); 

        // -------------------------------------
        // Step 5: compute sb = s{k-1} and sc = s{k-2}.....s{m}
        // -------------------------------------
        std::vector<el_t> sb(out.numel());
        std::vector<el_t> sc(out.numel());
        JointInputArith(ctx, sb, sb_shr, 2, 3, 0, 1);
        JointInputArith(ctx, sc, sc_shr, 2, 3, 0, 1);

        // -------------------------------------
        // Step 6: compute sb = s{k-1} and sc = s{k-2}.....s{m}
        // -------------------------------------
        auto sb_mul_rb = wrap_mul_aa(ctx->sctx(), sb_shr, rb_shr);
        NdArrayView<shr_t> _sb_mul_rb(sb_mul_rb);
        pforeach(0, out.numel(), [&](int64_t idx) {
          _overflow[idx][0] = _rb_shr[idx][0] + _sb_shr[idx][0] - 2*_sb_mul_rb[idx][0];
          _overflow[idx][1] = _rb_shr[idx][1] + _sb_shr[idx][1] - 2*_sb_mul_rb[idx][1];
          _overflow[idx][2] = _rb_shr[idx][2] + _sb_shr[idx][2] - 2*_sb_mul_rb[idx][2];
          
          _out[idx][0] = _sc_shr[idx][0] - _rc_shr[idx][0] + (_overflow[idx][0] << (k - bits - 1));
          _out[idx][1] = _sc_shr[idx][1] - _rc_shr[idx][1] + (_overflow[idx][1] << (k - bits - 1));
          _out[idx][2] = _sc_shr[idx][2] - _rc_shr[idx][2] + (_overflow[idx][2] << (k - bits - 1));
          
        });  
    }

    if(rank == (size_t)2){
        std::vector<el_t> r2(out.numel());
        // std::vector<el_t> r3(out.numel());
        // std::vector<el_t> r0(out.numel());
        std::vector<el_t> rb(out.numel());
        std::vector<el_t> rc(out.numel());
        prg_state->fillPrssTuple<el_t>(r2.data(), nullptr, nullptr, r2.size(),
                                PrgState::GenPrssCtrl::First);
        
        // -------------------------------------
        // Step 2: Generate the share of rb, rc
        // -------------------------------------
        JointInputArith(ctx, rb, rb_shr, 0, 1, 3, 2);
        JointInputArith(ctx, rc, rc_shr, 0, 1, 3, 2);

        // -------------------------------------
        // Step 3: compute [x] + [r]
        //          [r] = r0 + r1 + r2 + r3, only r1 and r2 are non-zero
        // -------------------------------------
        std::vector<el_t> masked_input_shr_2(out.numel());
        pforeach(0, out.numel(), [&](int64_t idx) {
          _masked_input[idx][0] = _in[idx][0] + r2[idx];
          _masked_input[idx][1] = _in[idx][1];
          _masked_input[idx][2] = _in[idx][2];

          masked_input_shr_2[idx] = _masked_input[idx][0];
        });

        // -------------------------------------
        // Step 4: Let P2 and P3 reconstruct s = x + r
        //         by P1 sends s1 to P2
        //            P2 sends s2 to P3
        // -------------------------------------
        comm->sendAsync<el_t>(3, masked_input_shr_2, "masked shr 2");
        auto missing_shr = comm->recv<el_t>(1, "masked shr 1");
        std::vector<el_t> s(out.numel());
        pforeach(0, out.numel(), [&](int64_t idx) {
          s[idx] = _masked_input[idx][0] + _masked_input[idx][1] + _masked_input[idx][2] + missing_shr[idx];
        });

        // -------------------------------------
        // Step 5: compute sb = s{k-1} and sc = s{k-2}.....s{m}
        // -------------------------------------
        std::vector<el_t> sb(out.numel());
        std::vector<el_t> sc(out.numel());
        pforeach(0, out.numel(), [&](int64_t idx) {
          sb[idx] = s[idx] >> (k-1);
          sc[idx] = (s[idx] << 1) >> (bits + 1);
        });
        JointInputArith(ctx, sb, sb_shr, 2, 3, 0, 1);
        JointInputArith(ctx, sc, sc_shr, 2, 3, 0, 1);

        // -------------------------------------
        // Step 6: compute sb = s{k-1} and sc = s{k-2}.....s{m}
        // -------------------------------------
        auto sb_mul_rb = wrap_mul_aa(ctx->sctx(), sb_shr, rb_shr);
        NdArrayView<shr_t> _sb_mul_rb(sb_mul_rb);
        pforeach(0, out.numel(), [&](int64_t idx) {
          _overflow[idx][0] = _rb_shr[idx][0] + _sb_shr[idx][0] - 2*_sb_mul_rb[idx][0];
          _overflow[idx][1] = _rb_shr[idx][1] + _sb_shr[idx][1] - 2*_sb_mul_rb[idx][1];
          _overflow[idx][2] = _rb_shr[idx][2] + _sb_shr[idx][2] - 2*_sb_mul_rb[idx][2];

          _out[idx][0] = _sc_shr[idx][0] - _rc_shr[idx][0] + (_overflow[idx][0] << (k - bits - 1));
          _out[idx][1] = _sc_shr[idx][1] - _rc_shr[idx][1] + (_overflow[idx][1] << (k - bits - 1));
          _out[idx][2] = _sc_shr[idx][2] - _rc_shr[idx][2] + (_overflow[idx][2] << (k - bits - 1));
        });                 
    }

    if(rank == (size_t)3){
        // std::vector<el_t> r3(out.numel());
        // std::vector<el_t> r0(out.numel());
        std::vector<el_t> r1(out.numel());
        std::vector<el_t> rb(out.numel());
        std::vector<el_t> rc(out.numel());
        prg_state->fillPrssTuple<el_t>(nullptr, nullptr, r1.data(), r1.size(),
                                PrgState::GenPrssCtrl::Third);

        // -------------------------------------
        // Step 2: Generate the share of rb, rc
        // -------------------------------------
        JointInputArith(ctx, rb, rb_shr, 0, 1, 3, 2);
        JointInputArith(ctx, rc, rc_shr, 0, 1, 3, 2);

        // -------------------------------------
        // Step 3: compute [x] + [r]
        //          [r] = r0 + r1 + r2 + r3, only r1 and r2 are non-zero
        // -------------------------------------
        pforeach(0, out.numel(), [&](int64_t idx) {
          _masked_input[idx][0] = _in[idx][0];
          _masked_input[idx][1] = _in[idx][1];
          _masked_input[idx][2] = _in[idx][2] + r1[idx];
        });

        // -------------------------------------
        // Step 4: Let P2 and P3 reconstruct s = x + r
        //         by P1 sends s1 to P2
        //            P2 sends s2 to P3
        // -------------------------------------
        auto missing_shr = comm->recv<el_t>(2, "masked shr 2");
        std::vector<el_t> s(out.numel());
        pforeach(0, out.numel(), [&](int64_t idx) {
          s[idx] = _masked_input[idx][0] + _masked_input[idx][1] + _masked_input[idx][2] + missing_shr[idx];
        });

        // -------------------------------------
        // Step 5: compute sb = s{k-1} and sc = s{k-2}.....s{m}
        // -------------------------------------
        std::vector<el_t> sb(out.numel());
        std::vector<el_t> sc(out.numel());
        pforeach(0, out.numel(), [&](int64_t idx) {
          sb[idx] = s[idx] >> (k-1);
          sc[idx] = (s[idx] << 1) >> (bits + 1);
        });
        JointInputArith(ctx, sb, sb_shr, 2, 3, 0, 1);
        JointInputArith(ctx, sc, sc_shr, 2, 3, 0, 1);

        // -------------------------------------
        // Step 6: compute sb = s{k-1} and sc = s{k-2}.....s{m}
        // -------------------------------------
        auto sb_mul_rb = wrap_mul_aa(ctx->sctx(), sb_shr, rb_shr);
        NdArrayView<shr_t> _sb_mul_rb(sb_mul_rb);
        pforeach(0, out.numel(), [&](int64_t idx) {
          _overflow[idx][0] = _rb_shr[idx][0] + _sb_shr[idx][0] - 2*_sb_mul_rb[idx][0];
          _overflow[idx][1] = _rb_shr[idx][1] + _sb_shr[idx][1] - 2*_sb_mul_rb[idx][1];
          _overflow[idx][2] = _rb_shr[idx][2] + _sb_shr[idx][2] - 2*_sb_mul_rb[idx][2];

          _out[idx][0] = _sc_shr[idx][0] - _rc_shr[idx][0] + (_overflow[idx][0] << (k - bits - 1));
          _out[idx][1] = _sc_shr[idx][1] - _rc_shr[idx][1] + (_overflow[idx][1] << (k - bits - 1));
          _out[idx][2] = _sc_shr[idx][2] - _rc_shr[idx][2] + (_overflow[idx][2] << (k - bits - 1));
        });                                    
    }

    return out;
  });
}


} // namespace spu::mpc::fantastic4