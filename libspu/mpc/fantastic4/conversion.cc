#include "libspu/mpc/fantastic4/conversion.h"

#include <functional>

#include "yacl/utils/platform_utils.h"

#include "libspu/core/parallel_utils.h"
#include "libspu/core/prelude.h"
#include "libspu/core/trace.h"
#include "libspu/mpc/ab_api.h"
#include "libspu/mpc/fantastic4/type.h"
#include "libspu/mpc/fantastic4/value.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::fantastic4 {

namespace {

  size_t PrevRankC(size_t rank, size_t world_size){
    return (rank + world_size -1) % world_size;
  }

  size_t OffsetRankC(size_t myrank, size_t other, size_t world_size){
    size_t offset = (myrank + world_size -other) % world_size;
    if(offset == 3){
      offset = 1;
    }
    return offset;
  }

    template <typename el_t>
  void JointInputArithmetic(KernelEvalContext* ctx, const std::vector<el_t>& input, NdArrayRef& output, size_t sender, size_t backup, size_t receiver, size_t outsider){
    auto* comm = ctx->getState<Communicator>();
    size_t world_size =  comm->getWorldSize();
    auto* prg_state = ctx->getState<PrgState>();
    auto myrank = comm->getRank();

    using shr_t = std::array<el_t, 3>;
    NdArrayView<shr_t> _out(output);
    
    // Receiver's Previous Party Rank
    // The mask corresponds to the prev party of receiver, receiver doesn't have the correpsonding PRG of its prev party
    size_t receiver_prev_rank = PrevRankC(receiver, world_size);

    // My offset from the receiver_prev_rank. 
    // 0- i'm the receiver_prev_rank
    // 1- i'm prev/next party of receiver_prev_rank
    // 2- next next
    size_t offset_from_receiver_prev = OffsetRankC(myrank, receiver_prev_rank, world_size);
    // size_t offset_from_receiver = OffsetRank(myrank, receiver, world_size);
    size_t offset_from_outsider_prev = OffsetRankC(myrank, (outsider + 4 - 1)%4 , world_size);

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

  template <typename el_t>
  void JointInputBoolean(KernelEvalContext* ctx, std::vector<el_t>& input, NdArrayRef& output, size_t sender, size_t backup, size_t receiver, size_t outsider){
    auto* comm = ctx->getState<Communicator>();
    size_t world_size =  comm->getWorldSize();
    auto* prg_state = ctx->getState<PrgState>();
    auto myrank = comm->getRank();
    
    // SPU_ENFORCE_EQ(input.size(), output.numel());
    // SPU_ENFORCE_EQ(row * col, output.numel());

    using shr_t = std::array<el_t, 3>;
    NdArrayView<shr_t> _out(output);
    
    // Receiver's Previous Party Rank
    // The mask corresponds to the prev party of receiver, receiver doesn't have the correpsonding PRG of its prev party
    size_t receiver_prev_rank = PrevRankC(receiver, world_size);

    // My offset from the receiver_prev_rank. 
    // 0- i'm the receiver_prev_rank
    // 1- i'm prev/next party of receiver_prev_rank
    // 2- next next
    size_t offset_from_receiver_prev = OffsetRankC(myrank, receiver_prev_rank, world_size);
    // size_t offset_from_receiver = OffsetRank(myrank, receiver, world_size);
    size_t offset_from_outsider_prev = OffsetRankC(myrank, (outsider + 4 - 1)%4 , world_size);

    // printf("My rank = %zu, sender_rank = %zu, receiver_rank = %zu, receiver_prev = %zu, offset_from_recv_prev = %zu, offset_from_outsider_prev = %zu \n", myrank, sender, receiver, receiver_prev_rank, offset_from_receiver_prev, offset_from_outsider_prev);
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
          _out[idx][offset_from_receiver_prev] ^= r[idx];
      }); 

      if(myrank != outsider){

        std::vector<el_t> input_minus_r(output.numel());

        // For sender, backup
        // compute and set masked input x-r
        pforeach(0, output.numel(), [&](int64_t idx) {
          input_minus_r[idx] = (input[idx] ^ r[idx]);
          _out[idx][offset_from_outsider_prev] ^=  input_minus_r[idx];
          
          // printf("My rank = %zu, sender_rank = %zu, receiver_rank = %zu, receiver_prev = %zu, offset_from_recv_prev = %zu, offset_from_outsider_prev = %zu, x = %llu, r = %llu, x-r = %llu \n", myrank, sender, receiver, receiver_prev_rank, offset_from_receiver_prev, offset_from_outsider_prev, (unsigned long long)input[idx], (unsigned long long)r[idx], (unsigned long long)input_minus_r[idx]);
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
          _out[idx][offset_from_outsider_prev] ^= input_minus_r[idx];
      }); 

      // Todo: 
      // Mac update sender-backup channel
    }

  }
}

static NdArrayRef wrap_add_bb(SPUContext* ctx, const NdArrayRef& x,
                              const NdArrayRef& y) {
  SPU_ENFORCE(x.shape() == y.shape());
  return UnwrapValue(add_bb(ctx, WrapValue(x), WrapValue(y)));
}

static NdArrayRef wrap_and_bb(SPUContext* ctx, const NdArrayRef& x,
                              const NdArrayRef& y) {
  SPU_ENFORCE(x.shape() == y.shape());
  return UnwrapValue(and_bb(ctx, WrapValue(x), WrapValue(y)));
}

// Optimized A2B By Ranyang Liu, Nankai University
// The semi-honest A2B of ABY3 fails in malicious settings since the corrupted party could contribute x1 + x2 + err while the honest party
// Fantastic4 adopts the malicious method of ABY3 that leverages local share conversion with FA and Binary Adders
// We take the advantage of 4-party replicated secret sharing
// Let
//      (P0, P1) share (x1 + x2)
//      (P2, P3) share (x0 + x3)
//      Jointly Compute Binary Adder, e.g. PPA((x1 + x2), (x0 + x3)) without the need of tree reduction (2k Full Adders)

NdArrayRef A2B::proc(KernelEvalContext* ctx, const NdArrayRef& in) const {
  const auto field = in.eltype().as<Ring2k>()->field();

  auto* comm = ctx->getState<Communicator>();

  auto rank = comm->getRank();

  const PtType out_btype = calcBShareBacktype(SizeOf(field) * 8);
  const auto out_ty = makeType<BShrTy>(out_btype, SizeOf(out_btype) * 8);
  NdArrayRef m(out_ty, in.shape());
  NdArrayRef n(out_ty, in.shape());

  auto numel = in.numel();

  DISPATCH_ALL_FIELDS(field, [&]() {
    using ashr_t = std::array<ring2k_t, 3>;
    NdArrayView<ashr_t> _in(in);

    DISPATCH_UINT_PT_TYPES(out_btype, [&]() {
      using bshr_el_t = ScalarT;
      using bshr_t = std::array<bshr_el_t, 3>;

      NdArrayView<bshr_t> _m(m);
      NdArrayView<bshr_t> _n(n);

      std::vector<bshr_el_t> half0(numel);
      std::vector<bshr_el_t> half1(numel);
      pforeach(0, numel, [&](int64_t idx) {
        half0[idx] = 0U;
        
       
        half1[idx] = 0U;

        _m[idx][0] = 0U;
        _m[idx][1] = 0U;
        _m[idx][2] = 0U;
        _n[idx][0] = 0U;
        _n[idx][1] = 0U;
        _n[idx][2] = 0U;
      });
      if(rank == 0){
        pforeach(0, numel, [&](int64_t idx) { 
            half0[idx] ^= _in[idx][1] + _in[idx][2]; 
        });
      }
      else if(rank == 1){
        pforeach(0, numel, [&](int64_t idx) { 
            half0[idx] ^= _in[idx][0] + _in[idx][1]; 
        });
      }
      else if(rank == 2){
        pforeach(0, numel, [&](int64_t idx) { 
            half1[idx] ^= _in[idx][1] + _in[idx][2]; 
        });
      }
      else if(rank == 3){
        pforeach(0, numel, [&](int64_t idx) { 
            half1[idx] ^= _in[idx][0] + _in[idx][1]; 
        });
      }
      JointInputBoolean(ctx, half0, m, 0, 1, 2, 3);
      JointInputBoolean(ctx, half1, n, 3, 2, 1, 0);
    });
  });

  return wrap_add_bb(ctx->sctx(), m, n);  // comm => log(k) + 1, 2k(logk) + k
}

// Optimized B2A By Ranyang Liu, Nankai University
// Fantastic4 gives the 4-party generation of edabits based on its A2B
// Given edabitsthe B2A can be implemented using Binary Adders
// We take the advantage of 4-party replicated secret sharing
// We do not require r to be unknown to all party, instead r only unknown to other group
// Let
//      (P0, P1) share random r in both arithmetic and boolean form
//      Jointly Compute Binary Adder to obtain [x + r]_B and reveal x + r to (P2, P3)
//      (P2, P3) share x + r in arithmetic form
//      All parties set [x]_A = [x + r]_A - [r]_A

NdArrayRef B2A::proc(KernelEvalContext* ctx, const NdArrayRef& in) const {
  const auto field = ctx->getState<Z2kState>()->getDefaultField();
  const auto* in_ty = in.eltype().as<BShrTy>();
  const size_t in_nbits = in_ty->nbits();

  SPU_ENFORCE(in_nbits <= SizeOf(field) * 8, "invalid nbits={}", in_nbits);
  const auto out_ty = makeType<AShrTy>(field);
  NdArrayRef out(out_ty, in.shape());

  auto numel = in.numel();

  if (in_nbits == 0) {
    // special case, it's known to be zero.
    DISPATCH_ALL_FIELDS(field, [&]() {
      NdArrayView<std::array<ring2k_t, 2>> _out(out);
      pforeach(0, numel, [&](int64_t idx) {
        _out[idx][0] = 0;
        _out[idx][1] = 0;
      });
    });
    return out;
  }

  auto* comm = ctx->getState<Communicator>();
  auto* prg_state = ctx->getState<PrgState>();

  DISPATCH_UINT_PT_TYPES(in_ty->getBacktype(), [&]() {
    using bshr_t = std::array<ScalarT, 3>;
    NdArrayView<bshr_t> _in(in);

    DISPATCH_ALL_FIELDS(field, [&]() {
      using ashr_el_t = ring2k_t;
      using ashr_t = std::array<ashr_el_t, 3>;

      // first expand b share to a share length.
      const auto expanded_ty = makeType<BShrTy>(
          calcBShareBacktype(SizeOf(field) * 8), SizeOf(field) * 8);
      NdArrayRef x(expanded_ty, in.shape());
      NdArrayView<ashr_t> _x(x);

      pforeach(0, numel, [&](int64_t idx) {
        const auto& v = _in[idx];
        _x[idx][0] = v[0];
        _x[idx][1] = v[1];
        _x[idx][2] = v[2];
      });
      
      // P0 & P1 invoke PRG[1], PRG[2]
      // P2 invoke PRG[2], P3 invoke PRG[1]
      std::vector<ashr_el_t> r1(numel);
      std::vector<ashr_el_t> r2(numel);
      std::vector<ashr_el_t> r(numel);
      std::vector<ashr_el_t> neg_r(numel);

      NdArrayRef neg_r_shr(expanded_ty, in.shape());
      NdArrayView<ashr_t> _neg_r_shr(neg_r_shr);

      NdArrayRef r_shr(expanded_ty, in.shape());
      NdArrayView<ashr_t> _r_shr(r_shr);

      NdArrayRef x_minus_r_shr(expanded_ty, in.shape());
      NdArrayView<ashr_t> _x_minus_r_shr(x_minus_r_shr);

      pforeach(0, numel, [&](int64_t idx) {
        _neg_r_shr[idx][0] = 0U;
        _neg_r_shr[idx][1] = 0U;
        _neg_r_shr[idx][2] = 0U;

        _r_shr[idx][0] = 0U;
        _r_shr[idx][1] = 0U;
        _r_shr[idx][2] = 0U;

        _x_minus_r_shr[idx][0] = 0U;
        _x_minus_r_shr[idx][1] = 0U;
        _x_minus_r_shr[idx][2] = 0U;
      });

      if (comm->getRank() == 0) {
        // Sample r1, r2
        prg_state->fillPrssTuple<ashr_el_t>(nullptr, r1.data(), nullptr, r1.size(),
                              PrgState::GenPrssCtrl::Second);
        prg_state->fillPrssTuple<ashr_el_t>(nullptr, nullptr, r2.data(), r2.size(),
                              PrgState::GenPrssCtrl::Third);
        // r = r1 + r2
        pforeach(0, numel, [&](int64_t idx) {
            r[idx] = r1[idx] + r2[idx];
            neg_r[idx] = - r[idx];
        });

      } else if (comm->getRank() == 1) {

        prg_state->fillPrssTuple<ashr_el_t>(r1.data(), nullptr, nullptr, r1.size(),
                              PrgState::GenPrssCtrl::First);
        prg_state->fillPrssTuple<ashr_el_t>(nullptr, r2.data(), nullptr, r2.size(),
                              PrgState::GenPrssCtrl::Second);

        pforeach(0, numel, [&](int64_t idx) {
            r[idx] = r1[idx] + r2[idx];
            neg_r[idx] = - r[idx];
        });

      } else if (comm->getRank() == 2) {

        prg_state->fillPrssTuple<ashr_el_t>(r2.data(), nullptr, nullptr, r2.size(),
                              PrgState::GenPrssCtrl::First);

      } else if (comm->getRank() == 3) {

        prg_state->fillPrssTuple<ashr_el_t>(nullptr, nullptr, r1.data(), r1.size(),
                              PrgState::GenPrssCtrl::Third);

      }
      
      // P0, P1 share [-r]B
      JointInputArithmetic(ctx, r, r_shr, 0, 1, 2, 3);

      JointInputBoolean(ctx, neg_r, neg_r_shr, 0, 1, 2, 3);

      // compute [x-r]B
      // comm => log(k) + 1, 2k(logk) + k
      auto x_minus_r = wrap_add_bb(ctx->sctx(), x, neg_r_shr);

      // reveal x-r to P2, P3
      // todo: MAC
      NdArrayView<ashr_t> _x_minus_r(x_minus_r);

      std::vector<ashr_el_t> plaintext_x_minus_r(numel);

      if (comm->getRank() == 2) {
        // P2 send global shr[2] (own::shr[0]) to P3
        std::vector<ashr_el_t> shr_for_P3(numel);
        pforeach(0, numel,
                 [&](int64_t idx) { shr_for_P3[idx] = _x_minus_r[idx][0]; });
        comm->sendAsync<ashr_el_t>(3, shr_for_P3, "reveal.x_minus_r.to.P3");

        std::vector<ashr_el_t> missing_shr = comm->recv<ashr_el_t>(3, "reveal.x_minus_r.to.P2");
        
        pforeach(0, numel,
                 [&](int64_t idx) { plaintext_x_minus_r[idx] = _x_minus_r[idx][0] ^ _x_minus_r[idx][1] ^ _x_minus_r[idx][2] ^ missing_shr[idx]; });

      }
      if (comm->getRank() == 3) {
        // P3 send global shr[1] (own::shr[2]) to P2
        std::vector<ashr_el_t> shr_for_P2(numel);
        pforeach(0, numel,
                 [&](int64_t idx) { shr_for_P2[idx] = _x_minus_r[idx][2]; });
        comm->sendAsync<ashr_el_t>(2, shr_for_P2, "reveal.x_minus_r.to.P2");

        std::vector<ashr_el_t> missing_shr = comm->recv<ashr_el_t>(2, "reveal.x_minus_r.to.P3");

        pforeach(0, numel,
                 [&](int64_t idx) { plaintext_x_minus_r[idx] = _x_minus_r[idx][0] ^ _x_minus_r[idx][1] ^ _x_minus_r[idx][2] ^ missing_shr[idx]; });

      }

      JointInputArithmetic(ctx, plaintext_x_minus_r, x_minus_r_shr, 2, 3, 0, 1);

      NdArrayView<ashr_t> _out(out);
      pforeach(0, numel, [&](int64_t idx) {
        _out[idx][0] = _x_minus_r_shr[idx][0] + _r_shr[idx][0];
        _out[idx][1] = _x_minus_r_shr[idx][1] + _r_shr[idx][1];
        _out[idx][2] = _x_minus_r_shr[idx][2] + _r_shr[idx][2];
      });

    });
  });
  return out;
}


// Optimized EQZ By Ranyang Liu, Nankai University
// Fantastic4 gives the 4-party generation of edabits based on its A2B
// Given edabitsthe EQZ can be implemented using Binary Adders
// We take the advantage of 4-party replicated secret sharing
// Each group holds half of x, if x = 0, then they are negative of each other
// Each group contributes it's half securely, otherwise the malicious party will be detected
// Let
//      (P0, P1) share ~(x1 + x2)
//      (P2, P3) share -(x0 + x3)
//      Locally Compute [s] = [~(x1 + x2) ^ -(x0 + x3)]
//      Jointly Compute AND of all bits of s, [s_0] & .... & [s_k-1] (k AND gates, logk rounds)

NdArrayRef eqz(KernelEvalContext* ctx, const NdArrayRef& in) {
  // auto* prg_state = ctx->getState<PrgState>();
  auto* comm = ctx->getState<Communicator>();

  const auto field = in.eltype().as<AShrTy>()->field();
  const PtType in_bshr_btype = calcBShareBacktype(SizeOf(field) * 8);
  const auto numel = in.numel();

  // uint 8
  NdArrayRef out(makeType<BShrTy>(calcBShareBacktype(8), 8), in.shape());

  DISPATCH_ALL_FIELDS(field, [&]() {
    using ashr_el_t = ring2k_t;
    using ashr_t = std::array<ashr_el_t, 3>;

    NdArrayView<ashr_t> _in(in);

    DISPATCH_UINT_PT_TYPES(in_bshr_btype, [&]() {
      using bshr_el_t = ScalarT;
      using bshr_t = std::array<bshr_el_t, 3>;

      NdArrayRef first_half(makeType<BShrTy>(calcBShareBacktype(field), 8), in.shape());
      NdArrayView<bshr_t> _first_half(first_half);
      NdArrayRef second_half(makeType<BShrTy>(calcBShareBacktype(field), 8), in.shape());
      NdArrayView<bshr_t> _second_half(second_half);

      NdArrayRef test_all_one(makeType<BShrTy>(calcBShareBacktype(field), 8), in.shape());
      NdArrayView<bshr_t> _test_all_one(test_all_one);

      std::vector<bshr_el_t> plaintext_first_half(numel);
      std::vector<bshr_el_t> plaintext_second_half(numel);

      pforeach(0, numel, [&](int64_t idx) {
        _first_half[idx][0] = 0U;
        _first_half[idx][1] = 0U;
        _first_half[idx][2] = 0U;

        _second_half[idx][0] = 0U;
        _second_half[idx][1] = 0U;
        _second_half[idx][2] = 0U;

        _test_all_one[idx][0] = 0U;
        _test_all_one[idx][1] = 0U;
        _test_all_one[idx][2] = 0U;

      });
      
      if (comm->getRank() == 0) {
        pforeach(0, numel, [&](int64_t idx) {
          // ~(x1 + x2)
          plaintext_first_half[idx] = ~(_in[idx][1] + _in[idx][2]);
        });
      }
      if (comm->getRank() == 1) {
        pforeach(0, numel, [&](int64_t idx) {
          // ~(x1 + x2)
          plaintext_first_half[idx] = ~(_in[idx][0] + _in[idx][1]);
        });
      }
      if (comm->getRank() == 2) {
        pforeach(0, numel, [&](int64_t idx) {
          // -(x3 + x0)
          plaintext_second_half[idx] = -(_in[idx][1] + _in[idx][2]);
        });
      }
      if (comm->getRank() == 3) {
        pforeach(0, numel, [&](int64_t idx) {
          // -(x3 + x0)
          plaintext_second_half[idx] = -(_in[idx][0] + _in[idx][1]);
        });
      }
      // [m]B = [~(x1 + x2)]B
      // [n]B = [-(x0 + x3)]B
      // if x = 0, x1 + x2 = -(x0 + x3)
      //   < ---- >(x1 + x2)[i] == (- x0 - x3)[i], for all i
      //   < ---- >(x1 + x2)[i] XOR (- x0 - x3)[i] == 0, for all i
      //   < ---- >~(x1 + x2)[i] XOR (- x0 - x3)[i] == 1, for all i
      JointInputBoolean(ctx, plaintext_first_half, first_half, 0, 1, 2, 3);
      JointInputBoolean(ctx, plaintext_second_half, second_half, 2, 3, 0, 1);


      
      // TODO:
      // For each element, the equality can be tested by AND all the bits of ~(x1 + x2) ^ (- x0 - x3)

      pforeach(0, numel, [&](int64_t idx) {
        _test_all_one[idx][0] = _first_half[idx][0] ^ _second_half[idx][0];
        _test_all_one[idx][1] = _first_half[idx][1] ^ _second_half[idx][1];
        _test_all_one[idx][2] = _first_half[idx][2] ^ _second_half[idx][2];

      });

      // test_all_one[idx]_0 & test_all_one[idx]_1 & .... test_all_one[idx]_k-1
      // Require k-1 AND Gates, log_2 k Rounds
      // or log_4 k rounds with 4-input AND gates


      if(comm->getRank() == 0 || comm->getRank() == 2 ) {
        printf("My rank = %zu, ", comm->getRank());
        for(auto idx = 0; idx < numel; idx ++) {
          printf("input shares = (%llu, %llu, %llu), flags = (%llu, %llu, %llu), xor results = %llu \n", 
            (unsigned long long)_in[idx][0], (unsigned long long)_in[idx][1], (unsigned long long)_in[idx][2], (unsigned long long)_test_all_one[idx][0], (unsigned long long)_test_all_one[idx][1], (unsigned long long)_test_all_one[idx][2], 
            (unsigned long long) (_test_all_one[idx][0] ^ _test_all_one[idx][1] ^ _test_all_one[idx][2]));
        }
      }


      
      // NdArrayView<std::array<std::byte, 3>> _out(out);

      std::vector<NdArrayRef> decomposed_bit_shrs;

      int bit_len = SizeOf(field) * 8;

      for(int bit_idx = 0; bit_idx < bit_len; bit_idx ++) {
        NdArrayRef cur_bit_shr(makeType<BShrTy>(calcBShareBacktype(8), 8), in.shape());
        NdArrayView<std::array<std::byte, 3>> _cur_bit_shr(cur_bit_shr);

        pforeach(0, numel, [&](int64_t idx) {
          _cur_bit_shr[idx][0] = (std::byte)((_test_all_one[idx][0] >> bit_idx) & 1);
          _cur_bit_shr[idx][1] = (std::byte)((_test_all_one[idx][1] >> bit_idx) & 1);
          _cur_bit_shr[idx][2] = (std::byte)((_test_all_one[idx][2] >> bit_idx) & 1);
          if( (comm->getRank() == 0 || comm->getRank() == 2) && (bit_idx < 2) ) {
            printf("My rank = %zu, ", comm->getRank());
            printf("flags = (%llu, %llu, %llu), decomposed flag[bit_idx] shares = (%u, %u, %u) \n", 
              (unsigned long long)_test_all_one[idx][0], (unsigned long long)_test_all_one[idx][1], (unsigned long long)_test_all_one[idx][2], 
              (unsigned int)_cur_bit_shr[idx][0], (unsigned int)_cur_bit_shr[idx][1], (unsigned int)_cur_bit_shr[idx][2]);
          }
        });
        decomposed_bit_shrs.push_back(cur_bit_shr);

      }

      out = vreduce(decomposed_bit_shrs.begin(), decomposed_bit_shrs.end(),
                           [&](const NdArrayRef& xx, const NdArrayRef& yy) {
                             return wrap_and_bb(ctx->sctx(), xx, yy);
                           });
      
    });
  });

  return out;
}

NdArrayRef EqualAP::proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                         const NdArrayRef& rhs) const {
  auto* comm = ctx->getState<Communicator>();
  const auto* lhs_ty = lhs.eltype().as<AShrTy>();
  const auto* rhs_ty = rhs.eltype().as<Pub2kTy>();

  SPU_ENFORCE(lhs_ty->field() == rhs_ty->field());
  const auto field = lhs_ty->field();
  NdArrayRef out(makeType<AShrTy>(field), lhs.shape());

  auto rank = comm->getRank();

  DISPATCH_ALL_FIELDS(field, [&]() {
    using el_t = ring2k_t;
    using shr_t = std::array<el_t, 3>;

    NdArrayView<shr_t> _out(out);
    NdArrayView<shr_t> _lhs(lhs);
    NdArrayView<el_t> _rhs(rhs);

    pforeach(0, lhs.numel(), [&](int64_t idx) {
      _out[idx][0] = _lhs[idx][0];
      _out[idx][1] = _lhs[idx][1];
      if (rank == 0) _out[idx][0] -= _rhs[idx];
      if (rank == 2) _out[idx][2] -= _rhs[idx];
      if (rank == 3) _out[idx][1] -= _rhs[idx];
    });
    return out;
  });

  return eqz(ctx, out);
}

NdArrayRef EqualAA::proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                         const NdArrayRef& rhs) const {
  const auto* lhs_ty = lhs.eltype().as<AShrTy>();
  const auto* rhs_ty = rhs.eltype().as<AShrTy>();

  SPU_ENFORCE(lhs_ty->field() == rhs_ty->field());
  const auto field = lhs_ty->field();
  NdArrayRef out(makeType<AShrTy>(field), lhs.shape());

  DISPATCH_ALL_FIELDS(field, [&]() {
    using shr_t = std::array<ring2k_t, 3>;
    NdArrayView<shr_t> _out(out);
    NdArrayView<shr_t> _lhs(lhs);
    NdArrayView<shr_t> _rhs(rhs);

    pforeach(0, lhs.numel(), [&](int64_t idx) {
      _out[idx][0] = _lhs[idx][0] - _rhs[idx][0];
      _out[idx][1] = _lhs[idx][1] - _rhs[idx][1];
      _out[idx][2] = _lhs[idx][2] - _rhs[idx][2];
    });
  });

  return eqz(ctx, out);
}


} // namespace spu::mpc::fantastic4
