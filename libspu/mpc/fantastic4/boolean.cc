#include "libspu/mpc/fantastic4/boolean.h"

#include <algorithm>

#include "libspu/core/bit_utils.h"
#include "libspu/core/parallel_utils.h"
#include "libspu/mpc/fantastic4/type.h"
#include "libspu/mpc/fantastic4/value.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/common/pv2k.h"

namespace spu::mpc::fantastic4 {

namespace {


  size_t PrevRankB(size_t rank, size_t world_size){
    return (rank + world_size -1) % world_size;
  }

  size_t OffsetRankB(size_t myrank, size_t other, size_t world_size){
    size_t offset = (myrank + world_size -other) % world_size;
    if(offset == 3){
      offset = 1;
    }
    return offset;
  }

  template <typename el_t>
  void JointInputBool(KernelEvalContext* ctx, std::vector<el_t>& input, NdArrayRef& output, size_t sender, size_t backup, size_t receiver, size_t outsider){
    auto* comm = ctx->getState<Communicator>();
    size_t world_size =  comm->getWorldSize();
    auto* prg_state = ctx->getState<PrgState>();
    auto myrank = comm->getRank();

    using shr_t = std::array<el_t, 3>;
    NdArrayView<shr_t> _out(output);
    
    // Receiver's Previous Party Rank
    // The mask corresponds to the prev party of receiver, receiver doesn't have the correpsonding PRG of its prev party
    size_t receiver_prev_rank = PrevRankB(receiver, world_size);

    // My offset from the receiver_prev_rank. 
    // 0- i'm the receiver_prev_rank
    // 1- i'm prev/next party of receiver_prev_rank
    // 2- next next
    size_t offset_from_receiver_prev = OffsetRankB(myrank, receiver_prev_rank, world_size);
    // size_t offset_from_receiver = OffsetRank(myrank, receiver, world_size);
    size_t offset_from_outsider_prev = OffsetRankB(myrank, (outsider + 4 - 1)%4 , world_size);

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

void CommonTypeB::evaluate(KernelEvalContext* ctx) const {
  const Type& lhs = ctx->getParam<Type>(0);
  const Type& rhs = ctx->getParam<Type>(1);

  const size_t lhs_nbits = lhs.as<BShrTy>()->nbits();
  const size_t rhs_nbits = rhs.as<BShrTy>()->nbits();

  const size_t out_nbits = std::max(lhs_nbits, rhs_nbits);
  const PtType out_btype = calcBShareBacktype(out_nbits);

  ctx->pushOutput(makeType<BShrTy>(out_btype, out_nbits));
}

NdArrayRef CastTypeB::proc(KernelEvalContext*, const NdArrayRef& in,
                           const Type& to_type) const {
  NdArrayRef out(to_type, in.shape());
  DISPATCH_UINT_PT_TYPES(in.eltype().as<BShrTy>()->getBacktype(), [&]() {
    using in_el_t = ScalarT;
    using in_shr_t = std::array<in_el_t, 3>;

    DISPATCH_UINT_PT_TYPES(to_type.as<BShrTy>()->getBacktype(), [&]() {
      using out_el_t = ScalarT;
      using out_shr_t = std::array<out_el_t, 3>;

      NdArrayView<out_shr_t> _out(out);
      NdArrayView<in_shr_t> _in(in);

      pforeach(0, in.numel(), [&](int64_t idx) {
        const auto& v = _in[idx];
        _out[idx][0] = static_cast<out_el_t>(v[0]);
        _out[idx][1] = static_cast<out_el_t>(v[1]);
        _out[idx][2] = static_cast<out_el_t>(v[2]);
      });
    });
  });

  return out;
}

NdArrayRef B2P::proc(KernelEvalContext* ctx, const NdArrayRef& in) const {
  auto* comm = ctx->getState<Communicator>();
  const PtType btype = in.eltype().as<BShrTy>()->getBacktype();
  const auto field = ctx->getState<Z2kState>()->getDefaultField();

  return DISPATCH_UINT_PT_TYPES(btype, [&]() {
    using bshr_el_t = ScalarT;
    using bshr_t = std::array<bshr_el_t, 3>;

    return DISPATCH_ALL_FIELDS(field, [&]() {
      using pshr_el_t = ring2k_t;

      NdArrayRef out(makeType<Pub2kTy>(field), in.shape());

      NdArrayView<pshr_el_t> _out(out);
      NdArrayView<bshr_t> _in(in);

      std::vector<bshr_el_t> x3(in.numel()); 
      pforeach(0, in.numel(), [&](int64_t idx){  x3[idx] = _in[idx][2];  });  
      auto x4 = comm->rotate<bshr_el_t>(x3, "b2p");  // comm => 1, k

      pforeach(0, in.numel(), [&](int64_t idx) {
        const auto& v = _in[idx];
        _out[idx] = static_cast<pshr_el_t>(v[0] ^ v[1] ^ v[2] ^ x4[idx]);
      });

      return out;
    });
  });
}

NdArrayRef P2B::proc(KernelEvalContext* ctx, const NdArrayRef& in) const {
  auto* comm = ctx->getState<Communicator>();
  const auto* in_ty = in.eltype().as<Pub2kTy>();
  const auto field = in_ty->field();
  auto rank = comm->getRank();
  return DISPATCH_ALL_FIELDS(field, [&]() {
    const size_t nbits = maxBitWidth<ring2k_t>(in);
    const PtType btype = calcBShareBacktype(nbits);
    NdArrayView<ring2k_t> _in(in);

    return DISPATCH_UINT_PT_TYPES(btype, [&]() {
      using bshr_el_t = ScalarT;
      using bshr_t = std::array<bshr_el_t, 3>;

      NdArrayRef out(makeType<BShrTy>(btype, nbits), in.shape());
      NdArrayView<bshr_t> _out(out);

      pforeach(0, in.numel(), [&](int64_t idx) {
        _out[idx][0] = rank == 0 ? static_cast<bshr_el_t>(_in[idx]) : 0U;
        _out[idx][1] = rank == 3 ? static_cast<bshr_el_t>(_in[idx]) : 0U;
        _out[idx][2] = rank == 2 ? static_cast<bshr_el_t>(_in[idx]) : 0U;
      });
      return out;
    });
  });
}

NdArrayRef XorBP::proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                       const NdArrayRef& rhs) const {
  auto* comm = ctx->getState<Communicator>();                      
  const auto* lhs_ty = lhs.eltype().as<BShrTy>();
  const auto* rhs_ty = rhs.eltype().as<Pub2kTy>();

  return DISPATCH_ALL_FIELDS(rhs_ty->field(), [&]() {
    using rhs_scalar_t = ring2k_t;

    const size_t rhs_nbits = maxBitWidth<rhs_scalar_t>(rhs);
    const size_t out_nbits = std::max(lhs_ty->nbits(), rhs_nbits);
    const PtType out_btype = calcBShareBacktype(out_nbits);

    NdArrayView<rhs_scalar_t> _rhs(rhs);

    NdArrayRef out(makeType<BShrTy>(out_btype, out_nbits), lhs.shape());
    
    return DISPATCH_UINT_PT_TYPES(lhs_ty->getBacktype(), [&]() {
      using lhs_el_t = ScalarT;
      using lhs_shr_t = std::array<lhs_el_t, 3>;
      auto rank = comm->getRank();
    
      NdArrayView<lhs_shr_t> _lhs(lhs);

      return DISPATCH_UINT_PT_TYPES(out_btype, [&]() {
        using out_el_t = ScalarT;
        using out_shr_t = std::array<out_el_t, 3>;

        NdArrayView<out_shr_t> _out(out);
        pforeach(0, lhs.numel(), [&](int64_t idx) {
          const auto& l = _lhs[idx];
          const auto& r = _rhs[idx];
          _out[idx][0] = l[0];
          _out[idx][1] = l[1];
          _out[idx][2] = l[2];
          if (rank == 0) {_out[idx][0] ^= r;}
          if (rank == 2) {_out[idx][2] ^= r;}
          if (rank == 3) {_out[idx][1] ^= r;}
        });
        return out;
      });
    });
  });
}



NdArrayRef XorBB::proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                       const NdArrayRef& rhs) const {
  const auto* lhs_ty = lhs.eltype().as<BShrTy>();
  const auto* rhs_ty = rhs.eltype().as<BShrTy>();

  const size_t out_nbits = std::max(lhs_ty->nbits(), rhs_ty->nbits());
  const PtType out_btype = calcBShareBacktype(out_nbits);

  return DISPATCH_UINT_PT_TYPES(rhs_ty->getBacktype(), [&]() {
    using rhs_el_t = ScalarT;
    using rhs_shr_t = std::array<rhs_el_t, 3>;

    NdArrayView<rhs_shr_t> _rhs(rhs);

    return DISPATCH_UINT_PT_TYPES(lhs_ty->getBacktype(), [&]() {
      using lhs_el_t = ScalarT;
      using lhs_shr_t = std::array<lhs_el_t, 3>;

      NdArrayView<lhs_shr_t> _lhs(lhs);

      return DISPATCH_UINT_PT_TYPES(out_btype, [&]() {
        using out_el_t = ScalarT;
        using out_shr_t = std::array<out_el_t, 3>;

        NdArrayRef out(makeType<BShrTy>(out_btype, out_nbits), lhs.shape());
        NdArrayView<out_shr_t> _out(out);

        pforeach(0, lhs.numel(), [&](int64_t idx) {
          const auto& l = _lhs[idx];
          const auto& r = _rhs[idx];
          _out[idx][0] = l[0] ^ r[0];
          _out[idx][1] = l[1] ^ r[1];
          _out[idx][2] = l[2] ^ r[2];
        });
        return out;
      });
    });
  });
}

NdArrayRef AndBP::proc(KernelEvalContext*, const NdArrayRef& lhs,
                       const NdArrayRef& rhs) const {
  const auto* lhs_ty = lhs.eltype().as<BShrTy>();
  const auto* rhs_ty = rhs.eltype().as<Pub2kTy>();

  return DISPATCH_ALL_FIELDS(rhs_ty->field(), [&]() {
    using rhs_scalar_t = ring2k_t;

    const size_t rhs_nbits = maxBitWidth<rhs_scalar_t>(rhs);
    const size_t out_nbits = std::min(lhs_ty->nbits(), rhs_nbits);
    const PtType out_btype = calcBShareBacktype(out_nbits);

    NdArrayView<rhs_scalar_t> _rhs(rhs);

    return DISPATCH_UINT_PT_TYPES(lhs_ty->getBacktype(), [&]() {
      using lhs_el_t = ScalarT;
      using lhs_shr_t = std::array<lhs_el_t, 3>;

      NdArrayView<lhs_shr_t> _lhs(lhs);

      return DISPATCH_UINT_PT_TYPES(out_btype, [&]() {
        using out_el_t = ScalarT;
        using out_shr_t = std::array<out_el_t, 3>;

        NdArrayRef out(makeType<BShrTy>(out_btype, out_nbits), lhs.shape());
        NdArrayView<out_shr_t> _out(out);

        pforeach(0, lhs.numel(), [&](int64_t idx) {
          const auto& l = _lhs[idx];
          const auto& r = _rhs[idx];
          _out[idx][0] = l[0] & r;
          _out[idx][1] = l[1] & r;
          _out[idx][2] = l[2] & r;
        });

        return out;
      });
    });
  });
}

NdArrayRef AndBB::proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                       const NdArrayRef& rhs) const {
  auto* comm = ctx->getState<Communicator>();
  auto rank = comm->getRank();
  auto next_rank = (rank + 1) % 4;

  const auto* lhs_ty = lhs.eltype().as<BShrTy>();
  const auto* rhs_ty = rhs.eltype().as<BShrTy>();

  const size_t out_nbits = std::min(lhs_ty->nbits(), rhs_ty->nbits());
  const PtType out_btype = calcBShareBacktype(out_nbits);
  NdArrayRef out(makeType<BShrTy>(out_btype, out_nbits), lhs.shape());

  return DISPATCH_UINT_PT_TYPES(rhs_ty->getBacktype(), [&]() {
    using rhs_el_t = ScalarT;
    using rhs_shr_t = std::array<rhs_el_t, 3>;
    NdArrayView<rhs_shr_t> _rhs(rhs);

    return DISPATCH_UINT_PT_TYPES(lhs_ty->getBacktype(), [&]() {
      using lhs_el_t = ScalarT;
      using lhs_shr_t = std::array<lhs_el_t, 3>;
      NdArrayView<lhs_shr_t> _lhs(lhs);

      return DISPATCH_UINT_PT_TYPES(out_btype, [&]() {
        using out_el_t = ScalarT;
        using out_shr_t = std::array<out_el_t, 3>;

        NdArrayView<out_shr_t> _out(out);
        pforeach(0, lhs.numel(), [&](int64_t idx) {
            for(auto i = 0; i < 3 ; i++ ){
            _out[idx][i] = 0U;
            }
        });

        std::array<std::vector<out_el_t>, 5> a;

        for (auto& vec : a) {
            vec = std::vector<out_el_t>(lhs.numel());
        }
        pforeach(0, lhs.numel(), [&](int64_t idx) {
            for(auto i =0; i<5;i++){
            a[i][idx] = 0U;
            }
        });

        pforeach(0, lhs.numel(), [&](int64_t idx) {
            a[rank][idx] = (_lhs[idx][0] & _rhs[idx][0]) ^ (_lhs[idx][1] & _rhs[idx][0] ) ^ (_lhs[idx][0] & _rhs[idx][1]); // xi&yi ^ xi&yj ^ xj&yi
            a[next_rank][idx] = (_lhs[idx][1]  & _rhs[idx][1] ) ^ (_lhs[idx][2] & _rhs[idx][1] ) ^ (_lhs[idx][1] & _rhs[idx][2]);  // xj&yj ^ xj&yg ^ xg&yj
            a[4][idx] = (_lhs[idx][0] & _rhs[idx][2]) ^ (_lhs[idx][2] & _rhs[idx][0]);                    // xi&yg ^ xg&yi
        });

        JointInputBool<out_el_t>(ctx, a[1], out, 0, 1, 3, 2);
        JointInputBool<out_el_t>(ctx, a[2], out, 1, 2, 0, 3);
        JointInputBool<out_el_t>(ctx, a[3], out, 2, 3, 1, 0);
        JointInputBool<out_el_t>(ctx, a[0], out, 3, 0, 2, 1);
        JointInputBool<out_el_t>(ctx, a[4], out, 0, 2, 3, 1);
        JointInputBool<out_el_t>(ctx, a[4], out, 1, 3, 2, 0);

        return out;
      });
    });
  });
}


NdArrayRef LShiftB::proc(KernelEvalContext* ctx, const NdArrayRef& in,
                         const Sizes& bits) const {
  const auto* in_ty = in.eltype().as<BShrTy>();

  // TODO: the hal dtype should tell us about the max number of possible bits.
  const auto field = ctx->getState<Z2kState>()->getDefaultField();
  const size_t out_nbits = std::min<size_t>(
      in_ty->nbits() + *std::max_element(bits.begin(), bits.end()),
      SizeOf(field) * 8);
  const PtType out_btype = calcBShareBacktype(out_nbits);
  bool is_splat = bits.size() == 1;

  return DISPATCH_UINT_PT_TYPES(in_ty->getBacktype(), [&]() {
    using in_el_t = ScalarT;
    using in_shr_t = std::array<in_el_t, 3>;

    NdArrayView<in_shr_t> _in(in);

    return DISPATCH_UINT_PT_TYPES(out_btype, [&]() {
      using out_el_t = ScalarT;
      using out_shr_t = std::array<out_el_t, 3>;

      NdArrayRef out(makeType<BShrTy>(out_btype, out_nbits), in.shape());
      NdArrayView<out_shr_t> _out(out);

      pforeach(0, in.numel(), [&](int64_t idx) {
        const auto& v = _in[idx];
        auto shift_bit = is_splat ? bits[0] : bits[idx];
        _out[idx][0] = static_cast<out_el_t>(v[0]) << shift_bit;
        _out[idx][1] = static_cast<out_el_t>(v[1]) << shift_bit;
        _out[idx][2] = static_cast<out_el_t>(v[2]) << shift_bit;
      });

      return out;
    });
  });
}

NdArrayRef RShiftB::proc(KernelEvalContext*, const NdArrayRef& in,
                         const Sizes& bits) const {
  const auto* in_ty = in.eltype().as<BShrTy>();

  int64_t out_nbits = in_ty->nbits();
  out_nbits -= std::min(out_nbits, *std::min_element(bits.begin(), bits.end()));
  const PtType out_btype = calcBShareBacktype(out_nbits);
  bool is_splat = bits.size() == 1;

  return DISPATCH_UINT_PT_TYPES(in_ty->getBacktype(), [&]() {
    using in_shr_t = std::array<ScalarT, 3>;
    NdArrayView<in_shr_t> _in(in);

    return DISPATCH_UINT_PT_TYPES(out_btype, [&]() {
      using out_el_t = ScalarT;
      using out_shr_t = std::array<out_el_t, 3>;

      NdArrayRef out(makeType<BShrTy>(out_btype, out_nbits), in.shape());
      NdArrayView<out_shr_t> _out(out);

      pforeach(0, in.numel(), [&](int64_t idx) {
        const auto& v = _in[idx];
        auto shift_bit = is_splat ? bits[0] : bits[idx];
        _out[idx][0] = static_cast<out_el_t>(v[0] >> shift_bit);
        _out[idx][1] = static_cast<out_el_t>(v[1] >> shift_bit);
        _out[idx][2] = static_cast<out_el_t>(v[2] >> shift_bit);
      });

      return out;
    });
  });
}

NdArrayRef ARShiftB::proc(KernelEvalContext* ctx, const NdArrayRef& in,
                          const Sizes& bits) const {
  const auto field = ctx->getState<Z2kState>()->getDefaultField();
  const auto* in_ty = in.eltype().as<BShrTy>();
  bool is_splat = bits.size() == 1;

  // arithmetic right shift expects to work on ring, or the behaviour is
  // undefined.
  SPU_ENFORCE(in_ty->nbits() == SizeOf(field) * 8, "in.type={}, field={}",
              in.eltype(), field);
  const PtType out_btype = in_ty->getBacktype();
  const size_t out_nbits = in_ty->nbits();

  return DISPATCH_UINT_PT_TYPES(in_ty->getBacktype(), [&]() {
    using el_t = std::make_signed_t<ScalarT>;
    using shr_t = std::array<el_t, 3>;

    NdArrayRef out(makeType<BShrTy>(out_btype, out_nbits), in.shape());
    NdArrayView<shr_t> _out(out);
    NdArrayView<shr_t> _in(in);

    pforeach(0, in.numel(), [&](int64_t idx) {
      const auto& v = _in[idx];
      auto shift_bit = is_splat ? bits[0] : bits[idx];
      _out[idx][0] = v[0] >> shift_bit;
      _out[idx][1] = v[1] >> shift_bit;
      _out[idx][2] = v[2] >> shift_bit;
    });

    return out;
  });
}

NdArrayRef BitrevB::proc(KernelEvalContext*, const NdArrayRef& in, size_t start,
                         size_t end) const {
  SPU_ENFORCE(start <= end && end <= 128);

  const auto* in_ty = in.eltype().as<BShrTy>();
  const size_t out_nbits = std::max(in_ty->nbits(), end);
  const PtType out_btype = calcBShareBacktype(out_nbits);

  return DISPATCH_UINT_PT_TYPES(in_ty->getBacktype(), [&]() {
    using in_el_t = ScalarT;
    using in_shr_t = std::array<in_el_t, 3>;

    NdArrayView<in_shr_t> _in(in);

    return DISPATCH_UINT_PT_TYPES(out_btype, [&]() {
      using out_el_t = ScalarT;
      using out_shr_t = std::array<out_el_t, 3>;

      NdArrayRef out(makeType<BShrTy>(out_btype, out_nbits), in.shape());
      NdArrayView<out_shr_t> _out(out);

      auto bitrev_fn = [&](out_el_t el) -> out_el_t {
        out_el_t tmp = 0U;
        for (size_t idx = start; idx < end; idx++) {
          if (el & ((out_el_t)1 << idx)) {
            tmp |= (out_el_t)1 << (end - 1 - idx + start);
          }
        }

        out_el_t mask = ((out_el_t)1U << end) - ((out_el_t)1U << start);
        return (el & ~mask) | tmp;
      };

      pforeach(0, in.numel(), [&](int64_t idx) {
        const auto& v = _in[idx];
        _out[idx][0] = bitrev_fn(static_cast<out_el_t>(v[0]));
        _out[idx][1] = bitrev_fn(static_cast<out_el_t>(v[1]));
        _out[idx][2] = bitrev_fn(static_cast<out_el_t>(v[2]));
      });

      return out;
    });
  });
}

NdArrayRef BitIntlB::proc(KernelEvalContext*, const NdArrayRef& in,
                          size_t stride) const {
  // void BitIntlB::evaluate(KernelEvalContext* ctx) const {
  const auto* in_ty = in.eltype().as<BShrTy>();
  const size_t nbits = in_ty->nbits();
  SPU_ENFORCE(absl::has_single_bit(nbits));

  NdArrayRef out(in.eltype(), in.shape());
  DISPATCH_UINT_PT_TYPES(in_ty->getBacktype(), [&]() {
    using el_t = ScalarT;
    using shr_t = std::array<el_t, 3>;
    NdArrayView<shr_t> _out(out);
    NdArrayView<shr_t> _in(in);

    pforeach(0, in.numel(), [&](int64_t idx) {
      const auto& v = _in[idx];
      _out[idx][0] = BitIntl<el_t>(v[0], stride, nbits);
      _out[idx][1] = BitIntl<el_t>(v[1], stride, nbits);
      _out[idx][2] = BitIntl<el_t>(v[2], stride, nbits);
    });
  });

  return out;
}

NdArrayRef BitDeintlB::proc(KernelEvalContext*, const NdArrayRef& in,
                            size_t stride) const {
  const auto* in_ty = in.eltype().as<BShrTy>();
  const size_t nbits = in_ty->nbits();
  SPU_ENFORCE(absl::has_single_bit(nbits));

  NdArrayRef out(in.eltype(), in.shape());
  DISPATCH_UINT_PT_TYPES(in_ty->getBacktype(), [&]() {
    using el_t = ScalarT;
    using shr_t = std::array<el_t, 3>;
    NdArrayView<shr_t> _out(out);
    NdArrayView<shr_t> _in(in);

    pforeach(0, in.numel(), [&](int64_t idx) {
      const auto& v = _in[idx];
      _out[idx][0] = BitDeintl<el_t>(v[0], stride, nbits);
      _out[idx][1] = BitDeintl<el_t>(v[1], stride, nbits);
      _out[idx][2] = BitDeintl<el_t>(v[2], stride, nbits);
    });
  });

  return out;
}

} // namespace spu::mpc::fantastic4