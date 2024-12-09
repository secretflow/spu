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

// ///////////////////////////////////////////////////
// Layout of Rep4:
// P1(x1,x2,x3) P2(x2,x3,x4) P3(x3,x4,x1) P4(x4,x1,x2)
// ///////////////////////////////////////////////////

namespace {
  // Sender and Receiver jointly input a X
  static NdArrayRef wrap_mul_aa(SPUContext* ctx, const NdArrayRef& x,
                              const NdArrayRef& y) {
    SPU_ENFORCE(x.shape() == y.shape());
    return UnwrapValue(mul_aa(ctx, WrapValue(x), WrapValue(y)));
  }

  size_t PrevRank(size_t rank, size_t world_size){
    return (rank + world_size -1) % world_size;
  }

  size_t OffsetRank(size_t myrank, size_t other, size_t world_size){
    size_t offset = (myrank + world_size -other) % world_size;
    if(offset == 3){
      offset = 1;
    }
    return offset;
  }

  // template <typename el_t>
  // NdArrayRef JointInputArith(KernelEvalContext* ctx, const std::vector<el_t>& input, FieldType field, Shape shape, size_t sender, size_t backup, size_t receiver, size_t outsider){
  //   auto* comm = ctx->getState<Communicator>();
  //   size_t world_size =  comm->getWorldSize();
  //   auto* prg_state = ctx->getState<PrgState>();
  //   auto myrank = comm->getRank();
    
  //   // SPU_ENFORCE_EQ(input.size(), output.numel());

    
  //   using shr_t = std::array<el_t, 3>;
  //   NdArrayRef output(makeType<AShrTy>(field), shape);
  //   NdArrayView<shr_t> _out(output);
  //   pforeach(0, output.numel(), [&](int64_t idx) {
  //         _out[idx][0] = 0;
  //         _out[idx][1] = 0;
  //         _out[idx][2] = 0;
  //   });
  //   pforeach(0, output.numel(), [&](int64_t idx) {
  //     if(myrank == 0){
  //       printf("My rank = %zu, init output shares:", myrank);
  //       for(int64_t i =0; i<3;i++){
          
  //         printf("output[%ld] = %llu  ", i, (unsigned long long)_out[idx][i]);
  //       }
  //       printf("\n");
  //     }
  //   });
    
  //   // Receiver's Previous Party Rank
  //   // The mask corresponds to the prev party of receiver, receiver doesn't have the correpsonding PRG of its prev party
  //   size_t receiver_prev_rank = PrevRank(receiver, world_size);

  //   // My offset from the receiver_prev_rank. 
  //   // 0- i'm the receiver_prev_rank
  //   // 1- i'm prev/next party of receiver_prev_rank
  //   // 2- next next
  //   size_t offset_from_receiver_prev = OffsetRank(myrank, receiver_prev_rank, world_size);
  //   // size_t offset_from_receiver = OffsetRank(myrank, receiver, world_size);
  //   size_t offset_from_outsider_prev = OffsetRank(myrank, (outsider + 4 - 1)%4 , world_size);

  //   // printf("My rank = %zu, sender_rank = %zu, receiver_rank = %zu, receiver_prev = %zu, offset_from_recv_prev = %zu, offset_from_outsider_prev = %zu \n", myrank, sender, receiver, receiver_prev_rank, offset_from_receiver_prev, offset_from_outsider_prev);
  //   if(myrank != receiver){
  //     // Non-Interactive Random Masks Generation.
  //     std::vector<el_t> r(output.numel());

  //     if(offset_from_receiver_prev == 0){
  //         // should use PRG[0]
  //         prg_state->fillPrssTuple<el_t>(r.data(), nullptr, nullptr , r.size(),
  //                             PrgState::GenPrssCtrl::First);
  //     }
  //     if(offset_from_receiver_prev == 1){
  //         // should use PRG[1]
  //         prg_state->fillPrssTuple<el_t>(nullptr, r.data(), nullptr , r.size(),
  //                             PrgState::GenPrssCtrl::Second);
  //     }
  //     if(offset_from_receiver_prev == 2){
  //         // should use PRG[2]
  //         prg_state->fillPrssTuple<el_t>(nullptr, nullptr, r.data(), r.size(),
  //                             PrgState::GenPrssCtrl::Third);
  //     }

  //     // For sender,backup,outsider
  //     // the corresponding share is set to r
  //     pforeach(0, output.numel(), [&](int64_t idx) {
  //         _out[idx][offset_from_receiver_prev] += r[idx];
  //         // printf("My rank = %zu, out[%zu] = %llu \n", myrank, offset_from_receiver_prev, (unsigned long long)_out[idx][offset_from_receiver_prev]);
  //         // printf("My rank = %zu, sender_rank = %zu, receiver_rank = %zu, receiver_prev = %zu, offset_from_recv_prev = %zu, offset_from_outsider_prev = %zu, x = %llu, r = %llu, x-r = %llu \n", myrank, sender, receiver, receiver_prev_rank, offset_from_receiver_prev, offset_from_outsider_prev);
    
  //     }); 
  //     pforeach(0, output.numel(), [&](int64_t idx) {
  //     if(myrank == 0){
  //       printf("My rank = %zu, after generate r and set r %llu:", myrank, (unsigned long long)r[idx]);
  //       for(int64_t i =0; i<3;i++){
          
  //         printf("output[%ld] = %llu  ", i, (unsigned long long)_out[idx][i]);
  //       }
  //       printf("\n");
  //     }
  //     });
  //     if(myrank != outsider){

  //       std::vector<el_t> input_minus_r(output.numel());

  //       // For sender, backup
  //       // compute and set masked input x-r
  //       pforeach(0, output.numel(), [&](int64_t idx) {
  //         input_minus_r[idx] = (input[idx] - r[idx]);
  //         _out[idx][offset_from_outsider_prev] +=  input_minus_r[idx];
  //         // printf("My rank = %zu, out[%zu] = %llu \n", myrank, offset_from_outsider_prev, (unsigned long long)_out[idx][offset_from_outsider_prev]);
    
  //         // printf("My rank = %zu, sender_rank = %zu, receiver_rank = %zu, receiver_prev = %zu, offset_from_recv_prev = %zu, offset_from_outsider_prev = %zu, x = %llu, r = %llu, x-r = %llu \n", myrank, sender, receiver, receiver_prev_rank, offset_from_receiver_prev, offset_from_outsider_prev, (unsigned long long)input[idx], (unsigned long long)r[idx], (unsigned long long)input_minus_r[idx]);
    
  //       }); 
  //       pforeach(0, output.numel(), [&](int64_t idx) {
  //         if(myrank == 0){
  //           printf("My rank = %zu, after compute x-r and set:", myrank);
  //           for(int64_t i =0; i<3;i++){
              
  //             printf("output[%ld] = %llu  ", i, (unsigned long long)_out[idx][i]);
  //           }
  //           printf("\n");
  //         }
  //       });
  //       // Sender send x-r to receiver
  //       if(myrank == sender) {
  //         comm->sendAsync<el_t>(receiver, input_minus_r, "Joint Input");
  //       }

  //       // Backup update x-r for sender-to-receiver channel
  //       if(myrank == backup) {
  //         // Todo:
  //         // MAC update input_minus_r
  //       }
  //     }
  //   }

  //   if (myrank == receiver) {
  //     auto input_minus_r = comm->recv<el_t>(sender, "Joint Input");
  //     pforeach(0, output.numel(), [&](int64_t idx) {
  //         _out[idx][offset_from_outsider_prev] += input_minus_r[idx];
  //     }); 

  //     // Todo: 
  //     // Mac update sender-backup channel
  //   }
  //   pforeach(0, output.numel(), [&](int64_t idx) {
  //     if(myrank == 0){
  //       printf("My rank = %zu, Current input[%ld], the shares:", myrank, idx+1);
  //       for(int64_t i =0; i<3;i++){
          
  //         printf("output[%ld] = %llu  ", i, (unsigned long long)_out[idx][i]);
  //       }
  //       printf("\n");
  //     }
  //   });
    
  //   return output;
  // }


  template <typename el_t>
  void JointInputArith(KernelEvalContext* ctx, std::vector<el_t>& input, NdArrayRef& output, size_t sender, size_t backup, size_t receiver, size_t outsider){
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
    size_t receiver_prev_rank = PrevRank(receiver, world_size);

    // My offset from the receiver_prev_rank. 
    // 0- i'm the receiver_prev_rank
    // 1- i'm prev/next party of receiver_prev_rank
    // 2- next next
    size_t offset_from_receiver_prev = OffsetRank(myrank, receiver_prev_rank, world_size);
    // size_t offset_from_receiver = OffsetRank(myrank, receiver, world_size);
    size_t offset_from_outsider_prev = OffsetRank(myrank, (outsider + 4 - 1)%4 , world_size);

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
          _out[idx][offset_from_receiver_prev] += r[idx];
      }); 

      if(myrank != outsider){

        std::vector<el_t> input_minus_r(output.numel());

        // For sender, backup
        // compute and set masked input x-r
        pforeach(0, output.numel(), [&](int64_t idx) {
          input_minus_r[idx] = (input[idx] - r[idx]);
          _out[idx][offset_from_outsider_prev] +=  input_minus_r[idx];
          
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
          _out[idx][offset_from_outsider_prev] += input_minus_r[idx];
      }); 

      // Todo: 
      // Mac update sender-backup channel
    }

    // pforeach(0, output.numel(), [&](int64_t idx) {
      
    //     printf("My rank = %zu, Current input[%ld], the shares:", myrank, idx+1);
    //     for(int64_t i =0; i<3;i++){
          
    //       printf("output[%ld] = %llu  ", i, (unsigned long long)_out[idx][i]);
    //     }
    //     printf("\n");
      
    // });

  }


  template <typename el_t>
  void JointInputArith(KernelEvalContext* ctx, const std::vector<el_t>& input, NdArrayRef& output, size_t sender, size_t backup, size_t receiver, size_t outsider){
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
    size_t receiver_prev_rank = PrevRank(receiver, world_size);

    // My offset from the receiver_prev_rank. 
    // 0- i'm the receiver_prev_rank
    // 1- i'm prev/next party of receiver_prev_rank
    // 2- next next
    size_t offset_from_receiver_prev = OffsetRank(myrank, receiver_prev_rank, world_size);
    // size_t offset_from_receiver = OffsetRank(myrank, receiver, world_size);
    size_t offset_from_outsider_prev = OffsetRank(myrank, (outsider + 4 - 1)%4 , world_size);

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
          _out[idx][offset_from_receiver_prev] += r[idx];
      }); 

      if(myrank != outsider){

        std::vector<el_t> input_minus_r(output.numel());

        // For sender, backup
        // compute and set masked input x-r
        pforeach(0, output.numel(), [&](int64_t idx) {
          input_minus_r[idx] = (input[idx] - r[idx]);
          _out[idx][offset_from_outsider_prev] +=  input_minus_r[idx];
          
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
          _out[idx][offset_from_outsider_prev] += input_minus_r[idx];
      }); 

      // Todo: 
      // Mac update sender-backup channel
    }

    // pforeach(0, output.numel(), [&](int64_t idx) {
      
    //     printf("My rank = %zu, Current input[%ld], the shares:", myrank, idx+1);
    //     for(int64_t i =0; i<3;i++){
          
    //       printf("output[%ld] = %llu  ", i, (unsigned long long)_out[idx][i]);
    //     }
    //     printf("\n");
      
    // });

  }

}


// Pass the third share to previous party
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

    auto x4 = comm->rotate<ashr_el_t>(x3, "a2p");  // comm => 1, k

    pforeach(0, numel, [&](int64_t idx) {
      _out[idx] = _in[idx][0] + _in[idx][1] + _in[idx][2] + x4[idx];
      //std::cout << "Party" << (comm->getRank() + 1) << ": x = " << _out[idx] << " x1 = " << _in[idx][0] << " x2 = " << _in[idx][1] << " x3 = " << _in[idx][2] << " x4 = " << x4[idx] << std::endl;
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
     // TODO: debug masks?

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

    // pforeach(0, lhs.numel(), [&](int64_t idx) {
    //     printf("My rank = %zu, Current input[%ld], the shares:", rank, idx+1);
    //     for(int64_t i =0; i<5;i++){
    //       printf("a[%ld] = %llu  ", i, (unsigned long long)a[i][idx]);
    //     }
    //     printf("\n");
    // });

    

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

    if(rank == 0){
      printf("My rank = %zu, Init output:", rank);
      pforeach(0, x.shape()[0], [&](int64_t row) {
        for(int64_t col = 0; col < x.shape()[1] ; col++ ){
          printf("x[%ld][%ld] = (%llu, %llu, %llu)", row, col, (unsigned long long)_x[row * N + col][0], (unsigned long long)_x[row * N + col][1], (unsigned long long)_x[row * N + col][2]);
          }
      });
      pforeach(0, y.shape()[0], [&](int64_t row) {
        for(int64_t col = 0; col < y.shape()[1] ; col++ ){
          printf("y[%ld][%ld] = (%llu, %llu, %llu)", row, col, (unsigned long long)_y[row * N + col][0], (unsigned long long)_y[row * N + col][1], (unsigned long long)_y[row * N + col][2]);
          }
      });
    }
    pforeach(0, M, [&](int64_t row) {
      for(int64_t col = 0; col < N ; col++ ){
        _out[row * N + col][0] = 0;
        _out[row * N + col][1] = 0;
        _out[row * N + col][2] = 0;
        // printf("out[%ld][%ld] = (%llu, %llu, %llu)", row, col, (unsigned long long)_out[row * N + col][0], (unsigned long long)_out[row * N + col][1], (unsigned long long)_out[row * N + col][2]);
        // printf("a[][%ld][%ld] = (%llu, %llu, %llu)", row, col, (unsigned long long)_out[row][col][0], _out[row][col][1], _out[row][col][2] = 0;);
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
        
        printf("My rank = %zu , numel = %lu:", rank, out.numel());
        
        pforeach(0, out.numel(), [&](int64_t idx) {
          // r = r_{k-1}......r_{0}
          r[idx] = r1[idx] + r2[idx];
          // rb = r >> k-1
          rb[idx] = r[idx] >> (k-1);
          // rc = r_{k-2}.....r_{m}
          rc[idx] = (r[idx] << 1) >> (bits + 1);

          printf("in[%ld] = (%llu, %llu, %llu), binary: \n", idx, (unsigned long long)_in[idx][0], (unsigned long long)_in[idx][1], (unsigned long long)_in[idx][2]); 
          printBinary((unsigned long long)_in[idx][0], k);
          printf("\n");
          printf("r = ");
          printBinary((unsigned long long)r[idx], k);
          // printf("\n rb = ");
          // printBinary((unsigned long long)rb[idx], k);

          printf("\n r+x = %llu = ", (unsigned long long)(_in[idx][0] + r[idx]));
          printBinary((unsigned long long)((_in[idx][0] + r[idx])), k);

          // printf("\n rc = ");
          // printBinary((unsigned long long)rc[idx], k);
          // printf("r[%ld] = %llu, MSB = %llu, rc = %llu)", idx, (unsigned long long)r[idx], (unsigned long long)rb[idx], (unsigned long long)rc[idx]); 
        });
        // -------------------------------------
        // Step 2: Generate the share of rb, rc
        // -------------------------------------
        JointInputArith(ctx, rb, rb_shr, 0, 1, 3, 2);
        JointInputArith(ctx, rc, rc_shr, 0, 1, 3, 2);

        // pforeach(0, out.numel(), [&](int64_t idx) {
        //   printf("MSB = %llu, share = (%llu, %llu, %llu))", (unsigned long long)rb[idx], (unsigned long long)_rb_shr[idx][0], (unsigned long long)_rb_shr[idx][1], (unsigned long long)_rb_shr[idx][2]); 
        // });


        // -------------------------------------
        // Step 3: compute [x] + [r]
        //          [r] = r0 + r1 + r2 + r3, only r1 and r2 are non-zero
        // -------------------------------------
        
        pforeach(0, out.numel(), [&](int64_t idx) {
          _masked_input[idx][0] = _in[idx][0]; // r0 = 0
          _masked_input[idx][1] = _in[idx][1] + r1[idx];
          _masked_input[idx][2] = _in[idx][2] + r2[idx];
          printf("masked_input[%ld] = (%llu, %llu, %llu) \n", idx, (unsigned long long)_masked_input[idx][0], (unsigned long long)_masked_input[idx][1], (unsigned long long)_masked_input[idx][2]); 
          printf("rc_shr[%ld] = (%llu, %llu, %llu) \n", idx, (unsigned long long)_rc_shr[idx][0], (unsigned long long)_rc_shr[idx][1], (unsigned long long)_rc_shr[idx][2]); 
          
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
          printf("overflow[%ld] = (%llu, %llu, %llu) \n", idx, (unsigned long long)_overflow[idx][0], (unsigned long long)_overflow[idx][1], (unsigned long long)_overflow[idx][2]); 
          
          _out[idx][0] = _sc_shr[idx][0] - _rc_shr[idx][0] + (_overflow[idx][0] << (k - bits - 1));
          _out[idx][1] = _sc_shr[idx][1] - _rc_shr[idx][1] + (_overflow[idx][1] << (k - bits - 1));
          _out[idx][2] = _sc_shr[idx][2] - _rc_shr[idx][2] + (_overflow[idx][2] << (k - bits - 1));

          printf("out[%ld] = (%llu, %llu, %llu) \n", idx, (unsigned long long)_out[idx][0], (unsigned long long)_out[idx][1], (unsigned long long)_out[idx][2]); 
          
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
        
        printf("My rank = %zu, Init output:", rank);
        
        pforeach(0, out.numel(), [&](int64_t idx) {
          // r = r_{k-1}......r_{0}
          r[idx] = r1[idx] + r2[idx];
          // rb = r >> k-1
          rb[idx] = r[idx] >> (k-1);
          // rc = r_{k-2}.....r_{m}
          rc[idx] = (r[idx] << 1) >> (bits + 1);
          
          // printf("r = ");
          // printBinary((unsigned long long)r[idx], k);
          // printf("\n rb = ");
          // printBinary((unsigned long long)rb[idx], k);
          // printf("\n rc = ");
          // printBinary((unsigned long long)rc[idx], k);
          printf("r[%ld] = %llu, MSB = %llu, rc = %llu) \n", idx, (unsigned long long)r[idx], (unsigned long long)rb[idx], (unsigned long long)rc[idx]); 
        });

        // -------------------------------------
        // Step 2: Generate the share of rb, rc
        // -------------------------------------
        JointInputArith(ctx, rb, rb_shr, 0, 1, 3, 2);
        JointInputArith(ctx, rc, rc_shr, 0, 1, 3, 2);
        // pforeach(0, out.numel(), [&](int64_t idx) {
        //   printf("MSB = %llu, share = (%llu, %llu, %llu))", (unsigned long long)rb[idx], (unsigned long long)_rb_shr[idx][0], (unsigned long long)_rb_shr[idx][1], (unsigned long long)_rb_shr[idx][2]); 
        // });

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
          printf("masked_input[%ld] = (%llu, %llu, %llu) \n", idx, (unsigned long long)_masked_input[idx][0], (unsigned long long)_masked_input[idx][1], (unsigned long long)_masked_input[idx][2]); 
          printf("rc_shr[%ld] = (%llu, %llu, %llu) \n", idx, (unsigned long long)_rc_shr[idx][0], (unsigned long long)_rc_shr[idx][1], (unsigned long long)_rc_shr[idx][2]); 
          
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
          printf("overflow[%ld] = (%llu, %llu, %llu) \n", idx, (unsigned long long)_overflow[idx][0], (unsigned long long)_overflow[idx][1], (unsigned long long)_overflow[idx][2]); 
          
          _out[idx][0] = _sc_shr[idx][0] - _rc_shr[idx][0] + (_overflow[idx][0] << (k - bits - 1));
          _out[idx][1] = _sc_shr[idx][1] - _rc_shr[idx][1] + (_overflow[idx][1] << (k - bits - 1));
          _out[idx][2] = _sc_shr[idx][2] - _rc_shr[idx][2] + (_overflow[idx][2] << (k - bits - 1));
          printf("out[%ld] = (%llu, %llu, %llu) \n", idx, (unsigned long long)_out[idx][0], (unsigned long long)_out[idx][1], (unsigned long long)_out[idx][2]); 
          
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

        // printf("My rank = %zu, Init output:", rank);
        // pforeach(0, out.numel(), [&](int64_t idx) {

        //   printf("MSB = %llu, share = (%llu, %llu, %llu))", (unsigned long long)rb[idx], (unsigned long long)_rb_shr[idx][0], (unsigned long long)_rb_shr[idx][1], (unsigned long long)_rb_shr[idx][2]); 
        // });       

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

        // printf("My rank = %zu, Init output:", rank);
        // pforeach(0, out.numel(), [&](int64_t idx) {

        //   printf("MSB = %llu, share = (%llu, %llu, %llu))", (unsigned long long)rb[idx], (unsigned long long)_rb_shr[idx][0], (unsigned long long)_rb_shr[idx][1], (unsigned long long)_rb_shr[idx][2]); 
        // });

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
                  


  // auto r_future = std::async([&] {
  //   return prg_state->genPrssPair(field, in.shape(),
  //                                 PrgState::GenPrssCtrl::Both);
  // });

  // // in
  // const auto& x1 = getFirstShare(in);
  // const auto& x2 = getSecondShare(in);

  // const auto kComm = x1.elsize() * x1.numel();

  // // we only record the maximum communication, we need to manually add comm
  // comm->addCommStatsManually(1, kComm);  // comm => 1, 2

  // // ret
  // const Sizes shift_bit = {static_cast<int64_t>(bits)};
  // switch (comm->getRank()) {
  //   case 0: {
  //     const auto z1 = ring_arshift(x1, shift_bit);
  //     const auto z2 = comm->recv(1, x1.eltype(), kBindName());
  //     return makeAShare(z1, z2, field);
  //   }

  //   case 1: {
  //     auto r1 = r_future.get().second;
  //     const auto z1 = ring_sub(ring_arshift(ring_add(x1, x2), shift_bit), r1);
  //     comm->sendAsync(0, z1, kBindName());
  //     return makeAShare(z1, r1, field);
  //   }

  //   case 2: {
  //     const auto z2 = ring_arshift(x2, shift_bit);
  //     return makeAShare(r_future.get().first, z2, field);
  //   }

  //   default:
  //     SPU_THROW("Party number exceeds 3!");
  // }
}



// NdArrayRef MulAA::proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
//                        const NdArrayRef& rhs) const {
//   const auto field = lhs.eltype().as<Ring2k>()->field();
//   auto* comm = ctx->getState<Communicator>();
//   auto* prg_state = ctx->getState<PrgState>();
//   auto rank = comm->getRank();
//   return DISPATCH_ALL_FIELDS(field, [&]() {
//     using el_t = ring2k_t;
//     using shr_t = std::array<el_t, 3>;
//     NdArrayView<shr_t> _lhs(lhs);
//     NdArrayView<shr_t> _rhs(rhs);
//     NdArrayRef out(makeType<AShrTy>(field), lhs.shape());
//     NdArrayView<shr_t> _out(out);
//     // Me and prev have a0
//     // Me and next have a1
//     std::array<std::vector<el_t>, 5> a;
//     for (auto& vec : a) {
//         vec = std::vector<el_t>(lhs.numel());
//     }
//     std::vector<el_t>& a0 = a[0];
//     std::vector<el_t>& a1 = a[1];
//     std::vector<el_t>& a2 = a[2];
//     // std::vector<el_t>& a3 = a[3];
//     std::vector<el_t>& a4 = a[4];
//     // Me and next_next have cross term b
//     std::vector<el_t> b(lhs.numel());
//     std::vector<el_t> r0(lhs.numel());
//     std::vector<el_t> r1(lhs.numel());
//     std::vector<el_t> r2(lhs.numel());
//     prg_state->fillPrssTuple<el_t>(r0.data(), r1.data(), r2.data(), r2.size(),
//                             PrgState::GenPrssCtrl::All);
//     // z1 = (x1 * y1) + (x1 * y2) + (x2 * y1) + (r0 - r1);
//     pforeach(0, lhs.numel(), [&](int64_t idx) {
//         a0[idx] = (_lhs[idx][0] + _lhs[idx][1]) * _rhs[idx][0] + _lhs[idx][0] * _rhs[idx][1] - r1[idx];  // xi*yi + xi*yj + xj*yi
//         a1[idx] = (_lhs[idx][1] + _lhs[idx][2]) * _rhs[idx][1] + _lhs[idx][1] * _rhs[idx][2] - r2[idx];  // xj*yj + xj*yg + xg*yj
//         a4[idx] = _lhs[idx][0] * _rhs[idx][2] + _lhs[idx][2] * _rhs[idx][0];                    // xi*yg + xg*yi
//     });
//     a2 = comm->rotate<el_t>(a1, "mulaa");  // comm => 1, k
//     if (rank == 0) {
//       // rb = PRG[2], c = PRG[1]
//       std::vector<el_t> rb(lhs.numel());
//       std::vector<el_t> rc(lhs.numel());
//       prg_state->fillPrssTuple<el_t>(nullptr, nullptr, rb.data(), rb.size(),
//                             PrgState::GenPrssCtrl::Third);
//       prg_state->fillPrssTuple<el_t>(nullptr, rc.data(), nullptr, rc.size(),
//                             PrgState::GenPrssCtrl::Second);
//       pforeach(0, lhs.numel(), [&](int64_t idx) {
//         a4[idx] = a4[idx] - rb[idx]; // b = b - r'2
//         _out[idx][0] = a0[idx] + r0[idx] + a4[idx]; 
//         _out[idx][1] = a1[idx] + r1[idx] + rc[idx];
//         _out[idx][2] = a2[idx] + r2[idx] + rb[idx];
//       });        
//       comm->sendAsync<el_t>(3, a4, "mulaa 03");
//     }
//     else if (rank == 1) {
//       // rb = PRG[0], rc = PRG[1]
//       std::vector<el_t> rb(lhs.numel());
//       std::vector<el_t> rc(lhs.numel());
//       prg_state->fillPrssTuple<el_t>(rb.data(), nullptr, nullptr , rb.size(),
//                             PrgState::GenPrssCtrl::First);
//       prg_state->fillPrssTuple<el_t>(nullptr, rc.data(), nullptr, rc.size(),
//                             PrgState::GenPrssCtrl::Second);
//       pforeach(0, lhs.numel(), [&](int64_t idx) {
//         a4[idx] = a4[idx] - rb[idx];
//         _out[idx][0] = a0[idx] + r0[idx] + rb[idx]; 
//         _out[idx][1] = a1[idx] + r1[idx] + rc[idx];
//         _out[idx][2] = a2[idx] + r2[idx] + a4[idx];
//       });
//       comm->sendAsync<el_t>(2, a4, "mulaa 12");  // comm => 1, k  
//     }
//     else if (rank == 2) {
//       // rb = PRG[0]
//       std::vector<el_t> rb(lhs.numel());
//       prg_state->fillPrssTuple<el_t>(rb.data(), nullptr, nullptr , rb.size(),
//                             PrgState::GenPrssCtrl::First);
//       auto c = comm->recv<el_t>(1, "mulaa 12"); 
//       pforeach(0, lhs.numel(), [&](int64_t idx) {
//         a4[idx] = a4[idx] - rb[idx]; 
//         _out[idx][0] = a0[idx] + r0[idx] + rb[idx]; 
//         _out[idx][1] = a1[idx] + r1[idx] + c[idx];
//         _out[idx][2] = a2[idx] + r2[idx] + a4[idx];
//       });  
//     }
//     else if (rank == 3) {
//       // rb = PRG[2]
//       std::vector<el_t> rb(lhs.numel());
//       prg_state->fillPrssTuple<el_t>(nullptr, nullptr, rb.data(), rb.size(),
//                             PrgState::GenPrssCtrl::Third);
//       auto c = comm->recv<el_t>(0, "mulaa 03");
//       pforeach(0, lhs.numel(), [&](int64_t idx) {
//         a4[idx] = a4[idx] - rb[idx];  
//         _out[idx][0] = a0[idx] + r0[idx] + a4[idx]; 
//         _out[idx][1] = a1[idx] + r1[idx] + c[idx];
//         _out[idx][2] = a2[idx] + r2[idx] + rb[idx];
//       });  
//     }
//     return out;
//   });
// }







} // namespace spu::mpc::fantastic4





