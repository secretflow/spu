// Copyright 2025 Ant Group Co., Ltd.
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

#pragma once

// Use optimized F4 protocol
// #define OPTIMIZED_F4

#include "libspu/core/context.h"

#include "libspu/mpc/kernel.h"

#include "libspu/mpc/fantastic4/value.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/fantastic4/state.h"

namespace spu::mpc::fantastic4 {
    size_t PrevRank(size_t rank, size_t world_size);
    size_t OffsetRank(size_t myrank, size_t other, size_t world_size);

    // Protocol 2 Joint message passing in Section 2.2
    template <typename el_t>
    void JointMsgPass(KernelEvalContext* ctx, std::vector<el_t>& msg, size_t sender, size_t backup, size_t receiver){
      auto* comm = ctx->getState<Communicator>();
      auto myrank = comm->getRank();

      auto* mac_state = ctx->getState<Fantastic4MacState>();

      // Sender send x-r to receiver
      if(myrank == sender) {
        comm->sendAsync<el_t>(receiver, msg, "jmp" + std::to_string(sender) + std::to_string(backup) + std::to_string(receiver));
      }
      // Backup update x-r for sender-to-receiver channel
      else if(myrank == backup) {
        mac_state->update_msg<el_t>(sender, backup, receiver, msg);
      }
      else if(myrank == receiver) {
        msg = comm->recv<el_t>(sender, "jmp" + std::to_string(sender) + std::to_string(backup) + std::to_string(receiver));
        mac_state->update_msg<el_t>(sender, backup, receiver, msg);
      }
    }

    // Joint message rotate in single round, each party sends a msg to its previous party while serving as backup for next party's msg
    // The reason we do not use 4 sequential invocations of JointMsgPass is,
    //  here, we can let parties send "async" msgs first and then recv in single round
    template <typename el_t>
    std::vector<el_t> JointMsgRotate(KernelEvalContext* ctx, std::vector<el_t>& msg_to_send, std::vector<el_t>& msg_to_backup){
      auto* comm = ctx->getState<Communicator>();
      auto myrank = comm->getRank();
      auto* mac_state = ctx->getState<Fantastic4MacState>();

      // As sender, send msg_to_send to the previous party
      comm->sendAsync<el_t>(PrevRank(myrank, 4), msg_to_send, "rotate");
      // As backup, record the previous party's msg
      mac_state->update_msg<el_t>(PrevRank(myrank, 4), myrank, PrevRank(PrevRank(myrank, 4), 4), msg_to_backup);
      // As receiver, recv msg from the next party
      auto msg = comm->recv<el_t>((myrank + 1) % 4, "rotate");
      mac_state->update_msg<el_t>((myrank + 1) % 4, (myrank + 2) % 4, myrank, msg);
      return msg;
    }

    // Protocol 3 Shared Input in Section 2.3
    // Joint Input by two parties and share the secret in arithmetic sharing
    // Here we do not implement Joint Message Passing as an interface, but directly let the parties do what they should do
    //   - input: the secret common input of sender and backup
    //   - output: the output shares
    //   - sender: has secret input and send the masked input to receiver, adds the mask to corresponding output share
    //   - backup: has secret input and record the hash(masked input), adds the mask to corresponding output share
    //   - receiver: receives masked input from sender and record the hash, adds the masked input to corresponding output share
    //   - outsider: adds the mask to corresponding output share

    // Note: since we accumulate shares of input on the NdArrayRef output instead of assignment,
    //    ensure NdArrayRef elements are initiated as 0 (refer to the out_buf in MulAA)

    // Note: if there are crossing communications in a single round, e.g. MulAA in arithmetic.cc
    //    use JointInputArith could results in multiple rounds since a party could first wait for receive in previous call and then send its msg
    //    we should use JointInputArithSend / JointInputArithRecv function below instead
    template <typename el_t>
    void JointInputArith(KernelEvalContext* ctx, std::vector<el_t>& input, NdArrayRef& output, size_t sender, size_t backup, size_t receiver, size_t outsider){
      auto* comm = ctx->getState<Communicator>();
      size_t world_size =  comm->getWorldSize();
      auto* prg_state = ctx->getState<PrgState>();
      auto myrank = comm->getRank();

      using shr_t = std::array<el_t, 3>;
      NdArrayView<shr_t> _out(output);

      // The sender, backup and outsider will use their common PRG to generate the mask unknown to the receiver
      // Since in our sharing scheme,
      //    we let Party i (i in {0, 1, 2, 3}) holds x_i, x_i+1, x_i+2 (mod 4)
      //    for PRGs, we let Party i holds k_i--self, k_i+1 --next, k_i+2--next next (mod 4), see prg_state.h
      //    any party doesn't have the correpsonding PRG/share of its prev party
      //
      // Thus, this PRG corresponds to the PRG of the previous party of receiver,
      //    that is, global k_{receiver-1 mod 4}, which is held by sender, backup and outsider
      //
      // Specifically, in the own context of sender, backup or outsider,
      //    each party has three local PRGs in its view
      //    this global k_{receiver-1 mod 4} can be loacated by the rank offset from the previous party of receiver k_{receiver-1 mod 4}.
      //    should use own::PRG[offset_from_receiver_prev] to generate mask r (see fillPrssTuple: first, second, third)

      // Similarly, the sender, backup and outsider should set the share unknown to the receiver as the mask r
      //    this also corresponds the x_{receiver-1 mod 4} and can be loacated by offset_from_receiver_prev
      //
      // Also, the sender, backup and receiver will have x-r in common which should be add to the share unknown to the outsider
      //    in the own context of them, this share can be located by the rank offset from the previous party of outsider

      // For example, (P0, P2) jointly input x, and choose P0 as sender, P1 as the receiver, P2 as backup, P3 as outsider
      //    P1 has k1, k2, k3 and doesn't know k0, while P0, P2, P3 have k0 in common
      //    P0 is the previous party of receiver, thus P0, P2, P3 should use PRG(k0) to generate r, and add r to the global output share x0
      //    Let's see the view of each party and analyse the index location:
      //        P0::k = (k0, k1, k2). since P0's offset from P0 is 0, it uses k[0] = k0 -> correct (also holds for locating x0)
      //        P2::k = (k2, k3, k0). since P2's offset from P0 is 2, it uses k[2] = k0 -> correct (also holds for locating x0)
      //        P3::k = (k3, k0, k1). since P3's offset from P0 is 1, it uses k[1] = k0 -> correct (also holds for locating x0)
      //
      //    After the communication, P0, P1, P2 have masked input x-r in common that outsider P3 doesn't have
      //    x-r should be add to the global output share x2 corresponding outsider's previous party P2
      //    Let's see the view of each party and analyse the index location:
      //        P0::x = (x0, x1, x2). since P0's offset from P2 is 2, it locates x[2] = x2 -> correct
      //        P1::x = (x1, x2, x3). since P1's offset from P2 is 1, it locates x[1] = x2 -> correct
      //        P2::x = (k2, k3, k0). since P2's offset from P2 is 0, it locates x[0] = x2 -> correct

      size_t offset_from_receiver_prev = OffsetRank(myrank, PrevRank(receiver, world_size), world_size);
      size_t offset_from_outsider_prev = OffsetRank(myrank, PrevRank(outsider, world_size), world_size);

      if(myrank != receiver){
        // Non-Interactive Random Masks Generation.
        std::vector<el_t> r(output.numel());
        if(offset_from_receiver_prev == 0){
            // should use PRG[0]
            prg_state->fillPrssTuple<el_t>(r.data(), nullptr, nullptr , r.size(), PrgState::GenPrssCtrl::First);
        }
        else if(offset_from_receiver_prev == 1){
            // should use PRG[1]
            prg_state->fillPrssTuple<el_t>(nullptr, r.data(), nullptr , r.size(), PrgState::GenPrssCtrl::Second);
        }
        else{
            // should use PRG[2]
            prg_state->fillPrssTuple<el_t>(nullptr, nullptr, r.data(), r.size(), PrgState::GenPrssCtrl::Third);
        }

        // For sender,backup,outsider
        // the corresponding share is set to r
        // we accumulate this share on the output NdArray
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
          // Sender send x-r to receiver, Backup record MAC
          JointMsgPass(ctx, input_minus_r, sender, backup, receiver);
        }
      }
      if (myrank == receiver) {
        std::vector<el_t> input_minus_r(output.numel());
        JointMsgPass(ctx, input_minus_r, sender, backup, receiver);
        pforeach(0, output.numel(), [&](int64_t idx) {
            _out[idx][offset_from_outsider_prev] += input_minus_r[idx];
        });
      }
    }

    // The send and backup phase of Joint Input
    // if there are crossing communications in a single round, e.g. MulAA in arithmetic.cc
    // split phases allows parties send "async" msgs first and then waits in the receive phase
    template <typename el_t>
    void JointInputArithSend(KernelEvalContext* ctx, std::vector<el_t>& input, NdArrayRef& output, size_t sender, size_t backup, size_t receiver, size_t outsider){
      auto* comm = ctx->getState<Communicator>();
      size_t world_size =  comm->getWorldSize();
      auto* prg_state = ctx->getState<PrgState>();
      auto myrank = comm->getRank();

      using shr_t = std::array<el_t, 3>;
      NdArrayView<shr_t> _out(output);

      size_t offset_from_receiver_prev = OffsetRank(myrank, PrevRank(receiver, world_size), world_size);
      size_t offset_from_outsider_prev = OffsetRank(myrank, PrevRank(outsider, world_size), world_size);

      if(myrank != receiver){
        // Non-Interactive Random Masks Generation.
        std::vector<el_t> r(output.numel());
        if(offset_from_receiver_prev == 0){
            // should use PRG[0]
            prg_state->fillPrssTuple<el_t>(r.data(), nullptr, nullptr , r.size(), PrgState::GenPrssCtrl::First);
        }
        else if(offset_from_receiver_prev == 1){
            // should use PRG[1]
            prg_state->fillPrssTuple<el_t>(nullptr, r.data(), nullptr , r.size(), PrgState::GenPrssCtrl::Second);
        }
        else{
            // should use PRG[2]
            prg_state->fillPrssTuple<el_t>(nullptr, nullptr, r.data(), r.size(), PrgState::GenPrssCtrl::Third);
        }

        // For sender,backup,outsider
        // the corresponding share is set to r
        // we accumulate this share on the output NdArray
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
          // Sender send x-r to receiver, Backup record MAC
          JointMsgPass(ctx, input_minus_r, sender, backup, receiver);
        }
      }
    }

    // The receive phase of Joint Input
    // if there are crossing communications in a single round, e.g. MulAA in arithmetic.cc
    // split phases allows parties send "async" msgs first and then waits in the receive phase
    template <typename el_t>
    void JointInputArithRecv(KernelEvalContext* ctx, std::vector<el_t>& input, NdArrayRef& output, size_t sender, size_t backup, size_t receiver, size_t outsider){
      auto* comm = ctx->getState<Communicator>();
      size_t world_size =  comm->getWorldSize();
      // auto* prg_state = ctx->getState<PrgState>();
      auto myrank = comm->getRank();

      using shr_t = std::array<el_t, 3>;
      NdArrayView<shr_t> _out(output);

      size_t offset_from_outsider_prev = OffsetRank(myrank, PrevRank(outsider, world_size), world_size);

      if (myrank == receiver) {
        std::vector<el_t> input_minus_r(output.numel());
        JointMsgPass(ctx, input_minus_r, sender, backup, receiver);
        pforeach(0, output.numel(), [&](int64_t idx) {
            _out[idx][offset_from_outsider_prev] += input_minus_r[idx];
        });
      }
    }

    // Joint Input by two parties, and share the secret in Boolean sharing
    // Here use XOR instead of addition
    template <typename el_t>
    void JointInputBool(KernelEvalContext* ctx, std::vector<el_t>& input, NdArrayRef& output, size_t sender, size_t backup, size_t receiver, size_t outsider){
      auto* comm = ctx->getState<Communicator>();
      size_t world_size =  comm->getWorldSize();
      auto* prg_state = ctx->getState<PrgState>();
      auto myrank = comm->getRank();

      using shr_t = std::array<el_t, 3>;
      NdArrayView<shr_t> _out(output);

      // The mask corresponds to the prev party of receiver, receiver doesn't have the correpsonding PRG of its prev party
      size_t offset_from_receiver_prev = OffsetRank(myrank, PrevRank(receiver, world_size), world_size);
      size_t offset_from_outsider_prev = OffsetRank(myrank, PrevRank(outsider, world_size), world_size);

      if(myrank != receiver){
        // Non-Interactive Random Masks Generation.
        std::vector<el_t> r(output.numel());

        if(offset_from_receiver_prev == 0){
            // should use PRG[0]
            prg_state->fillPrssTuple<el_t>(r.data(), nullptr, nullptr , r.size(), PrgState::GenPrssCtrl::First);
        }
        else if(offset_from_receiver_prev == 1){
            // should use PRG[1]
            prg_state->fillPrssTuple<el_t>(nullptr, r.data(), nullptr , r.size(), PrgState::GenPrssCtrl::Second);
        }
        else{
            // should use PRG[2]
            prg_state->fillPrssTuple<el_t>(nullptr, nullptr, r.data(), r.size(), PrgState::GenPrssCtrl::Third);
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

          JointMsgPass(ctx, input_minus_r, sender, backup, receiver);
        }
      }

      if (myrank == receiver) {
        std::vector<el_t> input_minus_r(output.numel());
        JointMsgPass(ctx, input_minus_r, sender, backup, receiver);
        pforeach(0, output.numel(), [&](int64_t idx) {
            _out[idx][offset_from_outsider_prev] ^= input_minus_r[idx];
        });
      }
    }

    // The send and backup phase of Joint Input
    // if there are crossing communications in a single round, e.g. AndBB in boolean.cc
    // split phases allows parties send "async" msgs first and then waits in the receive phase
    template <typename el_t>
    void JointInputBoolSend(KernelEvalContext* ctx, std::vector<el_t>& input, NdArrayRef& output, size_t sender, size_t backup, size_t receiver, size_t outsider){
      auto* comm = ctx->getState<Communicator>();
      size_t world_size =  comm->getWorldSize();
      auto* prg_state = ctx->getState<PrgState>();
      auto myrank = comm->getRank();

      using shr_t = std::array<el_t, 3>;
      NdArrayView<shr_t> _out(output);


      // The mask corresponds to the prev party of receiver, receiver doesn't have the correpsonding PRG of its prev party
      size_t offset_from_receiver_prev = OffsetRank(myrank, PrevRank(receiver, world_size), world_size);
      size_t offset_from_outsider_prev = OffsetRank(myrank, PrevRank(outsider, world_size), world_size);

      if(myrank != receiver){
        // Non-Interactive Random Masks Generation.
        std::vector<el_t> r(output.numel());

        if(offset_from_receiver_prev == 0){
            // should use PRG[0]
            prg_state->fillPrssTuple<el_t>(r.data(), nullptr, nullptr , r.size(), PrgState::GenPrssCtrl::First);
        }
        else if(offset_from_receiver_prev == 1){
            // should use PRG[1]
            prg_state->fillPrssTuple<el_t>(nullptr, r.data(), nullptr , r.size(), PrgState::GenPrssCtrl::Second);
        }
        else{
            // should use PRG[2]
            prg_state->fillPrssTuple<el_t>(nullptr, nullptr, r.data(), r.size(), PrgState::GenPrssCtrl::Third);
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

          JointMsgPass(ctx, input_minus_r, sender, backup, receiver);
        }
      }
    }

    // The receive phase of Joint Input
    // if there are crossing communications in a single round, e.g. AndBB in boolean.cc
    // split phases allows parties send "async" msgs first and then waits in the receive phase
    template <typename el_t>
    void JointInputBoolRecv(KernelEvalContext* ctx, std::vector<el_t>& input, NdArrayRef& output, size_t sender, size_t backup, size_t receiver, size_t outsider){
      auto* comm = ctx->getState<Communicator>();
      size_t world_size =  comm->getWorldSize();
      auto myrank = comm->getRank();

      using shr_t = std::array<el_t, 3>;
      NdArrayView<shr_t> _out(output);

      // The mask corresponds to the prev party of receiver, receiver doesn't have the correpsonding PRG of its prev party
      size_t offset_from_outsider_prev = OffsetRank(myrank, PrevRank(outsider, world_size), world_size);

      if (myrank == receiver) {
        std::vector<el_t> input_minus_r(output.numel());
        JointMsgPass(ctx, input_minus_r, sender, backup, receiver);
        pforeach(0, output.numel(), [&](int64_t idx) {
            _out[idx][offset_from_outsider_prev] ^= input_minus_r[idx];
        });
      }
    }

}
