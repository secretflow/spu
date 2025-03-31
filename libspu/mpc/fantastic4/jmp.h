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

#define OPTIMIZED_F4

#include "libspu/core/context.h"

#include "libspu/mpc/kernel.h"

#include "libspu/mpc/fantastic4/value.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/prg_state.h"

namespace spu::mpc::fantastic4 {
    size_t PrevRank(size_t rank, size_t world_size);
    size_t OffsetRank(size_t myrank, size_t other, size_t world_size);

    template <typename el_t>
    void JointInputArith(KernelEvalContext* ctx, std::vector<el_t>& input, NdArrayRef& output, size_t sender, size_t backup, size_t receiver, size_t outsider){
      auto* comm = ctx->getState<Communicator>();
      size_t world_size =  comm->getWorldSize();
      auto* prg_state = ctx->getState<PrgState>();
      auto myrank = comm->getRank();

      using shr_t = std::array<el_t, 3>;
      NdArrayView<shr_t> _out(output);

      size_t receiver_prev_rank = PrevRank(receiver, world_size);
      size_t offset_from_receiver_prev = OffsetRank(myrank, receiver_prev_rank, world_size);
      size_t offset_from_outsider_prev = OffsetRank(myrank, (outsider + 4 - 1)%4 , world_size);

      if(myrank != receiver){
        // Non-Interactive Random Masks Generation.
        std::vector<el_t> r(output.numel());
        if(offset_from_receiver_prev == 0){
            // should use PRG[0]
            prg_state->fillPrssTuple<el_t>(r.data(), nullptr, nullptr , r.size(), PrgState::GenPrssCtrl::First);
        }
        if(offset_from_receiver_prev == 1){
            // should use PRG[1]
            prg_state->fillPrssTuple<el_t>(nullptr, r.data(), nullptr , r.size(), PrgState::GenPrssCtrl::Second);
        }
        if(offset_from_receiver_prev == 2){
            // should use PRG[2]
            prg_state->fillPrssTuple<el_t>(nullptr, nullptr, r.data(), r.size(), PrgState::GenPrssCtrl::Third);
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
    void JointInputBool(KernelEvalContext* ctx, std::vector<el_t>& input, NdArrayRef& output, size_t sender, size_t backup, size_t receiver, size_t outsider){
      auto* comm = ctx->getState<Communicator>();
      size_t world_size =  comm->getWorldSize();
      auto* prg_state = ctx->getState<PrgState>();
      auto myrank = comm->getRank();

      using shr_t = std::array<el_t, 3>;
      NdArrayView<shr_t> _out(output);

      // Receiver's Previous Party Rank
      // The mask corresponds to the prev party of receiver, receiver doesn't have the correpsonding PRG of its prev party
      size_t receiver_prev_rank = PrevRank(receiver, world_size);
      size_t offset_from_receiver_prev = OffsetRank(myrank, receiver_prev_rank, world_size);
      size_t offset_from_outsider_prev = OffsetRank(myrank, (outsider + 4 - 1)%4 , world_size);

      if(myrank != receiver){
        // Non-Interactive Random Masks Generation.
        std::vector<el_t> r(output.numel());

        if(offset_from_receiver_prev == 0){
            // should use PRG[0]
            prg_state->fillPrssTuple<el_t>(r.data(), nullptr, nullptr , r.size(), PrgState::GenPrssCtrl::First);
        }
        if(offset_from_receiver_prev == 1){
            // should use PRG[1]
            prg_state->fillPrssTuple<el_t>(nullptr, r.data(), nullptr , r.size(), PrgState::GenPrssCtrl::Second);
        }
        if(offset_from_receiver_prev == 2){
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
