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

#include "libspu/core/context.h"
#include "libspu/core/ndarray_ref.h"
#include "libspu/core/object.h"
#include "yacl/crypto/hash/hash_interface.h"
#include "yacl/link/link.h"
#include "yacl/crypto/hash/blake3.h"
#include "yacl/crypto/hash/hash_utils.h"
#include "libspu/spu.pb.h"

namespace spu::mpc::fantastic4{

// The MAC state, see Sec 2.2 of Fantastic Four
//  In Rep4, not only in F4, two parties can jointly pass a common message (JMP) to a third party with malicious security
//      by both parties sending msg to the receiver who can lately check the consistency.
//  For better efficiency, one party (sender) send the msg while the other party (backup) locally records the MAC across many JMP instances
//  The amortized cost is thus 1 element per JMP
// We implement this by using MacState to record MACs for different channels
// Same as MP-SPDZ, we only provide MAC state for abort security without further involved procedure for robustness
class Fantastic4MacState : public State {
    std::unique_ptr<yacl::crypto::HashInterface> hash_algo_;

    int64_t mac_len_;

    // my rank in the 4PC
    int64_t my_rank_;

    // refer MP-SPDZ implementation https://github.com/data61/MP-SPDZ/blob/f051dc7222203adaa5161a0e37b8bfaaf7691c16/Protocols/Rep4.hpp#L124
    // when as backup, record MAC of the msg should be sent from sender to receiver
    //   my send_mac_[sender][receiver] should be consistent with the receiver's recv_mac_[sender][me]
    std::vector<std::vector<uint8_t>> send_mac_;

    // refer MP-SPDZ implementation https://github.com/data61/MP-SPDZ/blob/f051dc7222203adaa5161a0e37b8bfaaf7691c16/Protocols/Rep4.hpp#L190
    // when as receiver, record MAC of the msg sent from sender and recorded by backup
    //   my recv_mac_[sender][backup] should be consistent with the backup's send_mac_[sender][me]
    std::vector<std::vector<uint8_t>> recv_mac_;

 private:
  Fantastic4MacState() = default;
 public:
    static constexpr const char* kBindName() { return "Fantastic4MacState"; }

    explicit Fantastic4MacState(const std::shared_ptr<yacl::link::Context>& lctx) {
        hash_algo_ = std::make_unique<yacl::crypto::Blake3Hash>();
        // 128 bit
        mac_len_ = 16;
        my_rank_ = lctx->Rank();

        // there are 4*4 = 16 possible pairs (sender, receiver)
        send_mac_.resize(16);

        // there are 4*4 = 16 possible paris (sender, backup)
        recv_mac_.resize(16);

        for (auto& vec : send_mac_) {
            vec.resize(mac_len_);
            std::fill(vec.begin(), vec.end(), 0);
        }
        for (auto& vec : recv_mac_) {
            vec.resize(mac_len_);
            std::fill(vec.begin(), vec.end(), 0);
        }
    }

    // Update input msg (Span<T const> type) in MAC state
    template <typename T>
    void update_msg(size_t sender, size_t backup, size_t receiver, absl::Span<T const> in){
        yacl::ByteContainerView bv(reinterpret_cast<uint8_t const*>(in.data()),
                             sizeof(T) * in.size());
        hash_algo_->Update(bv);
        std::vector<uint8_t> hash = hash_algo_->CumulativeHash();
        hash_algo_->Reset();
        // SPU_ENFORCE(mac_len_ <= hash.size());

        // if as backup, update channel (sender, receiver)
        if(my_rank_ == static_cast<int64_t>(backup)){
            size_t index = sender * 4 + receiver;
            SPU_ENFORCE(index <= 16);

            auto& target_mac = send_mac_[index];

            std::transform(target_mac.begin(), target_mac.end(),
                        hash.begin(), target_mac.begin(),
                        std::bit_xor<uint8_t>());
        }

        // if as receiver, update channel (sender, backup)
        else if(my_rank_ == static_cast<int64_t>(receiver)){
            size_t index = sender * 4 + backup;
            SPU_ENFORCE(index <= 16);

            auto& target_mac = recv_mac_[index];

            std::transform(target_mac.begin(), target_mac.end(),
                        hash.begin(), target_mac.begin(),
                        std::bit_xor<uint8_t>());
        }
        else{
            SPU_THROW("Sender/Outsider Do not update MAC");
        }
    }

    // Update input msg (NdArrayRef& type) in MAC state
    void update_msg(size_t sender, size_t backup, size_t receiver, const NdArrayRef& in){
        // compact array function from communicator.cc : getOrCreateCompactArray
        NdArrayRef array;
        if (!in.isCompact()) {
            array = in.clone();
        }
        else{
            array = in;
        }
        yacl::ByteContainerView bv(reinterpret_cast<uint8_t const*>(array.data()),
                             in.numel() * in.elsize());

        hash_algo_->Update(bv);
        std::vector<uint8_t> hash = hash_algo_->CumulativeHash();
        hash_algo_->Reset();
        // SPU_ENFORCE(mac_len_ <= hash.size());

        // if as backup, update channel (sender, receiver)
        if(my_rank_ == static_cast<int64_t>(backup)){
            size_t index = sender * 4 + receiver;
            SPU_ENFORCE(index <= 16);

            auto& target_mac = send_mac_[index];

            std::transform(target_mac.begin(), target_mac.end(),
                        hash.begin(), target_mac.begin(),
                        std::bit_xor<uint8_t>());
        }

        // if as receiver, update channel (sender, backup)
        else if(my_rank_ == static_cast<int64_t>(receiver)){
            size_t index = sender * 4 + backup;
            SPU_ENFORCE(index <= 16);

            auto& target_mac = recv_mac_[index];

            std::transform(target_mac.begin(), target_mac.end(),
                        hash.begin(), target_mac.begin(),
                        std::bit_xor<uint8_t>());
        }
        else{
            SPU_THROW("Sender/Outsider Do not update MAC");
        }
    }

    // print MAC in some time for debug and verify
    void print_MAC() {
        for(int64_t sender = 0; sender < 4; sender++) {
            for(int64_t receiver = 0; receiver < 4; receiver++) {
                size_t index = sender * 4 + receiver;
                auto& target_send_mac = send_mac_[index];
                auto& target_recv_mac = recv_mac_[index];
                for(int64_t idx = 0; idx < mac_len_; idx++){
                    printf("My rank = %ld, send_mac_[%ld][%ld][%ld] = %ld\n", my_rank_, sender, receiver, idx, static_cast<uint64_t>(target_send_mac[idx]));
                    printf("My rank = %ld, recv_mac_[%ld][%ld][%ld] = %ld\n", my_rank_, sender, receiver, idx, static_cast<uint64_t>(target_recv_mac[idx]));
                }
            }
        }
    }
};

} // namespace spu::mpc::fantastic4