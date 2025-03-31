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

// TODO: complete me
class Fantastic4MacState : public State {
    std::unique_ptr<yacl::crypto::HashInterface> hash_algo_;
    int64_t mac_len_;
    std::vector<std::vector<uint8_t>> mac;
    int64_t my_rank_;
 private:
  Fantastic4MacState() = default;
 public:
    static constexpr const char* kBindName() { return "Fantastic4MacState"; }

    explicit Fantastic4MacState(const std::shared_ptr<yacl::link::Context>& lctx) {
        hash_algo_ = std::make_unique<yacl::crypto::Blake3Hash>();
        // 128 bit
        mac_len_ = 16;
        my_rank_ = lctx->Rank();
        // The MAC of Pi sends to Pj be stored in i*4+j
        mac.resize(16);
        for (auto& vec : mac) {
            vec.resize(mac_len_);
            std::fill(vec.begin(), vec.end(), 0);
        }
    }

    template <typename T>
    void update_msg(size_t sender, size_t backup, size_t receiver, absl::Span<T const> in){
        yacl::ByteContainerView bv(reinterpret_cast<uint8_t const*>(in.data()),
                             sizeof(T) * in.size());
        hash_algo_->Update(bv);
        std::vector<uint8_t> hash = hash_algo_->CumulativeHash();
        // SPU_ENFORCE(mac_len_ <= hash.size());
        size_t index = sender * 4 + receiver;
        SPU_ENFORCE(index <= 16);

        auto& target_mac = mac[index];

        std::transform(target_mac.begin(), target_mac.end(),
                      hash.begin(), target_mac.begin(),
                      std::bit_xor<uint8_t>());
    }

    void update_msg(size_t sender, size_t backup, size_t receiver, const NdArrayRef& in){
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
        // SPU_ENFORCE(mac_len_ <= hash.size());
        size_t index = sender * 4 + receiver;
        SPU_ENFORCE(index <= 16);

        auto& target_mac = mac[index];

        std::transform(target_mac.begin(), target_mac.end(),
                      hash.begin(), target_mac.begin(),
                      std::bit_xor<uint8_t>());
    }

    // Only for verify at the end of computation
    void print_MAC() {
        for(int64_t sender = 0; sender < 4; sender++) {
            for(int64_t receiver = 0; receiver < 4; receiver++) {
                size_t index = sender * 4 + receiver;
                auto& target_mac = mac[index];
                for(int64_t idx = 0; idx < mac_len_; idx++){
                    printf("My rank = %ld, mac[%ld][%ld][%ld] = %ld", my_rank_, sender, receiver, idx, static_cast<uint64_t>(target_mac[idx]));
                }
            }
        }
    }
};

} // namespace spu::mpc::fantastic4