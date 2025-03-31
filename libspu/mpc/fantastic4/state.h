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
#include "libspu/spu.pb.h"

namespace spu::mpc::fantastic4 {

// TODO: complete me
class Fantastic4MacState : public State {
    std::unique_ptr<yacl::crypto::HashInterface> hash_algo_;
    size_t mac_len_;
    NdArrayRef send_hashes_(ring2k_t, {4, 4});
    NdArrayRef used_channels_(bool, {4, 4});

 private:
  Fantastic4MacState() = default;
 public:
    static constexpr const char* kBindName() { return "Fantastic4MacState"; }

    explicit Fantastic4MacState(const std::shared_ptr<yacl::link::Context>& lctx) {
        hash_algo_ = std::make_unique<yacl::crypto::Blake3Hash>();
        mac_len_ = 128;
    }
}

} // namespace spu::mpc::fantastic4