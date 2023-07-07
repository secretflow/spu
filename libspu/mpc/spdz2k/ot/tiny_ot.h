// Copyright 2023 Ant Group Co., Ltd.
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
#include <vector>

#include "yacl/crypto/primitives/ot/ot_store.h"

#include "libspu/mpc/common/communicator.h"

namespace spu::mpc::spdz2k {

struct AuthBit {
  std::vector<bool> choices;
  std::vector<uint128_t> mac;
  uint128_t key;
};

uint128_t GenSharedSeed(const std::shared_ptr<Communicator>& comm);
// TinyMacCheck protocol
// Reference: https://eprint.iacr.org/2014/101.pdf
// Page 17, protocol 10.
bool TinyMacCheck(const std::shared_ptr<Communicator>& comm,
                  std::vector<bool> open_bits, const AuthBit& bits);

AuthBit RandomBits(const std::shared_ptr<Communicator>& comm,
                   const std::shared_ptr<yacl::crypto::OtSendStore>& send_opts,
                   const std::shared_ptr<yacl::crypto::OtRecvStore>& recv_opts,
                   size_t size, uint128_t tinyot_key);

// Reference: https://eprint.iacr.org/2014/101.pdf
// Page 10, Protocol 7.
std::tuple<AuthBit, AuthBit, AuthBit> TinyMul(
    const std::shared_ptr<Communicator>& comm,
    const std::shared_ptr<yacl::crypto::OtSendStore>& send_opts,
    const std::shared_ptr<yacl::crypto::OtRecvStore>& recv_opts, size_t size,
    uint128_t tinyot_key);

};  // namespace spu::mpc::spdz2k