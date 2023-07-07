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

#pragma once

#include "yacl/crypto/primitives/ot/base_ot.h"
#include "yacl/link/context.h"

#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/spdz2k/beaver/beaver_interface.h"
#include "libspu/mpc/spdz2k/beaver/trusted_party.h"
#include "libspu/mpc/spdz2k/ot/basic_ot_prot.h"

namespace spu::mpc::spdz2k {

class BeaverTinyOt final : public Beaver {
 protected:
  // Only for rank0 party.
  TrustedParty tp_;

  std::shared_ptr<Communicator> comm_;

  std::shared_ptr<PrgState> prg_state_;

  PrgSeed seed_;

  // tinyOT alpha
  uint128_t tinyot_key_;

  // spzd key
  uint128_t spdz_key_;

  // base OT
  std::shared_ptr<yacl::crypto::OtRecvStore> recv_opts_;
  std::shared_ptr<yacl::crypto::OtSendStore> send_opts_;

  // ferret ot
  std::shared_ptr<BasicOTProtocols> spdz2k_ot_primitives_;

  // security parameters
  static constexpr int kappa_ = 128;

 public:
  using Triple = std::tuple<ArrayRef, ArrayRef, ArrayRef>;
  using Pair = std::pair<ArrayRef, ArrayRef>;
  using Pair_Pair = std::pair<Pair, Pair>;
  using Triple_Pair = std::pair<Triple, Triple>;

  BeaverTinyOt(std::shared_ptr<yacl::link::Context> lctx);

  uint128_t InitSpdzKey(FieldType field, size_t s) override;

  ArrayRef AuthArrayRef(const ArrayRef& value, FieldType field, size_t k,
                        size_t s) override;

  Pair AuthCoinTossing(FieldType field, size_t size, size_t k,
                       size_t s) override;

  Triple_Pair AuthMul(FieldType field, size_t size, size_t k,
                      size_t s) override;

  Triple_Pair AuthDot(FieldType field, size_t M, size_t N, size_t K, size_t k,
                      size_t s) override;

  Triple_Pair AuthAnd(FieldType field, size_t size, size_t s) override;

  Pair_Pair AuthTrunc(FieldType field, size_t size, size_t bits, size_t k,
                      size_t s) override;

  Pair AuthRandBit(FieldType field, size_t size, size_t k, size_t s) override;

  // Check the opened value only
  bool BatchMacCheck(const ArrayRef& open_value, const ArrayRef& mac, size_t k,
                     size_t s);
  // Open the low k_bits of value only
  std::pair<ArrayRef, ArrayRef> BatchOpen(const ArrayRef& value,
                                          const ArrayRef& mac, size_t k,
                                          size_t s);

  // public coin, used in malicious model, all party generate new seed, then
  // get exactly the same random variable.
  ArrayRef genPublCoin(FieldType field, size_t numel);

  // ROT encapsulation
  // s[i] = (a[i] == 0) ? q0[i] : q1[i]
  void rotSend(FieldType field, ArrayRef* q0, ArrayRef* q1);
  void rotRecv(FieldType field, const ArrayRef& a, ArrayRef* s);

  // Vector-OLE encapsulation
  // a[i] = b[i] + x[i] * alpha[i]
  // Sender: input x, receive b
  // Receiver: input alpha, receive a
  ArrayRef voleSend(FieldType field, const ArrayRef& x);
  ArrayRef voleRecv(FieldType field, const ArrayRef& alpha);

  // Private Matrix Multiplication by VOLE
  // W = V + A dot B
  // Sender: input A, receive V
  // Receiver: input B, receive W
  ArrayRef voleSendDot(FieldType field, const ArrayRef& x, size_t M, size_t N,
                       size_t K);
  ArrayRef voleRecvDot(FieldType field, const ArrayRef& alpha, size_t M,
                       size_t N, size_t K);

  // Generate semi-honest dot triple
  Triple dot(FieldType field, size_t M, size_t N, size_t K, size_t k, size_t s);
};

}  // namespace spu::mpc::spdz2k
