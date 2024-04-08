// Copyright 2021 Ant Group Co., Ltd.
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

#include "libspu/core/ndarray_ref.h"
#include "libspu/mpc/cheetah/ot/ferret_ot_interface.h"
#include "libspu/mpc/common/communicator.h"

#include "libspu/spu.pb.h"

namespace spu::mpc::cheetah {

class BasicOTProtocols {
 public:
  explicit BasicOTProtocols(std::shared_ptr<Communicator> conn,
                            CheetahOtKind kind);

  ~BasicOTProtocols();

  int Rank() const;

  NdArrayRef B2A(const NdArrayRef &inp);

  NdArrayRef RandBits(FieldType filed, const Shape &shape);

  // NOTE(lwj): compute the B2A(b) and output to the specified ring
  // Require: input is 1-bit boolean and 1 <= bit_width < k.
  NdArrayRef B2ASingleBitWithSize(const NdArrayRef &inp, int bit_width);

  // msg * select for select \in {0, 1}
  NdArrayRef Multiplexer(const NdArrayRef &msg, const NdArrayRef &select);

  // multiplexer with private choices (sender part)
  NdArrayRef PrivateMulxSend(const NdArrayRef &msg);

  // multiplexer with private choices (recv part)
  NdArrayRef PrivateMulxRecv(const NdArrayRef &msg, const NdArrayRef &select);

  NdArrayRef PrivateMulxRecv(const NdArrayRef &msg,
                             absl::Span<const uint8_t> select);

  // Create `numel` of AND-triple. Each element contains `k` bits
  // 1 <= k <= field size
  std::array<NdArrayRef, 3> AndTriple(FieldType field, const Shape &shape,
                                      size_t k);

  // [a, b, b', c, c'] such that c = a*b and c' = a*b' for the same a
  std::array<NdArrayRef, 5> CorrelatedAndTriple(FieldType field,
                                                const Shape &shape);

  NdArrayRef BitwiseAnd(const NdArrayRef &lhs, const NdArrayRef &rhs);

  // Compute the ANDs `lhs & rhs0` and `lhs & rhs1`
  std::array<NdArrayRef, 2> CorrelatedBitwiseAnd(const NdArrayRef &lhs,
                                                 const NdArrayRef &rhs0,
                                                 const NdArrayRef &rhs1);

  std::shared_ptr<FerretOtInterface> GetSenderCOT() { return ferret_sender_; }

  std::shared_ptr<FerretOtInterface> GetReceiverCOT() {
    return ferret_receiver_;
  }

  void Flush();

 protected:
  NdArrayRef SingleB2A(const NdArrayRef &inp, int bit_width = 0);

  NdArrayRef PackedB2A(const NdArrayRef &inp);

 private:
  std::shared_ptr<Communicator> conn_;
  std::shared_ptr<FerretOtInterface> ferret_sender_;
  std::shared_ptr<FerretOtInterface> ferret_receiver_;
};

}  // namespace spu::mpc::cheetah
