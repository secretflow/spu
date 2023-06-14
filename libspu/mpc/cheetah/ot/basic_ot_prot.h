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

#include "libspu/core/array_ref.h"
#include "libspu/mpc/cheetah/ot/ferret.h"

namespace spu::mpc::cheetah {

class BasicOTProtocols {
 public:
  explicit BasicOTProtocols(std::shared_ptr<Communicator> conn);

  ~BasicOTProtocols();

  int Rank() const;

  ArrayRef B2A(const ArrayRef &inp);

  // NOTE(lwj): compute the B2A(b) and output to the specified ring
  // Require: input is 1-bit boolean and 1 <= bit_width < k.
  ArrayRef B2ASingleBitWithSize(const ArrayRef &inp, int bit_width);

  ArrayRef RandBits(FieldType filed, size_t numel);

  // msg * select for select \in {0, 1}
  ArrayRef Multiplexer(const ArrayRef &msg, const ArrayRef &select);

  // Create `numel` of AND-triple. Each element contains `k` bits
  // 1 <= k <= field size
  std::array<ArrayRef, 3> AndTriple(FieldType field, size_t numel, size_t k);

  // [a, b, b', c, c'] such that c = a*b and c' = a*b' for the same a
  std::array<ArrayRef, 5> CorrelatedAndTriple(FieldType field, size_t numel);

  ArrayRef BitwiseAnd(const ArrayRef &lhs, const ArrayRef &rhs);

  // Compute the ANDs `lhs & rhs0` and `lhs & rhs1`
  // Require non-packed Boolean currently.
  std::array<ArrayRef, 2> CorrelatedBitwiseAnd(const ArrayRef &lhs,
                                               const ArrayRef &rhs0,
                                               const ArrayRef &rhs1);

  std::shared_ptr<FerretOT> GetSenderCOT() { return ferret_sender_; }

  std::shared_ptr<FerretOT> GetReceiverCOT() { return ferret_receiver_; }

  void Flush();

 protected:
  ArrayRef Compare(const ArrayRef &inp, bool greater_than, bool equality,
                   int radix_base);

  ArrayRef SingleB2A(const ArrayRef &inp, int bit_width = 0);

  ArrayRef PackedB2A(const ArrayRef &inp);

 private:
  std::shared_ptr<Communicator> conn_;
  std::shared_ptr<FerretOT> ferret_sender_;
  std::shared_ptr<FerretOT> ferret_receiver_;
};

}  // namespace spu::mpc::cheetah
