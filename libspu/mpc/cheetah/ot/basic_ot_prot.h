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

#include "libspu/core/memref.h"
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

  MemRef B2A(const MemRef &inp);

  MemRef RandBits(size_t filed, const Shape &shape);

  // NOTE(lwj): compute the B2A(b) and output to the specified ring
  // Require: input is 1-bit boolean and 1 <= bit_width < k.
  MemRef B2ASingleBitWithSize(const MemRef &inp, int bit_width);

  // msg * select for select \in {0, 1}
  MemRef Multiplexer(const MemRef &msg, const MemRef &select);

  // multiplexer with private choices (sender part)
  MemRef PrivateMulxSend(const MemRef &msg);

  // multiplexer with private choices (recv part)
  MemRef PrivateMulxRecv(const MemRef &msg, const MemRef &select);

  MemRef PrivateMulxRecv(const MemRef &msg, absl::Span<const uint8_t> select);

  // Create `numel` of AND-triple. Each element contains `k` bits
  // 1 <= k <= field size
  std::array<MemRef, 3> AndTriple(size_t field, const Shape &shape, size_t k);

  // [a, b, b', c, c'] such that c = a*b and c' = a*b' for the same a
  std::array<MemRef, 5> CorrelatedAndTriple(size_t field, const Shape &shape);

  MemRef BitwiseAnd(const MemRef &lhs, const MemRef &rhs);

  // Compute the ANDs `lhs & rhs0` and `lhs & rhs1`
  std::array<MemRef, 2> CorrelatedBitwiseAnd(const MemRef &lhs,
                                             const MemRef &rhs0,
                                             const MemRef &rhs1);

  std::shared_ptr<FerretOtInterface> GetSenderCOT() { return ferret_sender_; }

  std::shared_ptr<FerretOtInterface> GetReceiverCOT() {
    return ferret_receiver_;
  }

  void Flush();

 protected:
  MemRef SingleB2A(const MemRef &inp, int bit_width = 0);

  MemRef PackedB2A(const MemRef &inp);

 private:
  std::shared_ptr<Communicator> conn_;
  std::shared_ptr<FerretOtInterface> ferret_sender_;
  std::shared_ptr<FerretOtInterface> ferret_receiver_;
};

}  // namespace spu::mpc::cheetah
