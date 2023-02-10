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
  explicit BasicOTProtocols(std::shared_ptr<yacl::link::Context> conn);

  ~BasicOTProtocols();

  int Rank() const;

  ArrayRef B2A(const ArrayRef &inp);

  ArrayRef RandBits(FieldType filed, size_t numel);

  std::array<ArrayRef, 3> AndTriple(FieldType field, size_t numel,
                                    bool packed = true);

  ArrayRef BitwiseAnd(const ArrayRef &lhs, const ArrayRef &rhs);

  std::shared_ptr<FerretOT> GetSenderCOT() { return ferret_sender_; }

  std::shared_ptr<FerretOT> GetReceiverCOT() { return ferret_receiver_; }

  std::shared_ptr<yacl::link::Context> GetLink() { return conn_; }

  void Flush();

 protected:
  ArrayRef Compare(const ArrayRef &inp, bool greater_than, bool equality,
                   int radix_base);

  ArrayRef SingleB2A(const ArrayRef &inp);

  ArrayRef PackedB2A(const ArrayRef &inp);

 private:
  std::shared_ptr<yacl::link::Context> conn_;
  std::shared_ptr<FerretOT> ferret_sender_;
  std::shared_ptr<FerretOT> ferret_receiver_;
};

}  // namespace spu::mpc::cheetah
