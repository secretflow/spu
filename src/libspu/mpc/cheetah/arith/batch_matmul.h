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

#include <memory>

#include "yacl/link/context.h"
#include "iomanip"

#include "libspu/core/ndarray_ref.h"
#include "libspu/mpc/cheetah/arith/common.h"


namespace spu::mpc::cheetah {


class BatchMatMul {
 public:
  explicit BatchMatMul(std::shared_ptr<yacl::link::Context> lctx,
                      bool allow_high_prob_one_bit_error = false);

  ~BatchMatMul();

  BatchMatMul& operator=(const BatchMatMul&) = delete;

  BatchMatMul(const BatchMatMul&) = delete;

  BatchMatMul(BatchMatMul&&) = delete;

  void LazyInitKeys(FieldType field, uint32_t msg_width_hint = 0);

  // LHS.shape BxMxK, RHS.shape BxKxL => BxMxL
  // make sure to call InitKeys first
  NdArrayRef BatchDotOLE(const NdArrayRef& inp, yacl::link::Context* conn,
                         const Shape4D& dim4, bool is_self_lhs);

  int Rank() const;

  size_t OLEBatchSize() const;


 private:
  struct Impl;

  std::unique_ptr<Impl> impl_{nullptr};
};

}  // namespace spu::mpc::cheetah

inline std::ostream &operator<<(std::ostream &stream, seal::parms_id_type parms_id)
{
    /*
    Save the formatting information for std::cout.
    */
    std::ios old_fmt(nullptr);
    old_fmt.copyfmt(std::cout);

    stream << std::hex << std::setfill('0') << std::setw(16) << parms_id[0] << " " << std::setw(16) << parms_id[1]
           << " " << std::setw(16) << parms_id[2] << " " << std::setw(16) << parms_id[3] << " ";

    /*
    Restore the old std::cout formatting.
    */
    std::cout.copyfmt(old_fmt);

    return stream;
}