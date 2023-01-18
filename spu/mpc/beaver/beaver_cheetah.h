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

#include "spu/crypto/ot/silent/primitives.h"
#include "spu/mpc/beaver/beaver.h"

namespace spu::mpc {

namespace cheetah {
class MulAA;
class MatMulAA;
}  // namespace cheetah

// Cheetah beaver implementation.
// BeaverCheetah = AMul(BeaverHE) + And(OT)
class BeaverCheetah : public Beaver {
 protected:
  // Implementation for Mul
  // Ref: Rathee et al. "Improved Multiplication Triple Generation over Rings
  // via RLWE-based AHE"
  //  https://eprint.iacr.org/2019/577.pdf
  struct MulImpl;
  // Implementation for Dot using MatVec
  // Ref: Huang et al. "Cheetah: Lean and Fast Secure Two-Party Deep Neural
  // Network Inference"
  //  https://eprint.iacr.org/2022/207.pdf
  struct DotImpl;

  std::shared_ptr<MulImpl> mul_impl_;

  std::shared_ptr<DotImpl> dot_impl_;

  std::shared_ptr<spu::CheetahPrimitives> ot_primitives_{nullptr};

  std::shared_ptr<yacl::link::Context> lctx_;
  // Olivious Linear Evaluation (OLE)
  // compute the share of a*b where Alice holds `a` and Bob holds `b` privately.
  friend class cheetah::MulAA;
  ArrayRef MulOLE(const ArrayRef& inp, yacl::link::Context* conn,
                  bool evaluator);

  friend class cheetah::MatMulAA;
  // Compute the matrix product A*B where Alice inputs A and Bob inputs B
  // where |A| = M*K and |B| = K*N.
  // Assume the RHS matrix B is given in the column-major order.
  ArrayRef DotOLE(const ArrayRef& inp, yacl::link::Context* conn, size_t M,
                  size_t N, size_t K, bool is_right_hand_side);

 public:
  explicit BeaverCheetah(std::shared_ptr<yacl::link::Context> lctx);

  const spu::CheetahPrimitives* OTPrimitives() const {
    return ot_primitives_.get();
  }

  spu::CheetahPrimitives* OTPrimitives() { return ot_primitives_.get(); }

  Beaver::Triple Mul(FieldType field, size_t size) override;

  Beaver::Triple And(FieldType field, size_t size) override;

  Beaver::Triple Dot(FieldType field, size_t M, size_t N, size_t K) override;

  bool SupportTrunc() override { return false; }
  Beaver::Pair Trunc(FieldType field, size_t size, size_t bits) override;

  bool SupportRandBit() override { return false; }
  ArrayRef RandBit(FieldType field, size_t size) override;

  std::shared_ptr<yacl::link::Context> GetLink() const { return lctx_; }
};

}  // namespace spu::mpc
