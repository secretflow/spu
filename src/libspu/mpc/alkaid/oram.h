// Copyright 2024 Ant Group Co., Ltd.
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

#include "yacl/crypto/block_cipher/symmetric_crypto.h"

#include "libspu/core/ndarray_ref.h"
#include "libspu/mpc/aby3/oram.h"
#include "libspu/mpc/kernel.h"

namespace spu::mpc::alkaid {

// Ashared index, Ashared database
class OramOneHotAA : public OramOneHotKernel {
 public:
  static constexpr const char* kBindName() { return "oram_onehot_aa"; }

  Kind kind() const override { return Kind::Dynamic; }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in,
                  int64_t s) const override;
};

// Ashared index, Public database
class OramOneHotAP : public OramOneHotKernel {
 public:
  static constexpr const char* kBindName() { return "oram_onehot_ap"; }

  Kind kind() const override { return Kind::Dynamic; }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in,
                  int64_t s) const override;
};

class OramReadOA : public OramReadKernel {
 public:
  static constexpr const char* kBindName() { return "oram_read_aa"; }

  ce::CExpr latency() const override {
    // 1 * rotate: 1
    return ce::Const(1);
  }

  ce::CExpr comm() const override {
    // 1 * rotate: k
    auto n = ce::Variable("n", "cols of database");
    return ce::K() * n;
  }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& onehot,
                  const NdArrayRef& db, int64_t offset) const override;
};

class OramReadOP : public OramReadKernel {
 public:
  static constexpr const char* kBindName() { return "oram_read_ap"; }

  ce::CExpr latency() const override {
    // 1 * rotate: 1
    return ce::Const(1);
  }

  ce::CExpr comm() const override {
    // 1 * rotate: k
    auto n = ce::Variable("n", "cols of database");
    return ce::K() * n;
  }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& onehot,
                  const NdArrayRef& db, int64_t offset) const override;
};
}  // namespace spu::mpc::alkaid
