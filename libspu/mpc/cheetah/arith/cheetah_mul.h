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

#include "libspu/core/array_ref.h"

namespace spu::mpc::cheetah {

// Implementation for Mul
// Ref: Rathee et al. "Improved Multiplication Triple Generation over Rings
// via RLWE-based AHE"
//  https://eprint.iacr.org/2019/577.pdf
class CheetahMul {
 public:
  explicit CheetahMul(std::shared_ptr<yacl::link::Context> lctx);

  ~CheetahMul();

  CheetahMul& operator=(const CheetahMul&) = delete;

  CheetahMul(const CheetahMul&) = delete;

  CheetahMul(CheetahMul&&) = delete;

  ArrayRef MulOLE(const ArrayRef& inp, yacl::link::Context* conn,
                  bool evaluator);

  ArrayRef MulOLE(const ArrayRef& inp, bool evaluator);

  int Rank() const;

  size_t OLEBatchSize() const;

 private:
  struct Impl;

  std::unique_ptr<Impl> impl_{nullptr};
};

}  // namespace spu::mpc::cheetah
