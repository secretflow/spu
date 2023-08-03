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

#include <mutex>
#include <optional>
#include <shared_mutex>
#include <utility>

#include "absl/types/span.h"

#include "libspu/mpc/common/prg_tensor.h"

namespace spu::mpc::securenn {

class TrustedParty {
 private:
  using Seeds = absl::Span<const PrgSeed>;
  using Descs = absl::Span<const PrgArrayDesc>;

 public:
  static ArrayRef adjustMul(Descs descs, Seeds seeds);

  static ArrayRef adjustDot(Descs descs, Seeds seeds, size_t M, size_t N,
                            size_t K);

  static ArrayRef adjustAnd(Descs descs, Seeds seeds);

  static ArrayRef adjustTrunc(Descs descs, Seeds seeds, size_t bits);

  static std::pair<ArrayRef, ArrayRef> adjustTruncPr(Descs descs, Seeds seeds,
                                                     size_t bits);

  static ArrayRef adjustRandBit(Descs descs, Seeds seeds);
};

}  // namespace spu::mpc::securenn
