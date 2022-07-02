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

#include "absl/types/span.h"

#include "spu/mpc/beaver/prg_tensor.h"

namespace spu::mpc {

class TrustedParty {
 private:
  std::vector<std::optional<PrgSeed>> seeds_;
  mutable std::mutex seeds_mutex_;

 public:
  void setSeed(size_t rank, size_t world_size, const PrgSeed& seed);

  std::vector<PrgSeed> getSeeds() const;

  ArrayRef adjustMul(absl::Span<const PrgArrayDesc> descs);

  ArrayRef adjustDot(absl::Span<const PrgArrayDesc> descs, size_t M, size_t N,
                     size_t K);

  ArrayRef adjustAnd(absl::Span<const PrgArrayDesc> descs);

  ArrayRef adjustTrunc(absl::Span<const PrgArrayDesc> descs, size_t bits);

  ArrayRef adjustRandBit(const PrgArrayDesc& descs);
};

}  // namespace spu::mpc
