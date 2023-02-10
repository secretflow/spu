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

#include "libspu/core/array_ref.h"

namespace spu::mpc::cheetah {

class BasicOTProtocols;
class CompareProtocol {
 public:
  static constexpr size_t kCompareRadix = 4;

  explicit CompareProtocol(std::shared_ptr<BasicOTProtocols> base);

  ~CompareProtocol();

  ArrayRef Compute(const ArrayRef& inp, bool greater_than);

  std::array<ArrayRef, 2> ComputeWithEq(const ArrayRef& inp, bool greater_than);

 private:
  ArrayRef DoCompute(const ArrayRef& inp, bool greater_than,
                     ArrayRef* eq = nullptr);

  bool is_sender_{false};
  std::shared_ptr<BasicOTProtocols> basic_ot_prot_;
};

}  // namespace spu::mpc::cheetah
