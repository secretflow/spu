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

class TruncateProtocol {
 public:
  enum class MSB_st {
    zero,
    one,
    unknown,
  };

  struct Meta {
    MSB_st msb;
    bool use_heuristic;
    bool signed_arith;
    Meta() : msb(MSB_st::unknown), use_heuristic(false), signed_arith(true) {}
  };

  explicit TruncateProtocol(std::shared_ptr<BasicOTProtocols> base);

  ~TruncateProtocol();

  ArrayRef Compute(const ArrayRef &inp, Meta meta, size_t bits);

 private:
  ArrayRef ComputeWrap(const ArrayRef &inp, const Meta &meta);

  // w = msbA | msbB
  ArrayRef MSB0ToWrap(const ArrayRef &inp);

  // w = msbA & msbB
  ArrayRef MSB1ToWrap(const ArrayRef &inp);

  std::shared_ptr<BasicOTProtocols> basic_ot_prot_{nullptr};
};

}  // namespace spu::mpc::cheetah
