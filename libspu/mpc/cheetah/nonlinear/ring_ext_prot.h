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

#include <memory>

#include "libspu/core/ndarray_ref.h"
#include "libspu/core/type_util.h"

namespace spu::mpc::cheetah {

class BasicOTProtocols;

// Given [x] the share in modulo 2^k, compute the share in modulo 2^k' such that
// k' > k
class RingExtendProtocol {
 public:
  static constexpr size_t kHeuristicBound = 2;

  struct Meta {
    SignType sign = SignType::Unknown;
    bool use_heuristic = false;
    bool signed_arith = true;

    FieldType src_ring;
    FieldType dst_ring;

    int64_t src_width;  // [0, 2^k)
    int64_t dst_width;  // [0, 2^k')
  };

  explicit RingExtendProtocol(const std::shared_ptr<BasicOTProtocols> &base);

  ~RingExtendProtocol();

  NdArrayRef Compute(const NdArrayRef &inp, const Meta &meta);

 private:
  // input is unsigned value
  NdArrayRef UnsignedExtend(const NdArrayRef &inp, const Meta &meta);

  NdArrayRef ComputeWrap(const NdArrayRef &inp, const Meta &meta);

  // w = msbA | msbB
  NdArrayRef MSB0ToWrap(const NdArrayRef &inp, const Meta &meta);

  // w = msbA & msbB
  NdArrayRef MSB1ToWrap(const NdArrayRef &inp, const Meta &meta);

  std::shared_ptr<BasicOTProtocols> basic_ot_prot_ = nullptr;
};

}  // namespace spu::mpc::cheetah
