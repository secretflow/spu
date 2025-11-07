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

#include "libspu/mpc/cheetah/ot/basic_ot_prot.h"

namespace spu::mpc::cheetah {

// Truncate and Reduce: returns x >> s mod 2^{k - s}
// Output bit width and fxp precision are k-s and fxp-s;
// However, SPU now doesn't record the precision/bit width in Value, so user
// must "remember" the precision and bit width
//
// for exact version:
// REF: SIRNN: A Math Library for Secure RNN Inference
// for approx version, just ignore the truncated wrap
class RingTruncateAndReduceProtocol {
 public:
  struct Meta {
    bool exact = false;

    // TODO: support auto detect of src_ring and dst_ring
    FieldType src_ring;
    FieldType dst_ring;

    int64_t src_width;  // [0, 2^k)
    int64_t dst_width;  // [0, 2^k')
  };

  explicit RingTruncateAndReduceProtocol(
      const std::shared_ptr<BasicOTProtocols> &base)
      : basic_ot_prot_(base) {}

  ~RingTruncateAndReduceProtocol() = default;

  NdArrayRef Compute(const NdArrayRef &inp, const NdArrayRef &wrap_s,
                     const Meta &meta);

 private:
  NdArrayRef ComputeWithoutWrap(const NdArrayRef &inp, const Meta &meta);

  // just B2A
  NdArrayRef ComputeWithWrap(const NdArrayRef &inp, const NdArrayRef &wrap_s,
                             const Meta &meta);

  // 1-bit approx truncate
  NdArrayRef ComputeApprox(const NdArrayRef &inp, const Meta &meta);

  std::shared_ptr<BasicOTProtocols> basic_ot_prot_ = nullptr;
};

}  // namespace spu::mpc::cheetah