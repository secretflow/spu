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

// use the SIRNN version
class MixMulProtocol {
 public:
  static constexpr size_t kHeuristicBound = 2;

  struct Meta {
    SignType sign_x = SignType::Unknown;
    SignType sign_y = SignType::Unknown;

    // FIXME: only support in signed arith, and the bw must be full ring
    bool use_heuristic = false;

    bool signed_arith = false;

    int64_t bw_x;
    int64_t bw_y;
    int64_t bw_out;

    FieldType field_x;
    FieldType field_y;
    FieldType field_out;
  };

  explicit MixMulProtocol(const std::shared_ptr<BasicOTProtocols>& base)
      : basic_ot_prot_(base) {
    SPU_ENFORCE(base != nullptr);
  }

  ~MixMulProtocol() = default;

  NdArrayRef Compute(const NdArrayRef& x, const NdArrayRef& y,
                     const Meta& meta);

  // The following two functions should not be called directly unless you know
  // exactly what you are doing.
  // We put them here for unittest of these important building blocks.

  // unsigned private multiplication
  // TODO: maybe define an exclusive Meta class
  NdArrayRef CrossMul(const NdArrayRef& inp, const Meta& meta);

  // return (z, wrap_x, wrap_y), where z = x * y
  // wrap is for signed adjustment, and Bshr is enough
  std::tuple<NdArrayRef, NdArrayRef, NdArrayRef> UnsignedMixMul(
      const NdArrayRef& x, const NdArrayRef& y, const Meta& meta);

  struct WrapMeta {
    SignType sign;

    FieldType src_ring;
    int64_t src_width;

    FieldType dst_ring = FM8;  // will always be FM8
    int64_t dst_width = 1;     // will always be 1
  };

  NdArrayRef ComputeWrap(const NdArrayRef& inp, const WrapMeta& meta);

 private:
  std::shared_ptr<BasicOTProtocols> basic_ot_prot_;

  // These wrap functions are mainly adapted from ext_prot.cc
  // But we only need Bshare here because we have to compute w_x * y,
  // If w_x is Ashare, then this mul is expensive

  // w = msbA | msbB
  NdArrayRef MSB0ToWrap(const NdArrayRef& inp, const WrapMeta& meta);

  // w = msbA & msbB
  NdArrayRef MSB1ToWrap(const NdArrayRef& inp, const WrapMeta& meta);
};
}  // namespace spu::mpc::cheetah