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

#include "libspu/core/ndarray_ref.h"
#include "libspu/core/type_util.h"

namespace spu::mpc::alkaid {

// The layout of Alkaid share.
//
// Alkaid encompasses both the RSS of ABY3 and MRSS implementations. 
// In terms of RSS, the memory view of Alkaid's shares is consistent 
// with that of ABY3. In terms of MRSS,sSimiliar to ABY^3, the three 
// shares of Alkaid are interleaved in a array, for example, given n 
// element and k bytes per-element, we have
//
//   element          address
//   a[0].share0      0
//   a[0].share1      k
//   a[0].share2      2k
//   a[1].share0      3k
//   a[1].share1      4k
//   a[1].share2      5k
//   ...
//   a[n-1].share0    (n-1)*3*k+0
//   a[n-1].share1    (n-1)*3*k+k
//   a[n-1].share2    (n-1)*3*k+2k
//

NdArrayRef getShare(const NdArrayRef& in, int64_t share_idx);

NdArrayRef getFirstShare(const NdArrayRef& in);

NdArrayRef getSecondShare(const NdArrayRef& in);

NdArrayRef getThirdShare(const NdArrayRef& in);

NdArrayRef makeAShare(const NdArrayRef& s1, const NdArrayRef& s2,
                      FieldType field);

NdArrayRef makeAShare(const NdArrayRef& s1, const NdArrayRef& s2, const NdArrayRef& s3,
                      FieldType field);

PtType calcBShareBacktype(size_t nbits);

template <typename T>
std::vector<T> getShareAs(const NdArrayRef& in, size_t share_idx) {
  SPU_ENFORCE(share_idx == 0 || share_idx == 1);

  NdArrayRef share = getShare(in, share_idx);
  SPU_ENFORCE(share.elsize() == sizeof(T));

  auto numel = in.numel();

  std::vector<T> res(numel);
  DISPATCH_UINT_PT_TYPES(share.eltype().as<PtTy>()->pt_type(), [&]() {
    NdArrayView<ScalarT> _share(share);
    for (auto idx = 0; idx < numel; ++idx) {
      res[idx] = _share[idx];
    }
  });

  return res;
}

#define PFOR_GRAIN_SIZE 8192

}  // namespace spu::mpc::alkaid
