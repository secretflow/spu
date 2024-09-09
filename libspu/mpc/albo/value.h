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

namespace spu::mpc::albo {

// The layout of Albo share.
//
// Two shares are interleaved in a array, for example, given n element and k
// bytes per-element.
//
//   element          address
//   a[0].share0      0
//   a[0].share1      k
//   a[1].share0      2k
//   a[1].share1      3k
//   ...
//   a[n-1].share0    (n-1)*2*k+0
//   a[n-1].share1    (n-1)*2*k+k
//
// you can treat albo share as std::complex<T>, where
//   real(x) is the first share piece.
//   imag(x) is the second share piece.

NdArrayRef getShare(const NdArrayRef& in, int64_t share_idx);

NdArrayRef getFirstShare(const NdArrayRef& in);

NdArrayRef getSecondShare(const NdArrayRef& in);

NdArrayRef makeAShare(const NdArrayRef& s1, const NdArrayRef& s2, const NdArrayRef& s3,
                      FieldType field);

PtType calcBShareBacktype(size_t nbits);

template <typename T>
std::vector<T> getShareAs(const NdArrayRef& in, size_t share_idx) {
  SPU_ENFORCE(share_idx == 0 || share_idx == 1 || share_idx == 2);

  NdArrayRef share = getShare(in, share_idx);
  SPU_ENFORCE(share.elsize() == sizeof(T));

  auto numel = in.numel();

  std::vector<T> res(numel);
  DISPATCH_UINT_PT_TYPES(share.eltype().as<PtTy>()->pt_type(), "_", [&]() {
    NdArrayView<ScalarT> _share(share);
    for (auto idx = 0; idx < numel; ++idx) {
      res[idx] = _share[idx];
    }
  });

  return res;
}

#define PFOR_GRAIN_SIZE 8192

}  // namespace spu::mpc::albo
