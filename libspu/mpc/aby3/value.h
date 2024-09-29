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

#include "libspu/core/memref.h"
#include "libspu/core/type_util.h"

namespace spu::mpc::aby3 {

// The layout of Aby3 share.
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
// you can treat aby3 share as std::complex<T>, where
//   real(x) is the first share piece.
//   imag(x) is the second share piece.

MemRef getShare(const MemRef& in, int64_t share_idx);

MemRef getFirstShare(const MemRef& in);

MemRef getSecondShare(const MemRef& in);

MemRef makeArithShare(const MemRef& s1, const MemRef& s2,
                      SemanticType seman_type, size_t valid_bits);

SemanticType calcBShareSemanticType(size_t nbits);

template <typename T>
std::vector<T> getShareAs(const MemRef& in, size_t share_idx) {
  SPU_ENFORCE(share_idx == 0 || share_idx == 1);

  MemRef share = getShare(in, share_idx);
  SPU_ENFORCE(SizeOf(share.eltype().storage_type()) == sizeof(T));

  auto numel = in.numel();

  std::vector<T> res(numel);
  DISPATCH_ALL_STORAGE_TYPES(share.eltype().storage_type(), [&]() {
    MemRefView<ScalarT> _share(share);
    for (auto idx = 0; idx < numel; ++idx) {
      res[idx] = _share[idx];
    }
  });

  return res;
}

#define PFOR_GRAIN_SIZE 8192

}  // namespace spu::mpc::aby3
