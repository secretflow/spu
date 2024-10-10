// Copyright 2023 Ant Group Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "libspu/mpc/utils/permute.h"

#include <algorithm>
#include <random>

#include "yacl/crypto/rand/rand.h"

#include "libspu/core/memref.h"
#include "libspu/core/type_util.h"

namespace spu::mpc {

Index ring2pv(const MemRef& x) {
  SPU_ENFORCE(x.eltype().isa<BaseRingType>(), "must be ring2k_type, got={}",
              x.eltype());
  Index pv(x.numel());
  DISPATCH_ALL_STORAGE_TYPES(x.eltype().storage_type(), [&]() {
    MemRefView<ScalarT> _x(x);
    pforeach(0, x.numel(), [&](int64_t idx) { pv[idx] = int64_t(_x[idx]); });
  });
  return pv;
}

MemRef applyInvPerm(const MemRef& x, absl::Span<const int64_t> pv) {
  SPU_ENFORCE_EQ(x.shape().ndim(), 1U, "x should be 1-d memref");

  MemRef y(x.eltype(), x.shape());
  DISPATCH_ALL_STORAGE_TYPES(x.eltype().storage_type(), [&]() {
    MemRefView<ScalarT> _x(x);
    MemRefView<ScalarT> _y(y);
    for (int64_t i = 0; i < y.numel(); i++) {
      _y[pv[i]] = _x[i];
    }
  });
  return y;
}

MemRef applyInvPerm(const MemRef& x, const MemRef& pv) {
  SPU_ENFORCE_EQ(x.shape().ndim(), 1U, "x should be 1-d tensor");
  SPU_ENFORCE_EQ(x.shape(), pv.shape(), "x and pv should have same shape");

  MemRef y(x.eltype(), x.shape());
  DISPATCH_ALL_STORAGE_TYPES(x.eltype().storage_type(), [&]() {
    using OT = ScalarT;
    MemRefView<OT> _x(x);
    MemRefView<OT> _y(y);
    DISPATCH_ALL_STORAGE_TYPES(pv.eltype().storage_type(), [&]() {
      using IT = ScalarT;
      MemRefView<IT> _pv(pv);
      for (int64_t i = 0; i < y.numel(); i++) {
        _y[_pv[i]] = _x[i];
      }
    });
  });
  return y;
}

MemRef applyPerm(const MemRef& x, absl::Span<const int64_t> pv) {
  SPU_ENFORCE_EQ(x.shape().ndim(), 1U, "x should be 1-d memref");

  MemRef y(x.eltype(), x.shape());
  DISPATCH_ALL_STORAGE_TYPES(x.eltype().storage_type(), [&]() {
    MemRefView<ScalarT> _x(x);
    MemRefView<ScalarT> _y(y);
    for (int64_t i = 0; i < y.numel(); i++) {
      _y[i] = _x[pv[i]];
    }
  });
  return y;
}

MemRef applyPerm(const MemRef& x, const MemRef& pv) {
  SPU_ENFORCE_EQ(x.shape().ndim(), 1U, "x should be 1-d tensor");
  SPU_ENFORCE_EQ(x.shape(), pv.shape(), "x and pv should have same shape");

  MemRef y(x.eltype(), x.shape());
  DISPATCH_ALL_STORAGE_TYPES(x.eltype().storage_type(), [&]() {
    using OT = ScalarT;
    MemRefView<OT> _x(x);
    MemRefView<OT> _y(y);
    DISPATCH_ALL_STORAGE_TYPES(pv.eltype().storage_type(), [&]() {
      using IT = ScalarT;
      MemRefView<IT> _pv(pv);
      for (int64_t i = 0; i < y.numel(); i++) {
        _y[i] = _x[_pv[i]];
      }
    });
  });
  return y;
}

MemRef genInversePerm(const MemRef& perm) {
  MemRef ret(perm.eltype(), perm.shape());
  DISPATCH_ALL_STORAGE_TYPES(perm.eltype().storage_type(), [&]() {
    MemRefView<ScalarT> _ret(ret);
    MemRefView<ScalarT> _perm(perm);
    for (int64_t i = 0; i < perm.numel(); ++i) {
      _ret[_perm[i]] = ScalarT(i);
    }
  });
  return ret;
}

Index genPermBySort(const MemRef& x) {
  SPU_ENFORCE_EQ(x.shape().ndim(), 1U, "x should be 1-d tensor");
  Index perm(x.shape()[0]);
  std::iota(perm.begin(), perm.end(), 0);
  DISPATCH_ALL_STORAGE_TYPES(x.eltype().storage_type(), [&]() {
    using T = std::make_signed_t<ScalarT>;

    MemRefView<T> _x(x);
    auto cmp = [&_x](int64_t a, int64_t b) { return _x[a] < _x[b]; };
    std::stable_sort(perm.begin(), perm.end(), cmp);
  });
  return perm;
}

Index genRandomPerm(size_t numel, uint128_t seed, uint64_t* ctr) {
  Index perm(numel);
  std::iota(perm.begin(), perm.end(), 0);
  yacl::crypto::ReplayShuffle(perm.begin(), perm.end(), seed, ctr);
  return perm;
}

}  // namespace spu::mpc