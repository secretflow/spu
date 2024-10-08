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

#include "libspu/core/ndarray_ref.h"
#include "libspu/core/type_util.h"

namespace spu::mpc {

Index ring2pv(const NdArrayRef& x) {
  SPU_ENFORCE(x.eltype().isa<Ring2k>(), "must be ring2k_type, got={}",
              x.eltype());
  const auto field = x.eltype().as<Ring2k>()->field();
  Index pv(x.numel());
  DISPATCH_ALL_FIELDS(field, [&]() {
    NdArrayView<ring2k_t> _x(x);
    pforeach(0, x.numel(), [&](int64_t idx) { pv[idx] = int64_t(_x[idx]); });
  });
  return pv;
}

NdArrayRef applyInvPerm(const NdArrayRef& x, absl::Span<const int64_t> pv) {
  SPU_ENFORCE_EQ(x.shape().ndim(), 1U, "x should be 1-d tensor");

  NdArrayRef y(x.eltype(), x.shape());
  const auto field = x.eltype().as<Ring2k>()->field();
  DISPATCH_ALL_FIELDS(field, [&]() {
    NdArrayView<ring2k_t> _x(x);
    NdArrayView<ring2k_t> _y(y);
    for (int64_t i = 0; i < y.numel(); i++) {
      _y[pv[i]] = _x[i];
    }
  });
  return y;
}

NdArrayRef applyInvPerm(const NdArrayRef& x, const NdArrayRef& pv) {
  SPU_ENFORCE_EQ(x.shape().ndim(), 1U, "x should be 1-d tensor");
  SPU_ENFORCE_EQ(x.shape(), pv.shape(), "x and pv should have same shape");

  NdArrayRef y(x.eltype(), x.shape());
  const auto field = x.eltype().as<Ring2k>()->field();
  DISPATCH_ALL_FIELDS(field, [&]() {
    NdArrayView<ring2k_t> _x(x);
    NdArrayView<ring2k_t> _y(y);
    const auto pv_field = pv.eltype().as<Ring2k>()->field();
    DISPATCH_ALL_FIELDS(pv_field, [&]() {
      NdArrayView<ring2k_t> _pv(pv);
      for (int64_t i = 0; i < y.numel(); i++) {
        _y[_pv[i]] = _x[i];
      }
    });
  });
  return y;
}

NdArrayRef applyPerm(const NdArrayRef& x, absl::Span<const int64_t> pv) {
  SPU_ENFORCE_EQ(x.shape().ndim(), 1U, "x should be 1-d tensor");

  NdArrayRef y(x.eltype(), x.shape());
  const auto field = x.eltype().as<Ring2k>()->field();
  DISPATCH_ALL_FIELDS(field, [&]() {
    NdArrayView<ring2k_t> _x(x);
    NdArrayView<ring2k_t> _y(y);
    for (int64_t i = 0; i < y.numel(); i++) {
      _y[i] = _x[pv[i]];
    }
  });
  return y;
}

NdArrayRef applyPerm(const NdArrayRef& x, const NdArrayRef& pv) {
  SPU_ENFORCE_EQ(x.shape().ndim(), 1U, "x should be 1-d tensor");
  SPU_ENFORCE_EQ(x.shape(), pv.shape(), "x and pv should have same shape");

  NdArrayRef y(x.eltype(), x.shape());
  const auto field = x.eltype().as<Ring2k>()->field();
  DISPATCH_ALL_FIELDS(field, [&]() {
    NdArrayView<ring2k_t> _x(x);
    NdArrayView<ring2k_t> _y(y);
    const auto pv_field = pv.eltype().as<Ring2k>()->field();
    DISPATCH_ALL_FIELDS(pv_field, [&]() {
      NdArrayView<ring2k_t> _pv(pv);
      for (int64_t i = 0; i < y.numel(); i++) {
        _y[i] = _x[_pv[i]];
      }
    });
  });
  return y;
}

NdArrayRef genInversePerm(const NdArrayRef& perm) {
  NdArrayRef ret(perm.eltype(), perm.shape());
  auto field = perm.eltype().as<Ring2k>()->field();
  DISPATCH_ALL_FIELDS(field, [&]() {
    NdArrayView<ring2k_t> _ret(ret);
    NdArrayView<ring2k_t> _perm(perm);
    for (int64_t i = 0; i < perm.numel(); ++i) {
      _ret[_perm[i]] = ring2k_t(i);
    }
  });
  return ret;
}

Index genPermBySort(const NdArrayRef& x) {
  SPU_ENFORCE_EQ(x.shape().ndim(), 1U, "x should be 1-d tensor");
  Index perm(x.shape()[0]);
  std::iota(perm.begin(), perm.end(), 0);
  const auto field = x.eltype().as<Ring2k>()->field();
  DISPATCH_ALL_FIELDS(field, [&]() {
    using T = std::make_signed_t<ring2k_t>;

    NdArrayView<T> _x(x);
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