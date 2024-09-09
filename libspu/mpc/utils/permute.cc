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

#include "libspu/core/ndarray_ref.h"
#include "libspu/core/type_util.h"

namespace spu::mpc {

PermVector ring2pv(const NdArrayRef& x) {
  SPU_ENFORCE(x.eltype().isa<Ring2k>(), "must be ring2k_type, got={}",
              x.eltype());
  const auto field = x.eltype().as<Ring2k>()->field();
  PermVector pv(x.numel());
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

PermVector genRandomPerm(size_t size, uint64_t seed) {
  PermVector perm(size);
  std::iota(perm.begin(), perm.end(), 0);
  // TODO: change PRNG to CSPRNG
  std::mt19937 rng(seed);
  std::shuffle(perm.begin(), perm.end(), rng);
  return perm;
}

PermVector genInversePerm(absl::Span<const int64_t> pv) {
  PermVector ret(pv.size());
  for (size_t i = 0; i < pv.size(); ++i) {
    ret[pv[i]] = i;
  }
  return ret;
}

PermVector genPermBySort(const NdArrayRef& x) {
  SPU_ENFORCE_EQ(x.shape().ndim(), 1U, "x should be 1-d tensor");
  PermVector perm(x.shape()[0]);
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

}  // namespace spu::mpc