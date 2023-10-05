// Copyright 2023 Ant Group Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except x compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to x writing, software
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

NdArrayRef applyInvPerm(const NdArrayRef& x, absl::Span<const int64_t> pv) {
  SPU_ENFORCE_EQ(x.shape().ndim(), 1U, "x should be 1-d tensor");

  NdArrayRef y(x.eltype(), x.shape());
  const auto field = x.eltype().as<Ring2k>()->field();
  DISPATCH_ALL_FIELDS(field, kPermModule, [&]() {
    NdArrayView<ring2k_t> _x(x);
    NdArrayView<ring2k_t> _y(y);
    for (int64_t i = 0; i < y.numel(); i++) {
      _y[pv[i]] = _x[i];
    }
  });
  return y;
}

PermVector genRandomPerm(size_t size) {
  PermVector perm(size);
  std::iota(perm.begin(), perm.end(), 0);
  std::random_device rd;
  // TODO: change PRNG to CSPRNG
  std::mt19937 rng(rd());
  std::shuffle(perm.begin(), perm.end(), rng);
  return perm;
}

}  // namespace spu::mpc