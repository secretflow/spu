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

#include "libspu/mpc/utils/lowmc_utils.h"

#include "libspu/core/prelude.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc {

namespace {

template <typename T>
constexpr T bit_parity(const T x) {
  static_assert(std::is_unsigned_v<T>);

  auto k = sizeof(T) * 8;
  T ret = x;
  while (k > 1) {
    ret ^= (ret >> (k / 2));
    k /= 2;
  }

  return ret & 1;
}

}  // namespace

NdArrayRef dot_product_gf2(const NdArrayRef& x, const NdArrayRef& y,
                           FieldType to_field) {
  // conceptually, x is an n*k binary matrix, y is a m*k binary matrix (y can
  // be multi-dimension, we take 2-d as an example);
  // ret is a m*n binary matrix, ret[i] = dot(x, y[i]);
  // IMPORTANT: the field of (x,y) and ret may be different!
  SPU_ENFORCE(x.elsize() == y.elsize(), "size mismatch");
  SPU_ENFORCE(x.shape().size() == 1,
              "x should be a 1-D array, i.e. n*k binary matrix.");

  const auto field = x.eltype().as<RingTy>()->field();
  const auto n = x.shape().dim(0);
  SPU_ENFORCE(SizeOf(to_field) * 8 == (uint64_t)n,
              "mismatch of output bit size and type.");

  auto out = ring_zeros(to_field, y.shape());

  DISPATCH_ALL_FIELDS(field, [&]() {
    using src_type = ring2k_t;

    DISPATCH_ALL_FIELDS(to_field, [&]() {
      using to_type = ring2k_t;

      NdArrayView<to_type> _out(out);

      Index ind(1, 0);
      for (int64_t i = 0; i < n; ++i) {
        ind[0] = i;
        const auto row = x.slice_scalar_at(ind).broadcast_to(y.shape(), {});
        auto prod = ring_and(y, row);
        NdArrayView<src_type> _prod(prod);

        pforeach(0, out.numel(), [&](int64_t idx) {  //
          _out[idx] =
              _out[idx] | (static_cast<to_type>(bit_parity(_prod[idx])) << i);
        });
      }
    });
  });

  return out;
}

std::vector<NdArrayRef> generate_round_keys(
    const std::vector<NdArrayRef>& key_matrices, uint128_t key, uint64_t rounds,
    FieldType to_field) {
  SPU_ENFORCE(key_matrices.size() == (rounds + 1), "key matrix size mismatch");

  NdArrayRef master_key(makeType<RingTy>(FM128), {1});
  NdArrayView<uint128_t> _master_key(master_key);
  _master_key[0] = key;

  std::vector<NdArrayRef> round_keys;
  round_keys.reserve(rounds + 1);
  // round keys has rounds + 1 elements, the first one is for initial whiten
  for (uint64_t i = 0; i <= rounds; ++i) {
    round_keys.push_back(
        dot_product_gf2(key_matrices[i], master_key, to_field));
  }

  return round_keys;
}

int64_t get_data_complexity(int64_t n) {
  const auto n_bits = Log2Ceil(n);

  if (n_bits <= 20) {
    return 20;
  } else if (n_bits <= 30) {
    return 30;
  } else if (n_bits <= 40) {
    return 40;
  }

  SPU_THROW("Support at most 2^40 now.");
}

}  // namespace spu::mpc
