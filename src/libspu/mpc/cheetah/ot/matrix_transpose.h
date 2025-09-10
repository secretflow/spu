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

#ifndef __aarch64__
// sse
#include <emmintrin.h>
#include <smmintrin.h>
// pclmul
#include <wmmintrin.h>
#else
#include "sse2neon.h"
#endif

#include "yacl/base/int128.h"

#include "libspu/core/prelude.h"

namespace spu::mpc::cheetah {
template <typename T>
struct is_uint : std::false_type {};

// FIXME: uint8_t and uint16_t is not correct now...
template <>
struct is_uint<uint8_t> : std::true_type {};

template <>
struct is_uint<uint16_t> : std::true_type {};

template <>
struct is_uint<uint32_t> : std::true_type {};

template <>
struct is_uint<uint64_t> : std::true_type {};

template <>
struct is_uint<uint128_t> : std::true_type {};

template <typename T>
constexpr bool is_uint_v = is_uint<T>::value;

template <typename T, std::enable_if_t<is_uint_v<T>, bool> = true>
void naive_transpose(const T *src, T *dest, uint64_t nrows, uint64_t ncols) {
  for (size_t i = 0; i < nrows; ++i) {
    for (size_t j = 0; j < ncols; ++j) {
      dest[j * nrows + i] = src[i * ncols + j];
    }
  }
}

#define SRC(x, y) src[(x) * ncols + (y)]
#define DEST(x, y) dest[(x) * nrows + (y)]

// some constrains to ncols, if not meet, fallback to cache_friendly_transpose
template <typename T, std::enable_if_t<is_uint_v<T>, bool> = true>
void cache_friendly_transpose(const T *src, T *dest, uint64_t nrows,
                              uint64_t ncols) {
  std::size_t const max_i = (nrows / 4) * 4;
  std::size_t const max_j = (ncols / 4) * 4;

  for (std::size_t i = 0; i != max_i; i += 4) {
    // 4*4 block
    for (std::size_t j = 0; j != max_j; j += 4) {
      for (std::size_t k = 0; k != 4; ++k) {
        for (std::size_t l = 0; l != 4; ++l) {
          DEST(j + l, i + k) = SRC(i + k, j + l);
        }
      }
    }
    // the rest cols
    for (std::size_t k = 0; k != 4; ++k) {
      for (std::size_t j = max_j; j < ncols; ++j) {
        DEST(j, i + k) = SRC(i + k, j);
      }
    }
  }

  // the rest rows
  for (std::size_t i = max_i; i != nrows; ++i) {
    for (std::size_t j = 0; j != ncols; ++j) {
      DEST(j, i) = SRC(i, j);
    }
  }
}

#ifdef __x86_64__
__attribute__((target("sse2")))
#endif
void sse_1pack_transpose(const uint128_t *src, uint128_t *dest, uint64_t nrows,
                         uint64_t ncols) {
  for (size_t i = 0; i < nrows; ++i) {
    for (size_t j = 0; j < ncols; ++j) {
      _mm_store_ps(reinterpret_cast<float *>(&DEST(j, i)),
                   _mm_load_ps(reinterpret_cast<float const *>(&SRC(i, j))));
    }
  }
}

#ifdef __x86_64__
__attribute__((target("sse2")))
#endif
void sse_2pack_transpose(const uint64_t *src, uint64_t *dest, uint64_t nrows,
                         uint64_t ncols) {
  __m128d r0, r1;

  std::size_t const max_i = (nrows / 2) * 2;
  std::size_t const max_j = (ncols / 2) * 2;

  if (max_j != ncols)
    for (std::size_t i = 0; i < max_i; i += 2) {
      for (std::size_t j = 0; j < max_j; j += 2) {
        r0 = _mm_load_pd(reinterpret_cast<double const *>(&SRC(i, j)));
        r1 = _mm_load_pd(reinterpret_cast<double const *>(&SRC(i + 1, j)));

        _mm_store_pd(reinterpret_cast<double *>(&DEST(j, i)),
                     _mm_shuffle_pd(r0, r1, 0b00));

        _mm_store_pd(reinterpret_cast<double *>(&DEST(j + 1, i)),
                     _mm_shuffle_pd(r0, r1, 0b11));
      }

      for (std::size_t k = 0; k < 2; ++k)
        for (std::size_t j = max_j; j < ncols; ++j)
          DEST(j, i + k) = SRC(i + k, j);
    }
  else
    for (std::size_t i = 0; i < max_i; i += 2) {
      for (std::size_t j = 0; j < max_j; j += 2) {
        r0 = _mm_load_pd(reinterpret_cast<double const *>(&SRC(i, j)));
        r1 = _mm_load_pd(reinterpret_cast<double const *>(&SRC(i + 1, j)));

        _mm_store_pd(reinterpret_cast<double *>(&DEST(j, i)),
                     _mm_shuffle_pd(r0, r1, 0b00));

        _mm_store_pd(reinterpret_cast<double *>(&DEST(j + 1, i)),
                     _mm_shuffle_pd(r0, r1, 0b11));
      }
    }

  if (max_i != nrows)
    for (std::size_t j = 0; j < ncols; ++j) DEST(j, max_i) = SRC(max_i, j);
}

#ifdef __x86_64__
__attribute__((target("sse2")))
#endif
void sse_4pack_transpose(const uint32_t *src, uint32_t *dest, uint64_t nrows,
                         uint64_t ncols) {

  std::size_t const max_i = (nrows / 4) * 4;
  std::size_t const max_j = (ncols / 4) * 4;

  __m128 r0, r1, r2, r3;

  for (std::size_t i = 0; i != max_i; i += 4) {
    for (std::size_t j = 0; j != max_j; j += 4) {
      r0 = _mm_load_ps(reinterpret_cast<float const *>(&SRC(i, j)));
      r1 = _mm_load_ps(reinterpret_cast<float const *>(&SRC(i + 1, j)));
      r2 = _mm_load_ps(reinterpret_cast<float const *>(&SRC(i + 2, j)));
      r3 = _mm_load_ps(reinterpret_cast<float const *>(&SRC(i + 3, j)));

      _MM_TRANSPOSE4_PS(r0, r1, r2, r3);

      _mm_store_ps(reinterpret_cast<float *>(&DEST(j, i)), r0);
      _mm_store_ps(reinterpret_cast<float *>(&DEST(j + 1, i)), r1);
      _mm_store_ps(reinterpret_cast<float *>(&DEST(j + 2, i)), r2);
      _mm_store_ps(reinterpret_cast<float *>(&DEST(j + 3, i)), r3);
    }

    for (std::size_t k = 0; k != 4; ++k)
      for (std::size_t j = max_j; j != ncols; ++j)
        DEST(j, i + k) = SRC(i + k, j);
  }

  for (std::size_t i = max_i; i != nrows; ++i)
    for (std::size_t j = 0; j != ncols; ++j) DEST(j, i) = SRC(i, j);
}

// adapted from sse-matrix-transpose
// the inp and oup must be 16 bytes aligned, otherwise, fallback to
// cache-friendly transpose
template <typename T, std::enable_if_t<is_uint_v<T>, bool> = true>
void sse_transpose(const T *src, T *dest, uint64_t nrows, uint64_t ncols) {
  constexpr std::size_t static alignment = 16;
  constexpr std::size_t data_size = sizeof(T);

  constexpr auto ratio = alignment / data_size;

  if ((ncols % ratio != 0) || (nrows % ratio != 0)) {
    // fallback to cache-friendly transpose
    return cache_friendly_transpose(src, dest, nrows, ncols);
  }

  if constexpr (4 == data_size) {
    sse_4pack_transpose(src, dest, nrows, ncols);
  } else if constexpr (8 == data_size) {
    sse_2pack_transpose(src, dest, nrows, ncols);
  } else if constexpr (16 == data_size) {
    // indeed, this is even slower than cache-friendly transpose
    sse_1pack_transpose(src, dest, nrows, ncols);
  } else {
    SPU_THROW(
        "invalid data type for sse transpose, only 32,64,128 bits "
        "supported.");
  }
}
}  // namespace spu::mpc::cheetah