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

#include "absl/types/span.h"
#include "yacl/base/int128.h"

#include "libspu/core/memref.h"
#include "libspu/core/prelude.h"
#include "libspu/mpc/common/communicator.h"

namespace spu::mpc::cheetah {

template <typename T>
T makeBitsMask(size_t nbits) {
  size_t max = sizeof(T) * 8;
  if (nbits == 0) {
    nbits = max;
  }
  SPU_ENFORCE(nbits <= max);
  T mask = static_cast<T>(-1);
  if (nbits < max) {
    mask = (static_cast<T>(1) << nbits) - 1;
  }
  return mask;
}

template <typename T>
inline T CeilDiv(T a, T b) {
  return (a + b - 1) / b;
}

template <typename T>
size_t ZipArray(absl::Span<const T> inp, size_t bit_width, absl::Span<T> oup) {
  size_t width = sizeof(T) * 8;
  SPU_ENFORCE(bit_width > 0 && width >= bit_width);
  size_t numel = inp.size();
  size_t packed_sze = CeilDiv(numel * bit_width, width);

  SPU_ENFORCE(oup.size() >= packed_sze);

  const T mask = makeBitsMask<T>(bit_width);
  for (size_t i = 0; i < packed_sze; ++i) {
    oup[i] = 0;
  }
  // shift will in [0, 2 * width]
  for (size_t i = 0, has_done = 0; i < numel; i += 1, has_done += bit_width) {
    T real_data = inp[i] & mask;
    size_t packed_index0 = i * bit_width / width;
    size_t shft0 = has_done % width;
    oup[packed_index0] |= (real_data << shft0);
    if (shft0 + bit_width > width) {
      size_t shft1 = width - shft0;
      size_t packed_index1 = packed_index0 + 1;
      oup[packed_index1] |= (real_data >> shft1);
    }
  }
  return packed_sze;
}

template <typename T>
size_t UnzipArray(absl::Span<const T> inp, size_t bit_width,
                  absl::Span<T> oup) {
  size_t width = sizeof(T) * 8;
  SPU_ENFORCE(bit_width > 0 && bit_width <= width);

  size_t packed_sze = inp.size();
  size_t n = oup.size();
  size_t raw_sze = packed_sze * width / bit_width;
  SPU_ENFORCE(n > 0 && n <= raw_sze);

  const T mask = makeBitsMask<T>(bit_width);
  for (size_t i = 0, has_done = 0; i < n; i += 1, has_done += bit_width) {
    size_t packed_index0 = i * bit_width / width;
    size_t shft0 = has_done % width;
    oup[i] = (inp[packed_index0] >> shft0);
    if (shft0 + bit_width > width) {
      size_t shft1 = width - shft0;
      size_t packed_index1 = packed_index0 + 1;
      oup[i] |= (inp[packed_index1] << shft1);
    }
    oup[i] &= mask;
  }
  return n;
}

template <typename T>
size_t ZipArrayBit(absl::Span<const T> inp, size_t bit_width,
                   absl::Span<T> oup) {
  return ZipArray<T>(inp, bit_width, oup);
}

template <typename T>
size_t UnzipArrayBit(absl::Span<const T> inp, size_t bit_width,
                     absl::Span<T> oup) {
  return UnzipArray<T>(inp, bit_width, oup);
}

template <typename T>
size_t PackU8Array(absl::Span<const uint8_t> u8array, absl::Span<T> packed) {
  constexpr size_t elsze = sizeof(T);
  const size_t nbytes = u8array.size();
  const size_t numel = CeilDiv(nbytes, elsze);

  SPU_ENFORCE(packed.size() >= numel);

  for (size_t i = 0; i < nbytes; i += elsze) {
    size_t this_batch = std::min(nbytes - i, elsze);
    T acc{0};
    for (size_t j = 0; j < this_batch; ++j) {
      acc = (acc << 8) | u8array[i + j];
    }
    packed[i / elsze] = acc;
  }

  return numel;
}

MemRef OpenShare(const MemRef &shr, ReduceOp op, size_t nbits,
                 std::shared_ptr<Communicator> conn);

uint8_t BoolToU8(absl::Span<const uint8_t> bits);

void U8ToBool(absl::Span<uint8_t> bits, uint8_t u8);

// Taken from emp-tool
// https://github.com/emp-toolkit/emp-tool/blob/master/emp-tool/utils/block.h#L113
void SseTranspose(uint8_t *out, uint8_t const *inp, uint64_t nrows,
                  uint64_t ncols);

}  // namespace spu::mpc::cheetah
