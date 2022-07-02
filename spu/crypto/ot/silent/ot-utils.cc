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

#include "ot-utils.h"

namespace spu {

template <typename basetype>
void pack_ot_messages(basetype *y, const basetype *const *data,
                      const emp::block *pad, size_t ysize, size_t bsize,
                      size_t bitsize, size_t N) {
  assert(y != nullptr && data != nullptr && pad != nullptr);
  size_t start_pos = 0;
  size_t end_pos = 0;
  size_t start_block = 0;
  size_t end_block = 0;
  basetype temp_bl = 0;
  basetype mask = ((basetype)1 << bitsize) - 1;
  if (8 * sizeof(basetype) == 64) {
    mask = (basetype)((1ULL << bitsize) - 1ULL);
  }
  if (8 * sizeof(basetype) == bitsize) {
    mask = (basetype)(-1);
  }
  size_t carriersize = 8 * (sizeof(basetype));
  for (size_t i = 0; i < ysize; i++) {
    y[i] = 0;
  }

  for (size_t i = 0; i < bsize; i++) {
    for (size_t k = 0; k < N; k++) {
      // OT message k
      start_pos = i * N * bitsize + k * bitsize;  // inclusive
      end_pos = start_pos + bitsize;
      end_pos -= 1;  // inclusive
      start_block = start_pos / carriersize;
      end_block = end_pos / carriersize;
      if (carriersize == 64) {
        if (start_block == end_block) {
          y[start_block] ^=
              ((((basetype)_mm_extract_epi64(pad[(N * i) + k], 0)) ^
                data[i][k]) &
               mask)
              << (start_pos % carriersize);

        } else {
          temp_bl = ((((basetype)_mm_extract_epi64(pad[(N * i) + k], 0)) ^
                      data[i][k]) &
                     mask);
          y[start_block] ^= (temp_bl) << (start_pos % carriersize);
          y[end_block] ^=
              (start_pos == 0
                   ? 0
                   : (temp_bl) >> (carriersize - (start_pos % carriersize)));
        }
      } else if (carriersize == 8) {
        if (start_block == end_block) {
          y[start_block] ^=
              ((((basetype)_mm_extract_epi8(pad[(N * i) + k], 0)) ^
                data[i][k]) &
               mask)
              << (start_pos % carriersize);
        } else {
          temp_bl = ((((basetype)_mm_extract_epi8(pad[(N * i) + k], 0)) ^
                      data[i][k]) &
                     mask);
          y[start_block] ^= (temp_bl) << (start_pos % carriersize);
          y[end_block] ^=
              (temp_bl) >> (carriersize - (start_pos % carriersize));
        }
      } else {
        throw std::invalid_argument("Not implemented");
      }
    }
  }
}

template <typename basetype>
void unpack_ot_messages(basetype *data, const uint8_t *r, const basetype *recvd,
                        const emp::block *pad, size_t bsize, size_t bitsize,
                        size_t N) {
  assert(data != nullptr && recvd != nullptr && pad != nullptr);
  size_t start_pos = 0;
  size_t end_pos = 0;
  size_t start_block = 0;
  size_t end_block = 0;
  basetype mask = ((basetype)1 << bitsize) - 1;
  if (8 * sizeof(basetype) == 64) {
    mask = (basetype)((1ULL << bitsize) - 1ULL);
  }
  if (8 * sizeof(basetype) == bitsize) {
    mask = static_cast<basetype>(-1);
  }
  size_t carriersize = 8 * (sizeof(basetype));

  for (size_t i = 0; i < bsize; i++) {
    start_pos = i * N * bitsize + static_cast<size_t>(r[i]) * bitsize;
    end_pos = start_pos + bitsize - 1;  // inclusive
    start_block = start_pos / carriersize;
    end_block = end_pos / carriersize;
    if (carriersize == 64) {
      if (start_block == end_block) {
        data[i] = ((recvd[start_block] >> (start_pos % carriersize)) ^
                   ((basetype)_mm_extract_epi64(pad[i], 0))) &
                  mask;
      } else {
        data[i] = 0;
        data[i] ^= (recvd[start_block] >> (start_pos % carriersize));
        data[i] ^=
            (recvd[end_block] << (carriersize - (start_pos % carriersize)));
        data[i] = (data[i] ^ ((basetype)_mm_extract_epi64(pad[i], 0))) & mask;
      }
    } else if (carriersize == 8) {
      if (start_block == end_block) {
        data[i] = ((recvd[start_block] >> (start_pos % carriersize)) ^
                   ((basetype)_mm_extract_epi8(pad[i], 0))) &
                  mask;
      } else {
        data[i] = 0;
        data[i] ^= (recvd[start_block] >> (start_pos % carriersize));
        data[i] ^=
            (recvd[end_block] << (carriersize - (start_pos % carriersize)));
        data[i] = (data[i] ^ ((basetype)_mm_extract_epi8(pad[i], 0))) & mask;
      }
    } else {
      throw std::invalid_argument("Not implemented");
    }
  }
}

void pack_cot_messages(uint64_t *y, const uint64_t *corr_data, size_t ysize,
                       size_t bsize, size_t bitsize) {
  assert(y != nullptr && corr_data != nullptr);
  size_t start_pos = 0;
  size_t end_pos = 0;
  size_t start_block = 0;
  size_t end_block = 0;
  uint64_t temp_bl = 0;
  uint64_t mask = (1ULL << bitsize) - 1;
  if (bitsize == 64) mask = -1;

  size_t carriersize = 64;
  for (size_t i = 0; i < ysize; i++) {
    y[i] = 0;
  }
  for (size_t i = 0; i < bsize; i++) {
    start_pos = i * bitsize;  // inclusive
    end_pos = start_pos + bitsize;
    end_pos -= 1;  // inclusive
    start_block = start_pos / carriersize;
    end_block = end_pos / carriersize;
    if (carriersize == 64) {
      if (start_block == end_block) {
        y[start_block] ^= (corr_data[i] & mask) << (start_pos % carriersize);
      } else {
        temp_bl = (corr_data[i] & mask);
        y[start_block] ^= (temp_bl) << (start_pos % carriersize);
        y[end_block] ^=
            (start_pos == 0
                 ? 0
                 : (temp_bl) >> (carriersize - (start_pos % carriersize)));
      }
    }
  }
}

void unpack_cot_messages(uint64_t *corr_data, const uint64_t *recvd,
                         size_t bsize, size_t bitsize) {
  assert(corr_data != nullptr && recvd != nullptr);
  size_t start_pos = 0;
  size_t end_pos = 0;
  size_t start_block = 0;
  size_t end_block = 0;
  uint64_t mask = (1ULL << bitsize) - 1;
  if (bitsize == 64) mask = -1;
  size_t carriersize = 64;

  for (size_t i = 0; i < bsize; i++) {
    start_pos = i * bitsize;
    end_pos = start_pos + bitsize - 1;  // inclusive
    start_block = start_pos / carriersize;
    end_block = end_pos / carriersize;
    if (carriersize == 64) {
      if (start_block == end_block) {
        corr_data[i] = (recvd[start_block] >> (start_pos % carriersize)) & mask;
      } else {
        corr_data[i] = 0;
        corr_data[i] ^= (recvd[start_block] >> (start_pos % carriersize));
        corr_data[i] ^=
            (start_pos == 0 ? 0
                            : (recvd[end_block]
                               << (carriersize - (start_pos % carriersize))));
      }
    }
  }
}

template void pack_ot_messages<uint64_t>(uint64_t *y,
                                         const uint64_t *const *data,
                                         const emp::block *pad, size_t ysize,
                                         size_t bsize, size_t bitsize,
                                         size_t N);
template void pack_ot_messages<uint8_t>(uint8_t *y, const uint8_t *const *data,
                                        const emp::block *pad, size_t ysize,
                                        size_t bsize, size_t bitsize, size_t N);

template void unpack_ot_messages<uint64_t>(uint64_t *data, const uint8_t *r,
                                           const uint64_t *recvd,
                                           const emp::block *pad, size_t bsize,
                                           size_t bitsize, size_t N);
template void unpack_ot_messages<uint8_t>(uint8_t *data, const uint8_t *r,
                                          const uint8_t *recvd,
                                          const emp::block *pad, size_t bsize,
                                          size_t bitsize, size_t N);

}  // namespace spu
