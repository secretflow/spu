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

#include "emp-tool/utils/block.h"
#include "emp-tool/utils/group.h"
#include "yasl/base/int128.h"  // For int 128 support

namespace spu {
using emp::BigInt;
using emp::block;
using emp::Group;
using emp::Point;

inline string encode_bigint(BigInt n) {
  std::stringstream stream;
  size_t len = n.size();
  stream.write(reinterpret_cast<char *>(&len), 4);
  unsigned char *tmp = new unsigned char[len];
  n.to_bin(tmp);
  stream.write(reinterpret_cast<char *>(tmp), len);
  return stream.str();
}

inline void decode_bigint(BigInt *n, const string *buf) {
  std::stringstream stream;
  stream << *buf;
  size_t len = 0;
  stream.read(reinterpret_cast<char *>(&len), 4);
  unsigned char *tmp = new unsigned char[len];
  stream.read(reinterpret_cast<char *>(tmp), len);
  n->from_bin(tmp, len);
}

inline string encode_pt(Point A) {
  std::stringstream stream;
  size_t len = A.size();
  A.group->resize_scratch(len);
  stream.write(reinterpret_cast<char *>(&len), 4);
  unsigned char *tmp = A.group->scratch;
  A.to_bin(tmp, len);
  stream.write(reinterpret_cast<char *>(tmp), len);
  return stream.str();
}

inline void decode_pt(Group *g, Point *A, const string *buf) {
  std::stringstream stream;
  stream << *buf;
  size_t len = 0;
  stream.read(reinterpret_cast<char *>(&len), 4);
  g->resize_scratch(len);
  unsigned char *tmp = g->scratch;
  stream.read(reinterpret_cast<char *>(tmp), len);
  A->from_bin(g, tmp, len);
}

inline void generate_key(void *buf) {
#ifndef ENABLE_RDSEED
  int *data = (int *)(buf);
  std::random_device rand_div;
  for (size_t i = 0; i < sizeof(block) / sizeof(int); ++i) data[i] = rand_div();
#else
  unsigned long long *data = (unsigned long long *)buf;
  _rdseed64_step(&data[0]);
  _rdseed64_step(&data[1]);
#endif
}

inline uint8_t bool_to_uint8(const uint8_t *data, size_t len = 0) {
  if (len != 0)
    len = (len > 8 ? 8 : len);
  else
    len = 8;
  uint8_t res = 0;
  for (size_t i = 0; i < len; ++i) {
    if (data[i]) res |= (1ULL << i);
  }
  return res;
}

inline void uint8_to_bool(uint8_t *data, uint8_t input, int length) {
  for (int i = 0; i < length; ++i) {
    data[i] = (input & 1) == 1;
    input >>= 1;
  }
}

inline int64_t bool_to_int64(const uint8_t *data, size_t len = 0) {
  if (len != 0)
    len = (len > 64 ? 64 : len);
  else
    len = 64;
  int64_t res = 0;
  for (size_t i = 0; i < len; ++i) {
    if (data[i]) res |= (1LL << i);
  }
  return res;
}

inline uint64_t bool_to_uint64(const uint8_t *data, size_t len = 0) {
  if (len != 0)
    len = (len > 64 ? 64 : len);
  else
    len = 64;
  uint64_t res = 0;
  for (size_t i = 0; i < len; ++i) {
    if (data[i]) res |= (1LL << i);
  }
  return res;
}

inline void from_block(uint8_t &dst, emp::block src) {
  dst = (uint8_t)_mm_extract_epi64(src, 0);
}

inline void from_block(uint32_t &dst, block src) {
  dst = (uint32_t)_mm_extract_epi64(src, 0);
}

inline void from_block(uint64_t &dst, block src) {
  dst = _mm_extract_epi64(src, 0);
}

inline void from_block(uint128_t &dst, block src) {
  dst = uint128_t(_mm_extract_epi64(src, 0)) |
        (uint128_t(_mm_extract_epi64(src, 1)) << 64);
}

// NOTE: This function doesn't return bitlen. It returns log_alpha
inline int bitlen(int x) {  // x & (-x)
  if (x < 1) return 0;
  for (int i = 0; i < 32; i++) {
    int curr = 1 << i;
    if (curr >= x) return i;
  }
  return 0;
}
}  // namespace spu
