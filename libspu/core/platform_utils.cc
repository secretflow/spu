// Copyright 2022 Ant Group Co., Ltd.
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

#include "libspu/core/platform_utils.h"

#ifdef __x86_64__
#include <immintrin.h>

#include "cpu_features/cpuinfo_x86.h"
#endif

namespace spu {

namespace impl {

// These reference implementations are inspired by
// https://gcc.gnu.org/pipermail/gcc-patches/2017-June/475893.html

// Reference pdep_u64 impl
inline uint64_t
#if defined(__x86_64__) && defined(__GNUC__)
    __attribute__((target("arch=sandybridge")))
#endif
    pdep_u64_impl(uint64_t x, uint64_t m) {
  uint64_t result = 0x0UL;
  const uint64_t mask = 0x8000000000000000UL;
  uint64_t c;
  uint64_t t;
  uint64_t p;

  // The pop-count of the mask gives the number of the bits from
  // source to process.  This is also needed to shift bits from the
  // source into the correct position for the result.
  p = 64 - __builtin_popcountl(m);

  // The loop is for the number of '1' bits in the mask and clearing
  // each mask bit as it is processed.
  while (m != 0) {
    c = __builtin_clzl(m);
    t = x << (p - c);
    m ^= (mask >> c);
    result |= (t & (mask >> c));
    p++;
  }
  return (result);
}

// Reference pext_u64 impl
inline uint64_t
#if defined(__x86_64__) && defined(__GNUC__)
    __attribute__((target("arch=sandybridge")))
#endif
    pext_u64_impl(uint64_t x, uint64_t m) {
  // initial bit permute control
  uint64_t p = 0x4040404040404040UL;
  const uint64_t mask = 0x8000000000000000UL;
  uint64_t c;
  uint64_t result;

  p = 64 - __builtin_popcountl(m);
  result = 0;
  // We could a use a for loop here, but that combined with
  // -funroll-loops can expand to a lot of code.  The while
  // loop avoids unrolling and the compiler commons the xor
  // from clearing the mask bit with the (m != 0) test.  The
  // result is a more compact loop setup and body.
  while (m != 0) {
    uint64_t t;
    c = __builtin_clzl(m);
    t = (x & (mask >> c)) >> (p - c);
    m ^= (mask >> c);
    result |= (t);
    p++;
  }
  return (result);
}

}  // namespace impl

#ifdef __x86_64__
static const auto kCpuFeatures = cpu_features::GetX86Info().features;
static const bool kHasBMI2 = kCpuFeatures.bmi2;
static const bool kHasAVX512 = kCpuFeatures.avx512ifma;
static const bool kHasAVX2 = kCpuFeatures.avx2;
#else
static const bool kHasBMI2 = false;
static const bool kHasAVX512 = false;
static const bool kHasAVX2 = false;
#endif

bool hasAVX2() { return kHasAVX2; }
bool hasBMI2() { return kHasBMI2; }
bool hasAVX512ifma() { return kHasAVX512; }

// There are no bmi2 intrinsics on platforms other than x86, so directly
// redirect them to ref implementations
uint64_t pdep_u64(uint64_t a, uint64_t b) {
#ifdef __x86_64__
  if (hasBMI2()) {
    return _pdep_u64(a, b);
  }
#endif
  return impl::pdep_u64_impl(a, b);
}

uint64_t pext_u64(uint64_t a, uint64_t b) {
#ifdef __x86_64__
  if (hasBMI2()) {
    return _pext_u64(a, b);
  }
#endif
  return impl::pext_u64_impl(a, b);
}

}  // namespace spu
