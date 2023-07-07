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

#include "libspu/core/array_ref.h"
#include "libspu/core/type_util.h"
#include "libspu/mpc/spdz2k/beaver/beaver_tfp.h"
#include "libspu/mpc/spdz2k/beaver/beaver_tinyot.h"

namespace spu::mpc::spdz2k {

// The layout of Spdz2k share.
//
// Two shares are interleaved in a array, for example, given n element and k
// bytes per-element.
//
//   element          address
//   a[0].share0      0
//   a[0].share1      k
//   a[1].share0      2k
//   a[1].share1      3k
//   ...
//   a[n-1].share0    (n-1)*2*k+0
//   a[n-1].share1    (n-1)*2*k+k
//
// you can imagine spdz2k share as std::complex<T>, where
//   real(x) is the value share piece.
//   imag(x) is the mac share piece.

// Different with other protocls!
// Only output values of valid bits for optimal memory usage
const ArrayRef getValueShare(const ArrayRef& in);

// Only output macs of valid bits for optimal memory usage
const ArrayRef getMacShare(const ArrayRef& in);

ArrayRef makeAShare(const ArrayRef& s1, const ArrayRef& s2, FieldType field,
                    bool has_mac = true);

// Different with other protocls!
// input s1: value shares of valid bits
// input s2: mac shares of valid bits
// output: boolean shares of fixed length
ArrayRef makeBShare(const ArrayRef& s1, const ArrayRef& s2, FieldType field,
                    size_t nbits);

size_t maxNumBits(const ArrayRef& lhs, const ArrayRef& rhs);
size_t minNumBits(const ArrayRef& lhs, const ArrayRef& rhs);

size_t minNumBits(const ArrayRef& lhs, const ArrayRef& rhs);

// Convert a BShare in new_nbits
// then output the corresponding value and mac
std::pair<ArrayRef, ArrayRef> BShareSwitch2Nbits(const ArrayRef& in,
                                                 size_t new_nbits);

PtType calcBShareBacktype(size_t nbits);

template <typename T>
size_t maxBitWidth(ArrayView<T> av) {
  // TODO: use av.maxBitWidth to improve performance
  return sizeof(T) * 8;
}

ArrayRef getShare(const ArrayRef& in, int64_t share_idx);

#define PFOR_GRAIN_SIZE 8192

}  // namespace spu::mpc::spdz2k
