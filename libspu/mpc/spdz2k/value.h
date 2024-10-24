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

#include "libspu/core/ndarray_ref.h"

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
const NdArrayRef getValueShare(const NdArrayRef& in);

// Only output macs of valid bits for optimal memory usage
const NdArrayRef getMacShare(const NdArrayRef& in);

NdArrayRef makeAShare(const NdArrayRef& s1, const NdArrayRef& s2,
                      FieldType field, bool has_mac = true);

// Different with other protocls!
// input s1: value shares of valid bits
// input s2: mac shares of valid bits
// output: boolean shares of fixed length
NdArrayRef makeBShare(const NdArrayRef& s1, const NdArrayRef& s2,
                      FieldType field, int64_t nbits);

size_t maxNumBits(const NdArrayRef& lhs, const NdArrayRef& rhs);
size_t minNumBits(const NdArrayRef& lhs, const NdArrayRef& rhs);

// Convert a BShare in new_nbits
// then output the corresponding value and mac
std::pair<NdArrayRef, NdArrayRef> BShareSwitch2Nbits(const NdArrayRef& in,
                                                     int64_t new_nbits);

PtType calcBShareBacktype(size_t nbits);

NdArrayRef getShare(const NdArrayRef& in, int64_t share_idx);

#define PFOR_GRAIN_SIZE 8192

}  // namespace spu::mpc::spdz2k
