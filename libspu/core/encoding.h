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

#include "libspu/core/memref.h"
#include "libspu/core/pt_buffer_view.h"

namespace spu {

// Encode plaintext array to ring array
//
// Given:
//   src: source plaintext array
//   out: target ring array
//   fxp_bits: number of fractional bits for fixed point.
//
// Then:
//   let scale = 1 << fxp_bits
//   in
//   y = cast<ring_type>(x * scale)   if type(x) is float
//    |= cast<ring_type>(x)           if type(x) is integer
void encodeToRing(const PtBufferView& src, MemRef& out, int64_t fxp_bits = 0);

// Decode ring to plaintext array
//
// Given:
//   src: source ring array
//   out: target plaintext array
//   fxp_bits: number of fractional bits for fixed point.
//
// Then:
//   let scale = 1 << fxp_bits
//   in
//   y = cast<PtType>(x) / scale   if dtype is FXP
//    |= cast<PtType>(x)           if dtype is INT
void decodeFromRing(const MemRef& src, PtBufferView& out, int64_t fxp_bits = 0);

}  // namespace spu
