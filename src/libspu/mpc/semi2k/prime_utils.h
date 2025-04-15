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

#include "libspu/core/context.h"
#include "libspu/mpc/kernel.h"

namespace spu::mpc::semi2k {
// Ring2k share -> Mersenne Prime - 1 share
// Given x0 + x1 = x mod 2^k
// Compute h0 + h1 = x mod p with probability > 1 - |x|/2^k
NdArrayRef ProbConvRing2k(const NdArrayRef& inp_share, int rank,
                          size_t shr_width);

// Mul open private share
std::tuple<NdArrayRef, NdArrayRef> MulPrivPrep(KernelEvalContext* ctx,
                                               const NdArrayRef& x);

// Note that [x] = (x_alice, x_bob) and x_alice + x_bob = x
// Note that we actually want to find the muliplication of x_alice and x_bob
// this function is currently achieved by doing (x_alice, 0) * (0, x_bob)
// optimization is possible.
NdArrayRef MulPrivModMP(KernelEvalContext* ctx, const NdArrayRef& x);
// We assume the input is ``positive''
// Given h0 + h1 = h mod p and h < p / 2
// Define b0 = 1{h0 >= p/2}
//        b1 = 1{h1 >= p/2}
// Compute w = 1{h0 + h1 >= p}
// It can be proved that w = (b0 or b1)
NdArrayRef WrapBitModMP(KernelEvalContext* ctx, const NdArrayRef& x);

// Mersenne Prime share -> Ring2k share
NdArrayRef ConvMP(KernelEvalContext* ctx, const NdArrayRef& h,
                  uint truncate_nbits);
}  // namespace spu::mpc::semi2k