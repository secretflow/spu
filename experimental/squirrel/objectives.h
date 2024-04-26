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
#include "libspu/core/value.h"

namespace squirrel {
// Approximated 1/sqrt(x)
// REQUIRE: 2^{-fxp} <= x < 2^{2*fxp} and 3*fxp + 2 < k for the 2^k ring.
// NOTE(lwj): This function can handle a larger range than hlo::Rsqrt which
// requires 2^{-fxp} <= x < 2^{fxp}
spu::Value Rsqrt(spu::SPUContext* ctx, const spu::Value& x, int iterations = 1);

// High-precision piece-wise logistic
// logistic(x) = { epsilon if x < -7.0
//               { 0.5 - P^3(|x|) if x \in [-7.0, 0)
//               { 0.5 + P^3(|x|) if x \in [0.0, 7.0)
//               { 1 - epsilon  if x > 7.0
// where P^3(*) is a degree-3 polynomial
spu::Value Logistic(spu::SPUContext* ctx, const spu::Value& x);

// Even higher precision numerical sigmoid
// sigmoid(x) = 0.5 + 0.5*x * rsqrt(1 + x^2)
// NOTE: no `clamp` is performed, so pay attention to overflow from the square
// term.
spu::Value Sigmoid(spu::SPUContext* ctx, const spu::Value& x);

// Gs in shape (n, B*m)
// Hs in shape (n, B*m)
// `n` is the number of nodes in the current level
// `m` is the number of features and `B` is the bin size
// The last bin Gs[:, -1] and Hs[:, -1] is the sum of all bins
//
// GA = Gs[:, -1]
// GL = Gs
// GR = GA - GL
//
// obj = |GA| * rsqrt(H + lambda)
// obj_L = |GL| * rsqrt(HL + lambda)
// obj_R = |GR| * rsqrt(HR + lambda)
// gain = obj_L + obj_R - obj
//
// return ArgMax(gain, axis=1)
spu::Value MaxGainOnLevel(spu::SPUContext* ctx, const spu::Value& Gs,
                          const spu::Value& Hs, double reg_lambda);

}  // namespace squirrel
