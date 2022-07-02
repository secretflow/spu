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

#include "spu/core/array_ref.h"
#include "spu/core/type_util.h"

namespace spu::mpc::aby3 {

// The layout of Aby3 share.
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
// you can treat aby3 share as std::complex<T>, where
//   real(x) is the first share piece.
//   imag(x) is the second share piece.

ArrayRef getFirstShare(const ArrayRef& in);

ArrayRef getSecondShare(const ArrayRef& in);

ArrayRef makeAShare(const ArrayRef& s1, const ArrayRef& s2, FieldType field);

ArrayRef makeBShare(const ArrayRef& s1, const ArrayRef& s2, FieldType field,
                    size_t nbits = std::numeric_limits<size_t>::max());

}  // namespace spu::mpc::aby3
