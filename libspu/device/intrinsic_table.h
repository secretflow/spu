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

#include "llvm/ADT/StringRef.h"

// clang-format off

//         MACRO_NAME       FCN_NAME
#define    IPOW             "@spu.ipower"
#define    TOPK             "@spu.topk"
#define    PREFER_A         "@spu.prefer_a"
#define    DBG_PRINT        "@spu.dbg_print"
#define    ROUND_NE         "@spu.round_nearest_even"
#define    SIMPLE_SORT      "@spu.simple_sort"
#define    GENERIC_SORT     "@spu.generic_sort"

// should be consistent with python level
#define    MAKE_CACHED_VAR  "@spu.make_cached_var"
#define    DROP_CACHED_VAR  "@spu.drop_cached_var"
#define    TRY_REVEAL_COND  "@spu.try_reveal_cond"

// clang-format on

namespace spu::device {

inline llvm::StringRef demangle_fcn_name(llvm::StringRef name) {
  auto end = name.find_first_of('#');
  return name.substr(0, end);
}

}  // namespace spu::device
