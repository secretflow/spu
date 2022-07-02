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

#include "spu/core/profile.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace spu {
namespace {

#define SPU_TRACE_OP(...)                                                     \
  __MACRO_SELECT_WITH_CTX(__VA_ARGS__, __TRACE_OP4, __TRACE_OP3, __TRACE_OP2, \
                          __TRACE_OP1)                                        \
  ("op", __func__, __VA_ARGS__)

class H {
 public:
  void proc(ProfilingContext* ctx, int a) { SPU_TRACE_OP(ctx, a); }
};

void g(ProfilingContext* ctx, int a) {
  SPU_TRACE_OP(ctx, a);
  H h;
  h.proc(ctx, a);
}

void f(ProfilingContext* ctx, int a, int b) {
  SPU_TRACE_OP(ctx, a, b);
  g(ctx, a);
  g(ctx, b);
}

}  // namespace

TEST(ProfilingContextTest, Works) {
  ProfilingContext pctx;
  pctx.setTracingEnabled(true);
  f(&pctx, 1, 2);
}

}  // namespace spu
