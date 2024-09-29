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

#include "xtensor/xrandom.hpp"

#include "libspu/core/context.h"
#include "libspu/core/memref.h"
#include "libspu/core/pt_buffer_view.h"
#include "libspu/kernel/hal/prot_wrapper.h"   // bad reference
#include "libspu/kernel/hal/public_helper.h"  // bad reference

namespace spu::kernel::test {

SPUContext makeSPUContext(RuntimeConfig config,
                          const std::shared_ptr<yacl::link::Context>& lctx);

SPUContext makeSPUContext(
    ProtocolKind prot_kind = ProtocolKind::REF2K, size_t field = 64,
    const std::shared_ptr<yacl::link::Context>& lctx = nullptr);

MemRef makeMemRef(SPUContext* ctx, PtBufferView init,
                  Visibility vtype = VIS_PUBLIC, const Shape& shape = {});

template <typename T>
auto xt_random(const std::vector<size_t>& shape, double min = -100,
               double max = 100) {
  if constexpr (std::is_integral_v<T>) {
    return xt::random::randint<T>(shape, static_cast<T>(min),
                                  static_cast<T>(max));
  } else if constexpr (std::is_floating_point_v<T>) {
    return xt::random::rand<T>(shape, static_cast<T>(min), static_cast<T>(max));
  } else {
    SPU_THROW("unsupported xt_random type");
  }
}

using UnaryOp = MemRef(SPUContext*, const MemRef&);
using BinaryOp = MemRef(SPUContext*, const MemRef&, const MemRef&);
using TernaryOp = MemRef(SPUContext*, const MemRef&, const MemRef&,
                         const MemRef&);

template <typename T>
xt::xarray<T> evalTernaryOp(Visibility in1_vtype, Visibility in2_vtype,
                            Visibility in3_vtype, TernaryOp* op,
                            PtBufferView in1, PtBufferView in2,
                            PtBufferView in3) {
  SPUContext ctx = makeSPUContext();

  MemRef a = makeMemRef(&ctx, in1, in1_vtype);
  MemRef b = makeMemRef(&ctx, in2, in2_vtype);
  MemRef c = makeMemRef(&ctx, in3, in3_vtype);

  MemRef d = op(&ctx, a, b, c);

  if (d.isSecret()) {
    d = hal::_s2p(&ctx, d);
  }
  SPU_ENFORCE(d.isPublic());

  return hal::dump_public_as<T>(&ctx, d, ctx.getFxpBits());
}

template <typename T>
xt::xarray<T> evalBinaryOp(Visibility lhs_vtype, Visibility rhs_vtype,
                           BinaryOp* op, PtBufferView lhs, PtBufferView rhs) {
  SPUContext ctx = makeSPUContext();

  MemRef a = makeMemRef(&ctx, lhs, lhs_vtype);
  MemRef b = makeMemRef(&ctx, rhs, rhs_vtype);

  MemRef c = op(&ctx, a, b);

  if (c.isSecret()) {
    c = hal::_s2p(&ctx, c);
  }
  SPU_ENFORCE(c.isPublic());

  return hal::dump_public_as<T>(&ctx, c, ctx.getFxpBits());
}

template <typename T>
xt::xarray<T> evalUnaryOp(Visibility in_vtype, UnaryOp* op, PtBufferView in) {
  SPUContext ctx = makeSPUContext();

  MemRef a = makeMemRef(&ctx, in, in_vtype);

  MemRef b = op(&ctx, a);

  if (b.isSecret()) {
    b = hal::_s2p(&ctx, b);
  }
  SPU_ENFORCE(b.isPublic());

  return hal::dump_public_as<T>(&ctx, b, ctx.getFxpBits());
}

}  // namespace spu::kernel::test
