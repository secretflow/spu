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
#include "libspu/core/shape_util.h"
#include "libspu/core/value.h"
#include "libspu/core/xt_helper.h"
#include "libspu/kernel/hal/prot_wrapper.h"   // bad reference
#include "libspu/kernel/hal/public_helper.h"  // bad reference

namespace spu::kernel::test {

SPUContext makeSPUContext(RuntimeConfig config,
                          const std::shared_ptr<yacl::link::Context>& lctx);

SPUContext makeSPUContext(
    ProtocolKind prot_kind = ProtocolKind::REF2K,
    FieldType field = FieldType::FM64,
    const std::shared_ptr<yacl::link::Context>& lctx = nullptr);

Value makeValue(SPUContext* ctx, PtBufferView init,
                Visibility vtype = VIS_PUBLIC, DataType dtype = DT_INVALID,
                ShapeView shape = {});

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

using UnaryOp = Value(SPUContext*, const Value&);
using BinaryOp = Value(SPUContext*, const Value&, const Value&);
using TernaryOp = Value(SPUContext*, const Value&, const Value&, const Value&);

template <typename T>
xt::xarray<T> evalTernaryOp(Visibility in1_vtype, Visibility in2_vtype,
                            Visibility in3_vtype, TernaryOp* op,
                            PtBufferView in1, PtBufferView in2,
                            PtBufferView in3) {
  SPUContext ctx = makeSPUContext();

  Value a = makeValue(&ctx, in1, in1_vtype);
  Value b = makeValue(&ctx, in2, in2_vtype);
  Value c = makeValue(&ctx, in3, in3_vtype);

  Value d = op(&ctx, a, b, c);

  if (d.isSecret()) {
    d = hal::_s2p(&ctx, d);
  }
  SPU_ENFORCE(d.isPublic());

  return hal::dump_public_as<T>(&ctx, d);
}

template <typename T>
xt::xarray<T> evalBinaryOp(Visibility lhs_vtype, Visibility rhs_vtype,
                           BinaryOp* op, PtBufferView lhs, PtBufferView rhs) {
  SPUContext ctx = makeSPUContext();

  Value a = makeValue(&ctx, lhs, lhs_vtype);
  Value b = makeValue(&ctx, rhs, rhs_vtype);

  Value c = op(&ctx, a, b);

  if (c.isSecret()) {
    c = hal::_s2p(&ctx, c).setDtype(c.dtype());
  }
  SPU_ENFORCE(c.isPublic());

  return hal::dump_public_as<T>(&ctx, c);
}

template <typename T>
xt::xarray<T> evalUnaryOp(Visibility in_vtype, UnaryOp* op, PtBufferView in) {
  SPUContext ctx = makeSPUContext();

  Value a = makeValue(&ctx, in, in_vtype);

  Value b = op(&ctx, a);

  if (b.isSecret()) {
    b = hal::_s2p(&ctx, b).setDtype(b.dtype());
  }
  SPU_ENFORCE(b.isPublic());

  return hal::dump_public_as<T>(&ctx, b);
}

}  // namespace spu::kernel::test
