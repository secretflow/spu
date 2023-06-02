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

#include "libspu/kernel/hal/constants.h"

#include "libspu/core/encoding.h"
#include "libspu/core/ndarray_ref.h"
#include "libspu/core/pt_buffer_view.h"
#include "libspu/core/shape_util.h"
#include "libspu/core/type_util.h"
#include "libspu/kernel/hal/prot_wrapper.h"
#include "libspu/kernel/hal/ring.h"
#include "libspu/mpc/common/pv2k.h"

namespace spu::kernel::hal {
namespace {

// make a public typed value.
//
// FIXME: this is a abstraction leakage, we should NOT invoke Pub2kTy directly.
Value make_pub2k(SPUContext* ctx, const PtBufferView& bv) {
  SPU_TRACE_HAL_DISP(ctx, bv);

  NdArrayRef raw = convertToNdArray(bv);

  const auto field = ctx->getField();
  const auto fxp_bits = ctx->getFxpBits();

  DataType dtype;
  NdArrayRef encoded = encodeToRing(raw, field, fxp_bits, &dtype);

  return Value(encoded.as(makeType<mpc::Pub2kTy>(field)), dtype);
}

// TODO: formalize and test it.
// clang-format off
// NOLINTBEGIN, readability-implicit-bool-conversion, modernize-use-bool-literals
bool kCastFlags[DT_F64+1][DT_F64+1] = {
//{_,  I1, I8, U8, I16,U16,I32,U32,I64,U64,F32,F64}
  {0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0  , 0},  // _
  {0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0  , 0},  // I1
  {0,  0,  1,  0,  1,  0,  1,  0,  1,  0,  0  , 0},  // I8
  {0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  0  , 0},  // U8
  {0,  0,  0,  0,  1,  0,  1,  0,  1,  0,  0  , 0},  // I16
  {0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  0  , 0},  // U16
  {0,  0,  0,  0,  0,  0,  1,  0,  1,  0,  0  , 0},  // I32
  {0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  0  , 0},  // U32
  {0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0  , 0},  // I64
  {0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  0  , 0},  // U64
  {0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1  , 1},  // F32
  {0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0  , 1},  // F64
};
// NOLINTEND
// clang-format on

bool canImplicitCastTo(DataType frm, DataType to) {
  return kCastFlags[frm][to];
}

}  // namespace

Value constant(SPUContext* ctx, PtBufferView init, DataType dtype,
               ShapeView shape) {
  SPU_TRACE_HAL_DISP(ctx, init, dtype, shape);

  const auto init_dtype = getEncodeType(init.pt_type);

  // FIXME: semantically, this is casting from literal to type, not from one
  // type to another type.
  SPU_ENFORCE(canImplicitCastTo(init_dtype, dtype), "cast from {} to {} failed",
              init_dtype, dtype);

  auto result = make_pub2k(ctx, init).setDtype(dtype, true);

  if (isEmpty(shape)) {
    return Value(NdArrayRef(nullptr, result.storage_type(), shape),
                 result.dtype());
  }

  // If view shape is same as destination shape, just make public
  if (shape.empty() || shape == init.shape) {
    return result;
  }

  // Same calcNumel but shape is different, do a reshape
  if (calcNumel(init.shape) == calcNumel(shape)) {
    return Value(result.data().reshape(shape), result.dtype());
  }

  // Other, do a broadcast, let broadcast handles the sanity check
  SPU_ENFORCE(calcNumel(init.shape) <= calcNumel(shape));
  return Value(result.data().broadcast_to(shape, {}), result.dtype());
}

spu::Value zeros(SPUContext* ctx, DataType dtype,
                 absl::Span<const int64_t> shape) {
  if (dtype == DT_F32 || dtype == DT_F64) {
    return constant(ctx, 0.0F, dtype, shape);
  } else {
    return constant(ctx, static_cast<uint8_t>(0), dtype, shape);
  }
}

Value iota(SPUContext* ctx, DataType dtype, int64_t numel) {
  return DISPATCH_ALL_NONE_BOOL_PT_TYPES(getDecodeType(dtype), "iota", [&]() {
    std::vector<ScalarT> arr(numel);
    std::iota(arr.begin(), arr.end(), 0);
    return constant(ctx, arr, dtype, {numel});
  });
}

Value epsilon(SPUContext* ctx, DataType dtype,
              absl::Span<const int64_t> shape) {
  return _constant(ctx, static_cast<int128_t>(1), shape).setDtype(dtype);
}

}  // namespace spu::kernel::hal
