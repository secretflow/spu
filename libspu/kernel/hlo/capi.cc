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

#include "libspu/kernel/hlo/capi.h"

#include <vector>

#include "libspu/kernel/hlo/builder.h"

using namespace spu::kernel::hlo;

#define DEFINE_CAPI_OBJECT_WRAP(name, cpptype)                \
  static inline name wrap(cpptype *cpp) { return name{cpp}; } \
  static inline cpptype *unwrap(name c) {                     \
    return static_cast<cpptype *>(c.ptr);                     \
  }

DEFINE_CAPI_OBJECT_WRAP(SpuHloBuilder, HloBuilder)

DEFINE_CAPI_OBJECT_WRAP(SpuHloPtBufferView, const spu::PtBufferView)
DEFINE_CAPI_OBJECT_WRAP(SpuHloValue, const spu::MemRef)

#ifdef ENABLE_HLO_RUNTIME
DEFINE_CAPI_OBJECT_WRAP(SpuHloRtContext, spu::SPUContext)
#endif  // ENABLE_HLO_RUNTIME

#undef DEFINE_CAPI_OBJECT_WRAP

#define DEFINE_CAPI_ARRAY_UNWRAP(name, cpptype) \
  static inline cpptype unwrap(name c) {        \
    return cpptype(c.data, c.data + c.size);    \
  }

DEFINE_CAPI_ARRAY_UNWRAP(SpuHloShape, spu::Shape)
DEFINE_CAPI_ARRAY_UNWRAP(SpuHloSizes, spu::Sizes)
DEFINE_CAPI_ARRAY_UNWRAP(SpuHloAxes, spu::Axes)
DEFINE_CAPI_ARRAY_UNWRAP(SpuHloIndex, spu::Index)
DEFINE_CAPI_ARRAY_UNWRAP(SpuHloStrides, spu::Strides)

#undef DEFINE_CAPI_ARRAY_UNWRAP

#define DEFINE_CAPI_ENUM_UNWRAP(name, cpptype) \
  static inline cpptype unwrap(name c) { return static_cast<cpptype>(c.data); }

DEFINE_CAPI_ENUM_UNWRAP(SpuHloPtType, spu::PtType)
DEFINE_CAPI_ENUM_UNWRAP(SpuHloVisibility, spu::Visibility)
DEFINE_CAPI_ENUM_UNWRAP(SpuHloReduceType, HloBuilder::ReduceType)
DEFINE_CAPI_ENUM_UNWRAP(SpuHloSortDirection, HloBuilder::SortDirection)

#undef DEFINE_CAPI_ENUM_UNWRAP

static inline MlirValue wrap(const mlir::Value *cpp) {
  return MlirValue{cpp->getImpl()};
}
static inline MlirValue wrap(mlir::Value cpp) { return wrap(&cpp); }
static inline mlir::Value unwrap(MlirValue c) {
  return static_cast<mlir::detail::ValueImpl *>(const_cast<void *>(c.ptr));
}

MlirValueArray wrap(const std::vector<mlir::Value> &src) {
  MlirValueArray dst;

  dst.size = src.size();
  dst.data = new MlirValue[dst.size];

  for (uint64_t i = 0; i < dst.size; ++i) {
    dst.data[i] = wrap(src[i]);
  }

  return dst;
}

std::vector<mlir::Value> unwrap(MlirValueArray src) {
  std::vector<mlir::Value> dst;
  dst.reserve(src.size);

  for (uint64_t i = 0; i < src.size; ++i) {
    dst.emplace_back(unwrap(src.data[i]));
  }

  return dst;
}

#ifdef ENABLE_HLO_RUNTIME
static inline SpuHloValue wrap(spu::MemRef cpp) {
  return SpuHloValue{new spu::MemRef{cpp}};
}

SpuHloValueArray wrap(const std::vector<spu::MemRef> &src) {
  SpuHloValueArray dst;

  dst.size = src.size();
  dst.data = new SpuHloValue[dst.size];

  for (size_t i = 0; i < dst.size; ++i) {
    dst.data[i] = wrap(src[i]);
  }

  return dst;
}

std::vector<spu::MemRef> unwrap(SpuHloValueArray src) {
  std::vector<spu::MemRef> dst;
  dst.reserve(src.size);

  for (uint64_t i = 0; i < src.size; ++i) {
    dst.emplace_back(*unwrap(src.data[i]));
  }

  return dst;
}
#endif  // ENABLE_HLO_RUNTIME

SpuHloBuilder spuHloBuilderCreate() {
  auto *builder = new HloBuilder();
  return wrap(builder);
}

void spuHloBuilderDestroy(SpuHloBuilder builder) { delete unwrap(builder); }

void spuHloCompile(SpuHloBuilder builder, MlirValueArray outputs) {
  unwrap(builder)->compile(unwrap(outputs));
}

#ifdef ENABLE_HLO_RUNTIME
SpuHloValueArray spuHloExecute(SpuHloBuilder builder, SpuHloRtContext rt_ctx,
                               SpuHloValueArray inputs) {
  return wrap(unwrap(builder)->execute(unwrap(rt_ctx), unwrap(inputs)));
}

void spuHloValueDestroy(SpuHloValue value) { delete unwrap(value); }
#endif  // ENABLE_HLO_RUNTIME

MlirValue spuHloConstant(SpuHloBuilder builder, SpuHloPtBufferView view,
                         SpuHloShape out_shape) {
  return wrap(unwrap(builder)->Constant(*unwrap(view), unwrap(out_shape)));
}

MlirValue spuHloArgument(SpuHloBuilder builder, SpuHloPtType pt_type,
                         SpuHloVisibility visibility, SpuHloShape shape) {
  return wrap(unwrap(builder)->Argument(unwrap(pt_type), unwrap(visibility),
                                        unwrap(shape)));
}

#define IMP_BINARY_OP(_Op_)                                       \
  MlirValue spuHlo##_Op_(SpuHloBuilder builder, MlirValue lhs,    \
                         MlirValue rhs) {                         \
    return wrap(unwrap(builder)->_Op_(unwrap(lhs), unwrap(rhs))); \
  }

IMP_BINARY_OP(Add)
IMP_BINARY_OP(Sub)
IMP_BINARY_OP(Mul)
IMP_BINARY_OP(Div)
IMP_BINARY_OP(Equal)
IMP_BINARY_OP(And)
IMP_BINARY_OP(Xor)
IMP_BINARY_OP(Or)
IMP_BINARY_OP(NotEqual)
IMP_BINARY_OP(Max)
IMP_BINARY_OP(Min)
IMP_BINARY_OP(Greater)
IMP_BINARY_OP(GreaterEqual)
IMP_BINARY_OP(Less)
IMP_BINARY_OP(LessEqual)
IMP_BINARY_OP(Remainder)

#undef IMP_BINARY_OP

#define IMP_UNARY_OP(_Op_)                                         \
  MlirValue spuHlo##_Op_(SpuHloBuilder builder, MlirValue input) { \
    return wrap(unwrap(builder)->_Op_(unwrap(input)));             \
  }

IMP_UNARY_OP(Not)
IMP_UNARY_OP(Sine)
IMP_UNARY_OP(Cosine)
IMP_UNARY_OP(Seal)
IMP_UNARY_OP(Reveal)

#undef IMP_UNARY_OP

MlirValue spuHloCast(SpuHloBuilder builder, MlirValue input,
                     SpuHloVisibility dst_vtype, SpuHloPtType dst_dtype) {
  return wrap(unwrap(builder)->Cast(unwrap(input), unwrap(dst_vtype),
                                    unwrap(dst_dtype)));
}

MlirValue spuHloConcatenate(SpuHloBuilder builder, MlirValueArray ops,
                            int64_t axis) {
  return wrap(unwrap(builder)->Concatenate(unwrap(ops), axis));
}

MlirValue spuHloPad(SpuHloBuilder builder, MlirValue input, MlirValue pad_value,
                    SpuHloSizes edge_low, SpuHloSizes edge_high,
                    SpuHloSizes inner) {
  return wrap(unwrap(builder)->Pad(unwrap(input), unwrap(pad_value),
                                   unwrap(edge_low), unwrap(edge_high),
                                   unwrap(inner)));
}

MlirValue spuHloReduce(SpuHloBuilder builder, MlirValueArray inputs,
                       MlirValueArray init_values, SpuHloAxes dims_to_reduce,
                       SpuHloReduceType reduce_type, int ignore_init_values) {
  auto inputs_ = unwrap(inputs);
  auto init_values_ = unwrap(init_values);

  return wrap(unwrap(builder)->Reduce(
      absl::Span<const mlir::Value>{inputs_.data(), inputs_.size()},
      absl::Span<const mlir::Value>{init_values_.data(), init_values_.size()},
      unwrap(dims_to_reduce), unwrap(reduce_type), ignore_init_values));
}

MlirValue spuHloSelect(SpuHloBuilder builder, MlirValue pred, MlirValue on_true,
                       MlirValue on_false) {
  return wrap(
      unwrap(builder)->Select(unwrap(pred), unwrap(on_true), unwrap(on_false)));
}

MlirValueArray spuHloSimpleSort(SpuHloBuilder builder, MlirValueArray inputs,
                                int64_t sort_dim, SpuHloSortDirection direction,
                                int64_t num_keys) {
  auto inputs_ = unwrap(inputs);
  return wrap(unwrap(builder)->SimpleSort(
      absl::Span<const mlir::Value>{inputs_.data(), inputs_.size()}, sort_dim,
      unwrap(direction), num_keys));
}

MlirValue spuHloSlice(SpuHloBuilder builder, MlirValue input, SpuHloIndex start,
                      SpuHloIndex end, SpuHloStrides strides) {
  return wrap(unwrap(builder)->Slice(unwrap(input), unwrap(start), unwrap(end),
                                     unwrap(strides)));
}

MlirValueArray spuHloShuffle(SpuHloBuilder builder, MlirValueArray inputs,
                             int64_t axis) {
  auto inputs_ = unwrap(inputs);
  return wrap(unwrap(builder)->Shuffle(
      absl::Span<const mlir::Value>{inputs_.data(), inputs_.size()}, axis));
}

MlirValue spuHloFilterByMask(SpuHloBuilder builder, MlirValue input,
                             SpuHloRawArray mask) {
  return wrap(unwrap(builder)->FilterByMask(
      unwrap(input), absl::Span<const uint8_t>{mask.data, mask.size}));
}

MlirValue spuHloLinearGather(SpuHloBuilder builder, MlirValue input,
                             SpuHloIndex indices) {
  return wrap(unwrap(builder)->LinearGather(unwrap(input), unwrap(indices)));
}

MlirValue spuHloLinearScatter(SpuHloBuilder builder, MlirValue input,
                              MlirValue update, SpuHloIndex indices) {
  return wrap(unwrap(builder)->LinearScatter(unwrap(input), unwrap(update),
                                             unwrap(indices)));
}

MlirValue spuHloBroadcast(SpuHloBuilder builder, MlirValue input,
                          SpuHloShape to_shape, SpuHloAxes in_dims) {
  return wrap(unwrap(builder)->Broadcast(unwrap(input), unwrap(to_shape),
                                         unwrap(in_dims)));
}

#ifdef ENABLE_HLO_RUNTIME
SpuHloRawArray spuHloDump(SpuHloRtContext rt_ctx, SpuHloValue input,
                          SpuHloPtType pt_type, int64_t fxp_bits) {
  auto buf = HloBuilder::Dump(unwrap(rt_ctx), *unwrap(input), unwrap(pt_type),
                              fxp_bits);
  SpuHloRawArray ret = {reinterpret_cast<uint8_t *>(buf->data()),
                        static_cast<uint64_t>(buf->size())};
  buf.reset();  // release ownership

  return ret;
}

char *spuHloEmitCodes(SpuHloBuilder builder) {
  auto codes_str = unwrap(builder)->EmitCodes();
  char *codes_cstr = (char *)malloc(codes_str.length() + 1);
  strcpy(codes_cstr, codes_str.c_str());
  return codes_cstr;
}

#endif  // ENABLE_HLO_RUNTIME
