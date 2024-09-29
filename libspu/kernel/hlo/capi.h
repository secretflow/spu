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

#ifndef SPU_HLO_API_H
#define SPU_HLO_API_H

#include <stdint.h>

#include "mlir-c/IR.h"

// used for debug only, should be removed later.
#define ENABLE_HLO_RUNTIME

#define SPU_HLO_CAPI __attribute__((visibility("default")))

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#define DEFINE_CAPI_OBJECT(name, storage) \
  struct name {                           \
    storage *ptr;                         \
  };                                      \
  typedef struct name name

DEFINE_CAPI_OBJECT(SpuHloBuilder, void);

DEFINE_CAPI_OBJECT(SpuHloPtBufferView, const void);
DEFINE_CAPI_OBJECT(SpuHloValue, const void);

#ifdef ENABLE_HLO_RUNTIME
DEFINE_CAPI_OBJECT(SpuHloRtContext, void);
#endif  // ENABLE_HLO_RUNTIME

#undef DEFINE_CAPI_OBJECT

#define DEFINE_CAPI_ARRAY(name, storage) \
  struct name {                          \
    storage *data;                       \
    uint64_t size;                       \
  };                                     \
  typedef struct name name

DEFINE_CAPI_ARRAY(MlirValueArray, MlirValue);
DEFINE_CAPI_ARRAY(SpuHloValueArray, SpuHloValue);
DEFINE_CAPI_ARRAY(SpuHloRawArray, const uint8_t);

DEFINE_CAPI_ARRAY(SpuHloShape, const int64_t);
DEFINE_CAPI_ARRAY(SpuHloSizes, const int64_t);
DEFINE_CAPI_ARRAY(SpuHloAxes, const int64_t);
DEFINE_CAPI_ARRAY(SpuHloIndex, const int64_t);
DEFINE_CAPI_ARRAY(SpuHloStrides, const int64_t);

#undef DEFINE_CAPI_ARRAY

#define DEFINE_CAPI_ENUM(name, storage) \
  struct name {                         \
    storage data;                       \
  };                                    \
  typedef struct name name

DEFINE_CAPI_ENUM(SpuHloPtType, int64_t);
DEFINE_CAPI_ENUM(SpuHloVisibility, int64_t);
DEFINE_CAPI_ENUM(SpuHloReduceType, int64_t);
DEFINE_CAPI_ENUM(SpuHloSortDirection, int64_t);

#undef DEFINE_CAPI_ENUM

SPU_HLO_CAPI SpuHloBuilder spuHloBuilderCreate();

SPU_HLO_CAPI void spuHloBuilderDestroy(SpuHloBuilder builder);

SPU_HLO_CAPI void spuHloCompile(SpuHloBuilder builder, MlirValueArray outputs);

#ifdef ENABLE_HLO_RUNTIME
SPU_HLO_CAPI SpuHloValueArray spuHloExecute(SpuHloBuilder builder,
                                            SpuHloRtContext rt_ctx,
                                            SpuHloValueArray inputs);
SPU_HLO_CAPI void spuHloValueDestroy(SpuHloValue value);
#endif  // ENABLE_HLO_RUNTIME

SPU_HLO_CAPI MlirValue spuHloConstant(SpuHloBuilder builder,
                                      SpuHloPtBufferView view,
                                      SpuHloShape out_shape);
SPU_HLO_CAPI MlirValue spuHloArgument(SpuHloBuilder builder,
                                      SpuHloPtType pt_type,
                                      SpuHloVisibility visibility,
                                      SpuHloShape shape);

SPU_HLO_CAPI MlirValue spuHloAdd(SpuHloBuilder builder, MlirValue lhs,
                                 MlirValue rhs);
SPU_HLO_CAPI MlirValue spuHloSub(SpuHloBuilder builder, MlirValue lhs,
                                 MlirValue rhs);
SPU_HLO_CAPI MlirValue spuHloMul(SpuHloBuilder builder, MlirValue lhs,
                                 MlirValue rhs);
SPU_HLO_CAPI MlirValue spuHloDiv(SpuHloBuilder builder, MlirValue lhs,
                                 MlirValue rhs);

SPU_HLO_CAPI MlirValue spuHloEqual(SpuHloBuilder builder, MlirValue lhs,
                                   MlirValue rhs);
SPU_HLO_CAPI MlirValue spuHloAnd(SpuHloBuilder builder, MlirValue lhs,
                                 MlirValue rhs);
SPU_HLO_CAPI MlirValue spuHloXor(SpuHloBuilder builder, MlirValue lhs,
                                 MlirValue rhs);
SPU_HLO_CAPI MlirValue spuHloOr(SpuHloBuilder builder, MlirValue lhs,
                                MlirValue rhs);
SPU_HLO_CAPI MlirValue spuHloNotEqual(SpuHloBuilder builder, MlirValue lhs,
                                      MlirValue rhs);

SPU_HLO_CAPI MlirValue spuHloMax(SpuHloBuilder builder, MlirValue lhs,
                                 MlirValue rhs);
SPU_HLO_CAPI MlirValue spuHloMin(SpuHloBuilder builder, MlirValue lhs,
                                 MlirValue rhs);

SPU_HLO_CAPI MlirValue spuHloGreater(SpuHloBuilder builder, MlirValue lhs,
                                     MlirValue rhs);
SPU_HLO_CAPI MlirValue spuHloGreaterEqual(SpuHloBuilder builder, MlirValue lhs,
                                          MlirValue rhs);

SPU_HLO_CAPI MlirValue spuHloLess(SpuHloBuilder builder, MlirValue lhs,
                                  MlirValue rhs);
SPU_HLO_CAPI MlirValue spuHloLessEqual(SpuHloBuilder builder, MlirValue lhs,
                                       MlirValue rhs);

SPU_HLO_CAPI MlirValue spuHloRemainder(SpuHloBuilder builder, MlirValue lhs,
                                       MlirValue rhs);

SPU_HLO_CAPI MlirValue spuHloNot(SpuHloBuilder builder, MlirValue input);
SPU_HLO_CAPI MlirValue spuHloSine(SpuHloBuilder builder, MlirValue input);
SPU_HLO_CAPI MlirValue spuHloCosine(SpuHloBuilder builder, MlirValue input);

SPU_HLO_CAPI MlirValue spuHloSeal(SpuHloBuilder builder, MlirValue input);
SPU_HLO_CAPI MlirValue spuHloReveal(SpuHloBuilder builder, MlirValue input);
SPU_HLO_CAPI MlirValue spuHloCast(SpuHloBuilder builder, MlirValue input,
                                  SpuHloVisibility dst_vtype,
                                  SpuHloPtType dst_dtype);

SPU_HLO_CAPI MlirValue spuHloConcatenate(SpuHloBuilder builder,
                                         MlirValueArray ops, int64_t axis);

SPU_HLO_CAPI MlirValue spuHloPad(SpuHloBuilder builder, MlirValue input,
                                 MlirValue pad_value, SpuHloSizes edge_low,
                                 SpuHloSizes edge_high, SpuHloSizes inner);

SPU_HLO_CAPI MlirValue spuHloReduce(SpuHloBuilder builder,
                                    MlirValueArray inputs,
                                    MlirValueArray init_values,
                                    SpuHloAxes dims_to_reduce,
                                    SpuHloReduceType reduce_type,
                                    int ignore_init_values);

SPU_HLO_CAPI MlirValue spuHloSelect(SpuHloBuilder builder, MlirValue pred,
                                    MlirValue on_true, MlirValue on_false);

SPU_HLO_CAPI MlirValueArray spuHloSimpleSort(SpuHloBuilder builder,
                                             MlirValueArray inputs,
                                             int64_t sort_dim,
                                             SpuHloSortDirection direction,
                                             int64_t num_keys);

SPU_HLO_CAPI MlirValue spuHloSlice(SpuHloBuilder builder, MlirValue input,
                                   SpuHloIndex start, SpuHloIndex end,
                                   SpuHloStrides strides);

SPU_HLO_CAPI MlirValueArray spuHloShuffle(SpuHloBuilder builder,
                                          MlirValueArray inputs, int64_t axis);
SPU_HLO_CAPI MlirValue spuHloFilterByMask(SpuHloBuilder builder,
                                          MlirValue input, SpuHloRawArray mask);

SPU_HLO_CAPI MlirValue spuHloLinearGather(SpuHloBuilder builder,
                                          MlirValue input, SpuHloIndex indices);

SPU_HLO_CAPI MlirValue spuHloLinearScatter(SpuHloBuilder builder,
                                           MlirValue input, MlirValue update,
                                           SpuHloIndex indices);

SPU_HLO_CAPI MlirValue spuHloBroadcast(SpuHloBuilder builder, MlirValue input,
                                       SpuHloShape to_shape,
                                       SpuHloAxes in_dims);

#ifdef ENABLE_HLO_RUNTIME
SPU_HLO_CAPI SpuHloRawArray spuHloDump(SpuHloRtContext rt_ctx,
                                       SpuHloValue input, SpuHloPtType pt_type,
                                       int64_t fxp_bits);
#endif  // ENABLE_HLO_RUNTIME

SPU_HLO_CAPI char *spuHloEmitCodes(SpuHloBuilder builder);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // SPU_HLO_API_H
