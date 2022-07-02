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

#include "spu/core/encoding.h"

namespace spu {

DataType getEncodeType(PtType pt_type) {
#define CASE(PTYPE, DTYPE) \
  case (PTYPE):            \
    return DTYPE;

  switch (pt_type) {
    MAP_PTTYPE_TO_DTYPE(CASE)
    default:
      YASL_THROW("invalid PtType {}", pt_type);
  }

#undef CASE
}

PtType getDecodeType(DataType dtype) {
#define CASE(DTYPE, PTYPE) \
  case (DTYPE):            \
    return PTYPE;

  switch (dtype) {
    MAP_DTYPE_TO_PTTYPE(CASE)
    default:
      YASL_THROW("invalid DataType {}", dtype);
  }

#undef CASE
}

ArrayRef encodeToRing(const ArrayRef& src, FieldType field, size_t fxp_bits,
                      DataType* out_dtype) {
  YASL_ENFORCE(src.eltype().isa<PtTy>(), "expect PtType, got={}", src.eltype());
  const PtType pt_type = src.eltype().as<PtTy>()->pt_type();
  const size_t numel = src.numel();
  ArrayRef dst(makeType<RingTy>(field), numel);

  if (out_dtype) {
    *out_dtype = getEncodeType(pt_type);
  }

  if (pt_type == PT_F32 || pt_type == PT_F64) {
    DISPATCH_FLOAT_PT_TYPES(pt_type, "_", [&]() {
      DISPATCH_ALL_FIELDS(field, "_", [&]() {
        using Float = _PtTypeT;
        ring2k_t* dst_ptr = &dst.at<ring2k_t>(0);
        Float const* src_ptr = &src.at<Float>(0);

        // Reference: https://eprint.iacr.org/2019/599.pdf
        // To make `msb based comparison` work, the safe range is
        // [-2^(k-2), 2^(k-2))
        const size_t k = sizeof(ring2k_t) * 8;
        const ring2k_t kScale = ring2k_t(1) << fxp_bits;
        const ring2k_t kFxpLower = -(ring2k_t)std::pow(2, k - 2);
        const ring2k_t kFxpUpper = (ring2k_t)std::pow(2, k - 2) - 1;
        const Float kFlpUpper = static_cast<Float>(kFxpUpper) / kScale;
        const Float kFlpLower = static_cast<Float>(kFxpLower) / kScale;

        // std::cout << kFlpUpper << " " << kFlpLower << std::endl;

        for (size_t idx = 0; idx < numel; idx++) {
          if (std::isnan(*src_ptr)) {
            // see numpy.nan_to_num
            // note(jint) I dont know why nan could be encoded as zero..
            *dst_ptr = 0;
          } else if (*src_ptr >= kFlpUpper) {
            *dst_ptr = kFxpUpper;
          } else if (*src_ptr <= kFlpLower) {
            *dst_ptr = kFxpLower;
          } else {
            *dst_ptr = static_cast<ring2k_t>(*src_ptr * kScale);
          }

          src_ptr += src.stride();
          dst_ptr += dst.stride();
        }
      });
    });

    return dst;
  } else {
    // handle integer & boolean
    DISPATCH_INT_PT_TYPES(pt_type, "_", [&]() {
      DISPATCH_ALL_FIELDS(field, "_", [&]() {
        using Integer = _PtTypeT;
        YASL_ENFORCE(sizeof(ring2k_t) >= sizeof(Integer),
                     "integer encoding failed, ring={} could not represent {}",
                     field, pt_type);

        // TODO: encoding integer in range [-2^(k-2),2^(k-2))
        ring2k_t* dst_ptr = &dst.at<ring2k_t>(0);
        Integer const* src_ptr = &src.at<Integer>(0);
        for (size_t idx = 0; idx < numel; idx++) {
          *dst_ptr = static_cast<ring2k_t>(*src_ptr);
          src_ptr += src.stride();
          dst_ptr += dst.stride();
        }
      });
    });

    return dst;
  }

  YASL_THROW("shold not be here");
}

NdArrayRef encodeToRing(const NdArrayRef& src, FieldType field, size_t fxp_bits,
                        DataType* out_dtype) {
  return unflatten(encodeToRing(flatten(src), field, fxp_bits, out_dtype),
                   src.shape());
}

ArrayRef decodeFromRing(const ArrayRef& src, DataType in_dtype, size_t fxp_bits,
                        PtType* out_pt_type) {
  const Type& src_type = src.eltype();
  const FieldType field = src_type.as<Ring2k>()->field();
  const PtType pt_type = getDecodeType(in_dtype);
  const size_t numel = src.numel();

  YASL_ENFORCE(src_type.isa<RingTy>(), "source must be ring_type, got={}",
               src_type);

  if (out_pt_type) {
    *out_pt_type = pt_type;
  }

  ArrayRef dst(makePtType(pt_type), src.numel());

  DISPATCH_ALL_FIELDS(field, "field", [&]() {
    DISPATCH_ALL_PT_TYPES(pt_type, "pt_type", [&]() {
      ring2k_t const* src_ptr = &src.at<ring2k_t>(0);
      _PtTypeT* dst_ptr = &dst.at<_PtTypeT>(0);

      if (in_dtype == DT_I1) {
        constexpr bool kSanity = std::is_same_v<_PtTypeT, bool>;
        YASL_ENFORCE(kSanity);
        for (size_t idx = 0; idx < numel; idx++) {
          *dst_ptr = !((*src_ptr & 0x1) == 0);
          src_ptr += src.stride();
          dst_ptr += dst.stride();
        }
      } else if (in_dtype == DT_FXP) {
        const ring2k_t kScale = ring2k_t(1) << fxp_bits;
        for (size_t idx = 0; idx < numel; idx++) {
          *dst_ptr = static_cast<_PtTypeT>(*src_ptr) / kScale;
          src_ptr += src.stride();
          dst_ptr += dst.stride();
        }
      } else {
        for (size_t idx = 0; idx < numel; idx++) {
          *dst_ptr = static_cast<_PtTypeT>(*src_ptr);
          src_ptr += src.stride();
          dst_ptr += dst.stride();
        }
      }
    });
  });

  return dst;
}

NdArrayRef decodeFromRing(const NdArrayRef& src, DataType in_dtype,
                          size_t fxp_bits, PtType* out_pt_type) {
  return unflatten(
      decodeFromRing(flatten(src), in_dtype, fxp_bits, out_pt_type),
      src.shape());
}

}  // namespace spu
