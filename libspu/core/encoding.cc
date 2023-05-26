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

#include "libspu/core/encoding.h"

#include <cstdint>
#include <type_traits>

#include "libspu/core/parallel_utils.h"

namespace spu {

DataType getEncodeType(PtType pt_type) {
#define CASE(PTYPE, DTYPE) \
  case (PTYPE):            \
    return DTYPE;

  switch (pt_type) {
    MAP_PTTYPE_TO_DTYPE(CASE)
    default:
      SPU_THROW("invalid PtType {}", pt_type);
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
      SPU_THROW("invalid DataType {}", dtype);
  }

#undef CASE
}

ArrayRef encodeToRing(const ArrayRef& src, FieldType field, size_t fxp_bits,
                      DataType* out_dtype) {
  SPU_ENFORCE(src.eltype().isa<PtTy>(), "expect PtType, got={}", src.eltype());
  const PtType pt_type = src.eltype().as<PtTy>()->pt_type();
  const size_t numel = src.numel();
  ArrayRef dst(makeType<RingTy>(field), numel);

  if (out_dtype != nullptr) {
    *out_dtype = getEncodeType(pt_type);
  }

  auto src_stride = src.stride();
  auto dst_stride = dst.stride();

  if (pt_type == PT_F32 || pt_type == PT_F64) {
    DISPATCH_FLOAT_PT_TYPES(pt_type, "_", [&]() {
      DISPATCH_ALL_FIELDS(field, "_", [&]() {
        using Float = ScalarT;
        using T = std::make_signed_t<ring2k_t>;
        T* dst_ptr = &dst.at<T>(0);
        Float const* src_ptr = &src.at<Float>(0);

        // Reference: https://eprint.iacr.org/2019/599.pdf
        // To make `msb based comparison` work, the safe range is
        // [-2^(k-2), 2^(k-2))
        const size_t k = sizeof(T) * 8;
        const T kScale = T(1) << fxp_bits;
        const T kFxpLower = -(T)std::pow(2, k - 2);
        const T kFxpUpper = (T)std::pow(2, k - 2) - 1;
        const Float kFlpUpper = static_cast<Float>(kFxpUpper) / kScale;
        const Float kFlpLower = static_cast<Float>(kFxpLower) / kScale;

        pforeach(0, numel, [&](int64_t idx) {
          auto src_value = src_ptr[idx * src_stride];
          if (std::isnan(src_value)) {
            // see numpy.nan_to_num
            // note(jint) I dont know why nan could be
            // encoded as zero..
            dst_ptr[idx * dst_stride] = 0;
          } else if (src_value >= kFlpUpper) {
            dst_ptr[idx * dst_stride] = kFxpUpper;
          } else if (src_value <= kFlpLower) {
            dst_ptr[idx * dst_stride] = kFxpLower;
          } else {
            dst_ptr[idx * dst_stride] = static_cast<T>(src_value * kScale);
          }
        });
      });
    });
    return dst;
  } else {
    // handle integer & boolean
    DISPATCH_INT_PT_TYPES(pt_type, "_", [&]() {
      DISPATCH_ALL_FIELDS(field, "_", [&]() {
        using Integer = ScalarT;
        SPU_ENFORCE(sizeof(ring2k_t) >= sizeof(Integer),
                    "integer encoding failed, ring={} could not represent {}",
                    field, pt_type);

        using T = std::make_signed_t<ring2k_t>;
        // TODO: encoding integer in range [-2^(k-2),2^(k-2))
        T* dst_ptr = &dst.at<T>(0);
        Integer const* src_ptr = &src.at<Integer>(0);
        pforeach(0, numel, [&](int64_t idx) {
          dst_ptr[idx * dst_stride] = static_cast<T>(src_ptr[idx * src_stride]);
        });
      });
    });

    return dst;
  }

  SPU_THROW("shold not be here");
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

  SPU_ENFORCE(src_type.isa<RingTy>(), "source must be ring_type, got={}",
              src_type);

  if (out_pt_type != nullptr) {
    *out_pt_type = pt_type;
  }

  ArrayRef dst(makePtType(pt_type), src.numel());

  auto src_stride = src.stride();
  auto dst_stride = dst.stride();

  DISPATCH_ALL_FIELDS(field, "field", [&]() {
    DISPATCH_ALL_PT_TYPES(pt_type, "pt_type", [&]() {
      using T = std::make_signed_t<ring2k_t>;
      T const* src_ptr = &src.at<T>(0);
      ScalarT* dst_ptr = &dst.at<ScalarT>(0);

      if (in_dtype == DT_I1) {
        constexpr bool kSanity = std::is_same_v<ScalarT, bool>;
        SPU_ENFORCE(kSanity);
        pforeach(0, numel, [&](int64_t idx) {
          dst_ptr[idx * dst_stride] = !((src_ptr[idx * src_stride] & 0x1) == 0);
        });
      } else if (in_dtype == DT_F32 || in_dtype == DT_F64) {
        const T kScale = T(1) << fxp_bits;
        pforeach(0, numel, [&](int64_t idx) {
          dst_ptr[idx * dst_stride] =
              static_cast<ScalarT>(src_ptr[idx * src_stride]) / kScale;
        });
      } else {
        pforeach(0, numel, [&](int64_t idx) {
          dst_ptr[idx * dst_stride] =
              static_cast<ScalarT>(src_ptr[idx * src_stride]);
        });
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
