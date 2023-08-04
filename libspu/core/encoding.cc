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

NdArrayRef encodeToRing(const NdArrayRef& src, FieldType field, size_t fxp_bits,
                        DataType* out_dtype) {
  SPU_ENFORCE(src.eltype().isa<PtTy>(), "expect PtType, got={}", src.eltype());
  const PtType pt_type = src.eltype().as<PtTy>()->pt_type();
  const size_t numel = src.numel();
  NdArrayRef dst(makeType<RingTy>(field), src.shape());

  if (out_dtype != nullptr) {
    *out_dtype = getEncodeType(pt_type);
  }

  if (pt_type == PT_F32 || pt_type == PT_F64 || pt_type == PT_F16) {
    DISPATCH_FLOAT_PT_TYPES(pt_type, "_", [&]() {
      DISPATCH_ALL_FIELDS(field, "_", [&]() {
        using Float = ScalarT;
        using T = std::make_signed_t<ring2k_t>;

        // Reference: https://eprint.iacr.org/2019/599.pdf
        // To make `msb based comparison` work, the safe range is
        // [-2^(k-2), 2^(k-2))
        const size_t k = sizeof(T) * 8;
        const T kScale = T(1) << fxp_bits;
        const T kFxpLower = -(T)std::pow(2, k - 2);
        const T kFxpUpper = (T)std::pow(2, k - 2) - 1;
        const Float kFlpUpper =
            static_cast<Float>(static_cast<double>(kFxpUpper) / kScale);
        const Float kFlpLower =
            static_cast<Float>(static_cast<double>(kFxpLower) / kScale);

        auto _src = NdArrayView<Float>(src);
        auto _dst = NdArrayView<T>(dst);

        pforeach(0, numel, [&](int64_t idx) {
          const auto src_value = _src[idx];
          if (std::isnan(src_value)) {
            // see numpy.nan_to_num
            // note(jint) I dont know why nan could be
            // encoded as zero..
            _dst[idx] = 0;
          } else if (src_value >= kFlpUpper) {
            _dst[idx] = kFxpUpper;
          } else if (src_value <= kFlpLower) {
            _dst[idx] = kFxpLower;
          } else {
            _dst[idx] = static_cast<T>(src_value * kScale);
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
        auto _src = NdArrayView<Integer>(src);
        auto _dst = NdArrayView<T>(dst);
        // TODO: encoding integer in range [-2^(k-2),2^(k-2))
        pforeach(0, numel, [&](int64_t idx) {
          _dst[idx] = static_cast<T>(_src[idx]);  // NOLINT
        });
      });
    });

    return dst;
  }

  SPU_THROW("shold not be here");
}

NdArrayRef decodeFromRing(const NdArrayRef& src, DataType in_dtype,
                          size_t fxp_bits, PtType* out_pt_type) {
  const Type& src_type = src.eltype();
  const FieldType field = src_type.as<Ring2k>()->field();
  const PtType pt_type = getDecodeType(in_dtype);
  const size_t numel = src.numel();

  SPU_ENFORCE(src_type.isa<RingTy>(), "source must be ring_type, got={}",
              src_type);

  if (out_pt_type != nullptr) {
    *out_pt_type = pt_type;
  }

  NdArrayRef dst(makePtType(pt_type), src.shape());

  DISPATCH_ALL_FIELDS(field, "field", [&]() {
    DISPATCH_ALL_PT_TYPES(pt_type, "pt_type", [&]() {
      using T = std::make_signed_t<ring2k_t>;

      auto _src = NdArrayView<T>(src);
      auto _dst = NdArrayView<ScalarT>(dst);

      if (in_dtype == DT_I1) {
        constexpr bool kSanity = std::is_same_v<ScalarT, bool>;
        SPU_ENFORCE(kSanity);
        pforeach(0, numel,
                 [&](int64_t idx) { _dst[idx] = !((_src[idx] & 0x1) == 0); });
      } else if (in_dtype == DT_F32 || in_dtype == DT_F64 ||
                 in_dtype == DT_F16) {
        const T kScale = T(1) << fxp_bits;
        pforeach(0, numel, [&](int64_t idx) {
          _dst[idx] =
              static_cast<ScalarT>(static_cast<double>(_src[idx]) / kScale);
        });
      } else {
        pforeach(0, numel, [&](int64_t idx) {
          _dst[idx] = static_cast<ScalarT>(_src[idx]);
        });
      }
    });
  });

  return dst;
}

}  // namespace spu
