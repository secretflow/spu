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

void encodeToRing(const PtBufferView& src, MemRef& out, int64_t fxp_bits) {
  const PtType pt_type = src.pt_type;
  const size_t numel = src.shape.numel();

  auto* const out_type = out.eltype().as<RingTy>();
  SPU_ENFORCE(out_type);

  const auto out_semantic_type = out.eltype().semantic_type();

  if (pt_type == PT_F16 || pt_type == PT_F32 || pt_type == PT_F64) {
    SPU_ENFORCE(fxp_bits > 0, "Invalid fxp bits {}", fxp_bits);

    DISPATCH_FLOAT_PT_TYPES(pt_type, [&]() {
      using Float = ScalarT;
      DISPATCH_ALL_STORAGE_TYPES(out_type->storage_type(), [&]() {
        using T = std::make_signed_t<ScalarT>;

        // Reference: https://eprint.iacr.org/2019/599.pdf
        // To make `msb based comparison` work, the safe range is
        // [-2^(k-2), 2^(k-2))
        const size_t k = sizeof(T) * 8;
        const T kScale = T(1) << fxp_bits;
        const T kFxpLower = -(T)std::pow(2, k - 2);
        const T kFxpUpper = (T)std::pow(2, k - 2) - 1;
        const auto kFlpUpper =
            static_cast<Float>(static_cast<double>(kFxpUpper) / kScale);
        const auto kFlpLower =
            static_cast<Float>(static_cast<double>(kFxpLower) / kScale);

        auto _dst = MemRefView<T>(out);

        pforeach(0, numel, [&](int64_t idx) {
          auto src_value = src.get<Float>(idx);
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
  } else {
    // handle integer & boolean
    DISPATCH_INT_PT_TYPES(pt_type, [&]() {
      using Integer = ScalarT;

      DISPATCH_ALL_STORAGE_TYPES(out_type->storage_type(), [&]() {
        SPU_ENFORCE(sizeof(ScalarT) >= sizeof(Integer),
                    "integer encoding failed, ring={} could not represent {}",
                    out_semantic_type, pt_type);

        using T = std::make_signed_t<ScalarT>;

        auto _dst = MemRefView<T>(out);
        // TODO: encoding integer in range [-2^(k-2),2^(k-2))
        pforeach(0, numel, [&](int64_t idx) {
          auto src_value = src.get<Integer>(idx);
          _dst[idx] = static_cast<T>(src_value);  // NOLINT
        });
      });
    });
  }
}

void decodeFromRing(const MemRef& src, PtBufferView& out, int64_t fxp_bits) {
  const auto* src_type = src.eltype().as<RingTy>();

  SPU_ENFORCE(src_type);

  const PtType pt_type = out.pt_type;

  const size_t numel = src.numel();

  DISPATCH_ALL_STORAGE_TYPES(src_type->storage_type(), [&]() {
    using ring2k_t = ScalarT;
    DISPATCH_ALL_PT_TYPES(pt_type, [&]() {
      using T = std::make_signed_t<ring2k_t>;

      auto _src = MemRefView<T>(src);

      if (pt_type == PT_I1) {
        pforeach(0, numel, [&](int64_t idx) {
          bool value = !((_src[idx] & 0x1) == 0);
          out.set<bool>(idx, value);
        });
      } else if (pt_type == PT_F32 || pt_type == PT_F64 || pt_type == PT_F16) {
        const T kScale = T(1) << fxp_bits;
        pforeach(0, numel, [&](int64_t idx) {
          auto value =
              static_cast<ScalarT>(static_cast<double>(_src[idx]) / kScale);
          out.set<ScalarT>(idx, value);
        });
      } else {
        pforeach(0, numel, [&](int64_t idx) {
          auto value = static_cast<ScalarT>(_src[idx]);
          out.set<ScalarT>(idx, value);
        });
      }
    });
  });
}

}  // namespace spu
