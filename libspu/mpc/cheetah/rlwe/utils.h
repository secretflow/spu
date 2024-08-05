// Copyright 2022 Ant Group Co., Ltd.
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

#pragma once
#include <set>
#include <sstream>
#include <vector>

#include "seal/context.h"
#include "yacl/base/buffer.h"
#include "yacl/base/int128.h"

#include "libspu/core/prelude.h"
#include "libspu/mpc/cheetah/rlwe/types.h"

#define CATCH_SEAL_ERROR(state)                          \
  do {                                                   \
    try {                                                \
      state;                                             \
    } catch (const std::logic_error &e) {                \
      SPU_THROW(fmt::format("SEAL error {}", e.what())); \
    }                                                    \
  } while (false)

namespace seal {
class Ciphertext;
class Plaintext;
}  // namespace seal

namespace spu {
class ArrayRef;
}

namespace spu::mpc::cheetah {

template <class SEALObj>
yacl::Buffer EncodeSEALObject(const SEALObj &obj) {
  size_t nbytes = obj.save_size();
  yacl::Buffer out;
  out.resize(nbytes);
  // NOTE(juhou): compr_sze <= nbytes due to the compression in SEAL
  size_t compr_sze = obj.save(out.data<seal::seal_byte>(), nbytes);
  out.resize(compr_sze);
  return out;
}

template <class SEALObj>
std::vector<yacl::Buffer> EncodeSEALObjects(
    const std::vector<SEALObj> &obj_array,
    const std::vector<seal::SEALContext> &contexts) {
  const size_t obj_count = obj_array.size();
  const size_t context_count = contexts.size();
  SPU_ENFORCE(obj_count > 0, fmt::format("doEncode: non object"));
  SPU_ENFORCE(
      0 == obj_count % context_count,
      fmt::format("doEncode: number of objects and SEALContexts mismatch"));

  std::vector<yacl::Buffer> out(obj_count);
  for (size_t idx = 0; idx < obj_count; ++idx) {
    out[idx] = EncodeSEALObject(obj_array[idx]);
  }

  return out;
}

template <class SEALObj>
void DecodeSEALObject(const yacl::Buffer &buf_view,
                      const seal::SEALContext &context, SEALObj *out,
                      bool skip_sanity_check = false) {
  yacl::CheckNotNull(out);
  auto bytes = reinterpret_cast<const seal::seal_byte *>(buf_view.data<char>());
  if (skip_sanity_check) {
    CATCH_SEAL_ERROR(out->unsafe_load(context, bytes, buf_view.size()));
  } else {
    CATCH_SEAL_ERROR(out->load(context, bytes, buf_view.size()));
  }
}

template <class SEALObj>
void DecodeSEALObjects(const std::vector<yacl::Buffer> &buf_view,
                       const std::vector<seal::SEALContext> &contexts,
                       std::vector<SEALObj> *out,
                       bool skip_sanity_check = false) {
  yacl::CheckNotNull(out);
  const size_t obj_count = buf_view.size();
  if (obj_count > 0) {
    const size_t context_count = contexts.size();
    SPU_ENFORCE(
        0 == obj_count % context_count,
        fmt::format("doDecode: number of objects and SEALContexts mismatch"));

    out->resize(obj_count);
    const size_t stride = obj_count / context_count;
    for (size_t idx = 0, c = 0; idx < obj_count; idx += stride, ++c) {
      for (size_t offset = 0; offset < stride; ++offset) {
        DecodeSEALObject(buf_view[idx + offset], contexts[c],
                         out->data() + idx + offset, skip_sanity_check);
      }
    }
  }
}

// Truncate ciphertexts for a smaller communication.
// NOTE: after truncation, further homomorphic operation is meaningless.
void TruncateBFVForDecryption(seal::Ciphertext &ct,
                              const seal::SEALContext &context);

void NttInplace(seal::Plaintext &pt, const seal::SEALContext &context);

void InvNttInplace(seal::Plaintext &pt, const seal::SEALContext &context);

// requires ciphertext.is_ntt_form() is `false`
//          ciphertext.size() is `2`
void RemoveCoefficientsInplace(RLWECt &ciphertext,
                               const std::set<size_t> &to_remove);

void KeepCoefficientsInplace(RLWECt &ciphertext,
                             const std::set<size_t> &to_keep);

// x mod prime
template <typename T>
uint64_t BarrettReduce(T x, const seal::Modulus &prime) {
  if constexpr (std::is_same_v<T, uint128_t>) {
    uint64_t z[2]{static_cast<uint64_t>(x), static_cast<uint64_t>(x >> 64)};
    return seal::util::barrett_reduce_128(z, prime);
  } else {
    return seal::util::barrett_reduce_64(static_cast<uint64_t>(x), prime);
  }
}

// Erase the memory automatically
struct AutoMemGuard {
  explicit AutoMemGuard(ArrayRef *obj);

  explicit AutoMemGuard(RLWEPt *pt);

  ~AutoMemGuard();

  ArrayRef *obj_{nullptr};
  RLWEPt *pt_{nullptr};
};

}  // namespace spu::mpc::cheetah
