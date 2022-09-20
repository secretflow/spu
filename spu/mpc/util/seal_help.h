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

// Author: Wen-jie Lu(juhou)

#pragma once
#include <sstream>
#include <vector>

#include "seal/context.h"
#include "yasl/base/buffer.h"

#define CATCH_SEAL_ERROR(state)                                       \
  do {                                                                \
    try {                                                             \
      state;                                                          \
    } catch (const std::logic_error &e) {                             \
      YASL_THROW_LOGIC_ERROR(fmt::format("SEAL error {}", e.what())); \
    }                                                                 \
  } while (false)

namespace seal {
class Ciphertext;
class Plaintext;
}  // namespace seal

namespace spu::mpc {
#include "absl/numeric/bits.h"

template <typename T>
inline bool IsTwoPower(T v) {
  return absl::has_single_bit(v);
}

inline uint64_t Next2Pow(uint64_t a) { return absl::bit_ceil(a); }

template <class SEALObj>
yasl::Buffer EncodeSEALObject(const SEALObj &obj) {
  size_t nbytes = obj.save_size();
  yasl::Buffer out;
  out.resize(nbytes);
  // NOTE(juhou): compr_sze <= nbytes due to the compression in SEAL
  size_t compr_sze = obj.save(out.data<seal::seal_byte>(), nbytes);
  out.resize(compr_sze);
  return out;
}

template <class SEALObj>
std::vector<yasl::Buffer> EncodeSEALObjects(
    const std::vector<SEALObj> &obj_array,
    const std::vector<seal::SEALContext> &contexts) {
  const size_t obj_count = obj_array.size();
  const size_t context_count = contexts.size();
  YASL_ENFORCE(obj_count > 0, fmt::format("doEncode: non object"));
  YASL_ENFORCE(
      0 == obj_count % context_count,
      fmt::format("doEncode: number of objects and SEALContexts mismatch"));

  std::vector<yasl::Buffer> out(obj_count);
  for (size_t idx = 0; idx < obj_count; ++idx) {
    out[idx] = EncodeSEALObject(obj_array[idx]);
  }

  return out;
}

template <class SEALObj>
void DecodeSEALObject(const yasl::Buffer &buf_view,
                      const seal::SEALContext &context, SEALObj *out,
                      bool skip_sanity_check = false) {
  yasl::CheckNotNull(out);
  auto bytes = reinterpret_cast<const seal::seal_byte *>(buf_view.data<char>());
  if (skip_sanity_check) {
    CATCH_SEAL_ERROR(out->unsafe_load(context, bytes, buf_view.size()));
  } else {
    CATCH_SEAL_ERROR(out->load(context, bytes, buf_view.size()));
  }
}

template <class SEALObj>
void DecodeSEALObjects(const std::vector<yasl::Buffer> &buf_view,
                       const std::vector<seal::SEALContext> &contexts,
                       std::vector<SEALObj> *out,
                       bool skip_sanity_check = false) {
  yasl::CheckNotNull(out);
  const size_t obj_count = buf_view.size();
  if (obj_count > 0) {
    const size_t context_count = contexts.size();
    YASL_ENFORCE(
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

}  // namespace spu::mpc
