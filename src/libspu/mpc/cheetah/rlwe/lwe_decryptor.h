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

#include <memory>

#include "libspu/mpc/cheetah/rlwe/modswitch_helper.h"
#include "libspu/mpc/cheetah/rlwe/types.h"

namespace spu::mpc::cheetah {

class LWEDecryptor {
 public:
  explicit LWEDecryptor(const LWESecretKey &sk,
                        const seal::SEALContext &context,
                        const ModulusSwitchHelper &ms_helper);

  ~LWEDecryptor();

  LWEDecryptor &operator=(const LWEDecryptor &) = delete;
  LWEDecryptor(const LWEDecryptor &) = delete;
  LWEDecryptor(LWEDecryptor &&) = delete;

  void Decrypt(const LWECt &ciphertext, uint32_t *out) const;

  void Decrypt(const LWECt &ciphertext, uint64_t *out) const;

  void Decrypt(const LWECt &ciphertext, uint128_t *out) const;

 protected:
  template <typename T>
  void DoDecrypt(const LWECt &ciphertext, T *out) const;

 private:
  LWESecretKey sk_;
  seal::SEALContext context_;
  ModulusSwitchHelper ms_helper_;
};

}  // namespace spu::mpc::cheetah
