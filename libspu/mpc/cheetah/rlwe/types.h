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

#include <iosfwd>

#include "seal/ciphertext.h"
#include "seal/plaintext.h"
#include "seal/secretkey.h"
#include "seal/serialization.h"

namespace spu::mpc::cheetah {

using RLWESecretKey = seal::SecretKey;

using RLWEPublicKey = seal::PublicKey;

using KSwitchKeys = seal::KSwitchKeys;

using GaloisKeys = seal::GaloisKeys;

using RLWECt = seal::Ciphertext;

using RLWEPt = seal::Plaintext;

class LWECt;

class PhantomLWECt;

}  // namespace spu::mpc::cheetah