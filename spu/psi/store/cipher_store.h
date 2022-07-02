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

#pragma once

namespace spu::psi {

/// ICipherStore stores dual encrypted results.
class ICipherStore {
 public:
  virtual ~ICipherStore() = default;

  // SaveSelf/SavePeer saves the dual encrypted ciphertext.
  //
  // Threading:
  // Each function is guaranteed to be called in one thread during the
  // `RunEcdhPsi`. However, the caller threads for these two functions are
  // different.
  //
  // Order:
  // The save order is same as the input order provided by `IBatchProvider`.
  //
  // Contraint:
  // The two functions wont be called by `RunEcdhPsi` if my rank does not
  // match the `target_rank`.
  virtual void SaveSelf(std::string ciphertext) = 0;
  virtual void SavePeer(std::string ciphertext) = 0;
};

}  // namespace spu::psi
