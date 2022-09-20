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
#include <string>
#include <vector>

#include "yasl/base/exception.h"
#include "yasl/link/link.h"

#include "spu/psi/psi.pb.h"

namespace spu::psi {

class MemoryPsi {
 public:
  explicit MemoryPsi(MemoryPsiConfig config,
                     std::shared_ptr<yasl::link::Context> lctx);
  ~MemoryPsi() = default;

  std::vector<std::string> Run(const std::vector<std::string>& inputs);

 private:
  void CheckOptions() const;

  std::vector<std::string> EcdhPsi(const std::vector<std::string>& inputs);

  std::vector<std::string> KkrtPsi(const std::vector<std::string>& inputs);

  std::vector<std::string> NPartyPsi(const std::vector<std::string>& inputs);

  std::vector<std::string> Ecdh3PartyPsi(
      const std::vector<std::string>& inputs);

  std::vector<std::string> Bc22Psi(const std::vector<std::string>& inputs);

 private:
  MemoryPsiConfig config_;

  std::shared_ptr<yasl::link::Context> lctx_;
};

}  // namespace spu::psi
