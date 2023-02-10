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

#include <string>
#include <vector>

#include "yacl/link/link.h"

#include "libspu/psi/psi.pb.h"

namespace spu::psi {

class PsiBaseOperator {
 public:
  explicit PsiBaseOperator(std::shared_ptr<yacl::link::Context> link_ctx);
  virtual ~PsiBaseOperator() = default;

  // after call OnRun, it decides whether to broadcast result or not based on
  // param `broadcast_result`
  std::vector<std::string> Run(const std::vector<std::string>& inputs,
                               bool broadcast_result);

  virtual std::vector<std::string> OnRun(
      const std::vector<std::string>& inputs) = 0;

 protected:
  std::shared_ptr<yacl::link::Context> link_ctx_;
};

}  // namespace spu::psi
