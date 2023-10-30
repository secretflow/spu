// Copyright 2023 zhangwfjh
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

#include "libspu/psi/core/fnp04_mp_psi/fnp04_mp_psi.h"
#include "libspu/psi/operator/base_operator.h"

namespace spu::psi {

class FnpMpPsiOperator : public PsiBaseOperator {
 public:
  using Options = FNP04Party::Options;

  explicit FnpMpPsiOperator(const Options& options)
      : PsiBaseOperator(options.link_ctx), options_(options) {}

  std::vector<std::string> OnRun(const std::vector<std::string>& inputs) final {
    return FNP04Party{options_}.Run(inputs);
  }

 private:
  Options options_;
};

}  // namespace spu::psi
