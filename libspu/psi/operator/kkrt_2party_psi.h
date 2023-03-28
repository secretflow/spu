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

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "libspu/psi/core/kkrt_psi.h"
#include "libspu/psi/operator/base_operator.h"

namespace spu::psi {

class KkrtPsiOperator : public PsiBaseOperator {
 public:
  struct Options {
    std::shared_ptr<yacl::link::Context> link_ctx;
    size_t receiver_rank = 0;
    size_t num_ot = 512;
  };

  static Options ParseConfig(const MemoryPsiConfig& config,
                             const std::shared_ptr<yacl::link::Context>& lctx);

  explicit KkrtPsiOperator(const Options& options)
      : PsiBaseOperator(options.link_ctx), options_(options) {}

  std::vector<std::string> OnRun(const std::vector<std::string>& inputs) final;

 private:
  Options options_;
};

}  // namespace spu::psi
