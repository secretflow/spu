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

#include "libspu/psi/core/dp_psi/dp_psi.h"
#include "libspu/psi/operator/base_operator.h"

namespace spu::psi {

class DpPsiOperator : public PsiBaseOperator {
 public:
  DpPsiOperator(const std::shared_ptr<yacl::link::Context>& lctx,
                const DpPsiOptions& options, size_t receiver_rank,
                CurveType curve_type = CurveType::CURVE_25519);

  std::vector<std::string> OnRun(
      const std::vector<std::string>& inputs) override final;

 private:
  DpPsiOptions dp_options_;
  size_t receiver_rank_;
  CurveType curve_type_ = CurveType::CURVE_25519;
};

}  // namespace spu::psi
