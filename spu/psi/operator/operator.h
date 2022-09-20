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

#include <any>
#include <memory>

#include "spu/psi/operator/bc22_2party_psi.h"
#include "spu/psi/operator/ecdh_3party_psi.h"
#include "spu/psi/operator/kkrt_2party_psi.h"
#include "spu/psi/operator/nparty_psi.h"

namespace spu::psi {

std::shared_ptr<PsiBaseOperator> CreatePsiOperator(const std::any& opts);

}  // namespace spu::psi
