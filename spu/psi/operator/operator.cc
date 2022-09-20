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

#include "spu/psi/operator/operator.h"

namespace spu::psi {

std::shared_ptr<PsiBaseOperator> CreatePsiOperator(const std::any& opts) {
  std::shared_ptr<PsiBaseOperator> op;
  if (opts.type() == typeid(Ecdh3PartyPsiOperator::Options)) {
    auto ecdh_3party_opts = std::any_cast<Ecdh3PartyPsiOperator::Options>(opts);
    op = std::make_shared<Ecdh3PartyPsiOperator>(ecdh_3party_opts);
  } else if (opts.type() == typeid(KkrtPsiOperator::Options)) {
    auto kkrt_opts = std::any_cast<KkrtPsiOperator::Options>(opts);
    op = std::make_shared<KkrtPsiOperator>(kkrt_opts);
  } else if (opts.type() == typeid(NpartyPsiOperator::Options)) {
    auto nparty_opts = std::any_cast<NpartyPsiOperator::Options>(opts);
    op = std::make_shared<NpartyPsiOperator>(nparty_opts);
  } else if (opts.type() == typeid(Bc22PcgPsiOperator::Options)) {
    auto pcg_opts = std::any_cast<Bc22PcgPsiOperator::Options>(opts);
    op = std::make_shared<Bc22PcgPsiOperator>(pcg_opts);
  } else {
    YASL_THROW("unknow psi opts type {}", opts.type().name());
  }

  return op;
}
}  // namespace spu::psi