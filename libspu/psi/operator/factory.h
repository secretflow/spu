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
#include <mutex>
#include <unordered_map>

#include "libspu/core/prelude.h"
#include "libspu/psi/operator/base_operator.h"

#include "libspu/psi/psi.pb.h"

namespace spu::psi {

using OperatorCreator = std::function<std::unique_ptr<PsiBaseOperator>(
    const MemoryPsiConfig& config,
    const std::shared_ptr<yacl::link::Context>& lctx)>;

class OperatorFactory {
 public:
  static OperatorFactory* GetInstance() {
    static OperatorFactory factory;
    return &factory;
  }

  void Register(const std::string& type, OperatorCreator creator) {
    std::lock_guard<std::mutex> lock(mutex_);
    SPU_ENFORCE(creators_.find(type) == creators_.end(),
                "duplicated creator registered for {}", type);
    creators_[type] = std::move(creator);
  }

  std::unique_ptr<PsiBaseOperator> Create(
      const MemoryPsiConfig& config,
      const std::shared_ptr<yacl::link::Context>& lctx) {
    std::string type = PsiType_Name(config.psi_type());
    auto creator = creators_[type];
    SPU_ENFORCE(creator, "no creator registered for operator type: {}", type);
    return creator(config, lctx);
  }

 protected:
  OperatorFactory() {}
  virtual ~OperatorFactory() {}
  OperatorFactory(const OperatorFactory&) = delete;
  OperatorFactory& operator=(const OperatorFactory&) = delete;
  OperatorFactory(OperatorFactory&&) = delete;
  OperatorFactory& operator=(OperatorFactory&&) = delete;

 private:
  std::unordered_map<std::string, OperatorCreator> creators_;
  std::mutex mutex_;
};

class OperatorRegistrar {
 public:
  explicit OperatorRegistrar(const std::string& type, OperatorCreator creator) {
    OperatorFactory::GetInstance()->Register(type, std::move(creator));
  }
};

#define REGISTER_OPERATOR(type, creator) \
  static OperatorRegistrar registrar__##type##__object(#type, creator);

}  // namespace spu::psi
