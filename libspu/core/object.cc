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

#include "libspu/core/object.h"

namespace spu {

std::unique_ptr<State> State::fork() {
  SPU_THROW("Not implemented, the sub class should override this");
}

std::unique_ptr<Object> Object::fork() const {
  auto new_id = fmt::format("{}-{}", id_, child_counter_++);
  auto new_obj = std::make_unique<Object>(new_id, id_);
  new_obj->kernels_ = kernels_;
  for (const auto& [key, val] : states_) {
    new_obj->addState(key, val->fork());
  }
  return new_obj;
}

bool Object::hasLowCostFork() const {
  for (const auto& [key, val] : states_) {
    if (!val->hasLowCostFork()) {
      return false;
    }
  }
  return true;
}

void Object::regKernel(const std::string& name,
                       std::unique_ptr<Kernel> kernel) {
  const auto itr = kernels_.find(name);
  SPU_ENFORCE(itr == kernels_.end(), "kernel={} already exist", name);
  kernels_.insert({name, std::move(kernel)});
}

Kernel* Object::getKernel(const std::string& name) const {
  const auto itr = kernels_.find(name);
  SPU_ENFORCE(itr != kernels_.end(), "kernel={} not found", name);
  return itr->second.get();
}

bool Object::hasKernel(const std::string& name) const {
  const auto itr = kernels_.find(name);
  return itr != kernels_.end();
}

}  // namespace spu
