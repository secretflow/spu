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

#include "spu/mpc/object.h"

namespace spu::mpc {

void Object::regKernel(std::string_view name, std::unique_ptr<Kernel> kernel) {
  const auto itr = kernels_.find(name);
  YACL_ENFORCE(itr == kernels_.end(), "kernel={} already exist", name);
  kernels_.insert({name, std::move(kernel)});
}

Kernel* Object::getKernel(std::string_view name) {
  const auto itr = kernels_.find(name);
  YACL_ENFORCE(itr != kernels_.end(), "kernel={} not found", name);
  return itr->second.get();
}

bool Object::hasKernel(std::string_view name) const {
  const auto itr = kernels_.find(name);
  return itr != kernels_.end();
}

}  // namespace spu::mpc
