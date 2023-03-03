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

#pragma once

#include "yacl/link/link.h"

#include "libspu/mpc/object.h"

namespace spu::mpc::test {

template <typename Func>
class NamedFunction : public std::function<Func> {
 private:
  std::string name_ = "unamed";

 public:
  explicit NamedFunction(std::function<Func> func)
      : std::function<Func>(std::move(func)) {}
  explicit NamedFunction(std::function<Func> func, std::string name)
      : std::function<Func>(std::move(func)), name_(std::move(name)) {}

  const std::string& name() const { return name_; }
};

using CreateObjectFn = NamedFunction<std::unique_ptr<Object>(
    const RuntimeConfig& conf,
    const std::shared_ptr<yacl::link::Context>& lctx)>;

using OpTestParams = std::tuple<CreateObjectFn, RuntimeConfig, size_t>;

}  // namespace spu::mpc::test