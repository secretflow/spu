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

#include <map>
#include <memory>
#include <string>

#include "libspu/core/cexpr.h"
#include "libspu/core/prelude.h"

namespace spu {

// TODO(jint) document the object model.
// - KernelEvalContext: the calling convention
// - Kernel: the dynamic member function.
// - State: the dynamic member variable.
// - Object: the dynamic binding object.

class KernelEvalContext;
class Kernel {
 public:
  enum class Kind {
    // Indicate the kernel's complexity is static known.
    //
    // Typically, static kernel does not depend on runtime options, such like
    // selecting different kernels according to different configs.
    //
    // By default, we should make kernels as 'Static' as possible.
    Static,

    // Indicate the kernel depends on runtime options, this kind of kernel is
    // hard to analysis statically.
    Dynamic,
  };

  virtual ~Kernel() = default;

  virtual Kind kind() const { return Kind::Static; }

  // Calculate number of comm rounds required for this kernel.
  virtual ce::CExpr latency() const { return nullptr; }

  // Calculate number of comm in bits.
  virtual ce::CExpr comm() const { return nullptr; }

  // Some kernel's communication can not be measured/implemented easily, this
  // field tells the tolerance of theory value and implementation diff, in
  // percentage. i.e.
  //
  // kernel's cost is infinitely close to 2.
  //   comm(kernel) = 1 + 1/2 + 1/4 + 1/8 + ... = lim(2)
  //
  // in implementation, program may not handle bit, but instead use byte as the
  // minimum unit, this make the error even larger.
  virtual float getCommTolerance() const { return 0.0; }

  // Evaluate this protocol within given context.
  virtual void evaluate(KernelEvalContext* ectx) const = 0;
};

class State {
 public:
  virtual ~State() = default;

  // Return true if the state could be forked with low cost.
  virtual bool hasLowCostFork() const { return false; }

  // TODO: this is a const method.
  virtual std::unique_ptr<State> fork();
};

// A dynamic object contains a set of kernels and a set of states.
//
// Class that inherit from this class could do `dynamic binding`.
class Object final {
  std::map<std::string, std::shared_ptr<Kernel>> kernels_;
  std::map<std::string, std::unique_ptr<State>> states_;

  std::string id_;   // this object id.
  std::string pid_;  // parent id.

  mutable int64_t child_counter_ = 0;

 public:
  explicit Object(std::string id, std::string pid = "")
      : id_(std::move(id)), pid_(std::move(pid)) {}

  virtual ~Object() = default;

  const std::string& id() const { return id_; }
  const std::string& pid() const { return pid_; }

  //
  std::unique_ptr<Object> fork() const;

  bool hasLowCostFork() const;

  void regKernel(const std::string& name, std::unique_ptr<Kernel> kernel);

  template <typename KernelT>
  void regKernel() {
    regKernel(KernelT::kBindName, std::make_unique<KernelT>());
  }

  template <typename KernelT>
  void regKernel(const std::string& name) {
    return regKernel(name, std::make_unique<KernelT>());
  }

  Kernel* getKernel(const std::string& name) const;
  bool hasKernel(const std::string& name) const;

  void addState(const std::string& name, std::unique_ptr<State> state) {
    const auto& itr = states_.find(name);
    SPU_ENFORCE(itr == states_.end(), "state={} already exist", name);
    states_.emplace(name, std::move(state));
  }

  template <typename StateT, typename... Args>
  void addState(Args&&... args) {
    addState(StateT::kBindName,
             std::make_unique<StateT>(std::forward<Args>(args)...));
  }

  template <typename StateT>
  StateT* getState() {
    const auto& itr = states_.find(StateT::kBindName);
    SPU_ENFORCE(itr != states_.end(), "state={} not found", StateT::kBindName);
    return dynamic_cast<StateT*>(itr->second.get());
  }

  //
  std::vector<std::string> getKernelNames() const {
    std::vector<std::string> names;
    names.reserve(kernels_.size());
    for (auto const& itr : kernels_) {
      names.push_back(itr.first);
    }
    return names;
  }
};

}  // namespace spu
