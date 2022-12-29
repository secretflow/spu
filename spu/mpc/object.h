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
#include <variant>

#include "yacl/base/exception.h"

#include "spu/core/array_ref.h"
#include "spu/mpc/util/cexpr.h"

namespace spu::mpc {

// TODO(jint) document the object model.
// - KernelEvalContext: the calling convention
// - Kernel: the dynamic member function.
// - State: the dynamic member variable.
// - Object: the dynamic binding object.

class Object;

// Helper class to instantiate kernel calls.
class KernelEvalContext final {
  // Please keep param types as less as possible.
  using ParamType = std::variant<  //
      ArrayRef,                    // value type
      size_t,                      // represent size(mmul), shift_bits(shift)
      Type,                        // type of type
      uint128_t                    // ring constant
      >;

  Object* caller_;

  std::vector<ParamType> params_;
  ParamType output_;

 public:
  explicit KernelEvalContext(Object* caller) : caller_(caller) {}

  std::string name() const {
    return "TODO";
    // return fmt::format("CTX:{}", std::this_thread::get_id());
  }

  size_t numParams() const { return params_.size(); }

  // Steal the output from this evaluation context.
  //
  // * usually called by kernel caller.
  template <typename T = ArrayRef>
  T&& stealOutput() {
    return std::move(std::get<T>(output_));
  }

  // Bind an input to this evaluation context.
  //
  // * usually called by kernel caller.
  template <typename T>
  void bindParam(const T& in) {
    params_.emplace_back(in);
  }

  // Get the caller's pointer.
  //
  // * usually called by kernel callee.
  template <typename T = Object>
  T* caller() {
    if (auto caller = dynamic_cast<T*>(caller_)) {
      return caller;
    }
    YACL_THROW("cast failed");
  }

  // Get the i'th parameter.
  //
  // * usually called by kernel callee.
  template <typename T>
  const T& getParam(size_t pos) const {
    YACL_ENFORCE(pos < params_.size(), "pos={} exceed num of inputs={}", pos,
                 params_.size());
    return std::get<T>(params_[pos]);
  }

  // Set the output.
  //
  // * usually called by kernel callee.
  template <typename T = ArrayRef>
  void setOutput(T&& out) {
    output_ = std::forward<T>(out);
  }
};

class Kernel {
 public:
  using EvalContext = KernelEvalContext;

  enum class Kind {
    // Indicate the kernel's complexity is static known.
    //
    // Typically, static kernel does not depend on runtime options, such like
    // selecting different kernels according to different configs.
    //
    // By default, we should make kernels as 'atomic' as possible.
    kStatic,

    // Indicate the kernel depends on runtime options, this kind of kernel is
    // hard to analysis statically.
    kDynamic,
  };

 public:
  virtual ~Kernel() = default;

  virtual Kind kind() const { return Kind::kStatic; }

  // Calculate number of comm rounds required for this kernel.
  virtual util::CExpr latency() const { return nullptr; }

  // Calculate number of comm in bits.
  virtual util::CExpr comm() const { return nullptr; }

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
  virtual void evaluate(EvalContext* ctx) const = 0;
};

class State {
 public:
  virtual ~State() = default;

  virtual bool hasLowCostFork() const { return false; }
  virtual std::unique_ptr<State> fork();
};

// A (kernel) dynamic object dispatch a function to a kernel at runtime.
//
// Class that inherit from this class could do `dynamic binding`.
class Object final {
  std::map<std::string_view, std::shared_ptr<Kernel>> kernels_;
  std::map<std::string_view, std::unique_ptr<State>> states_;

  std::string id_;   // this object id.
  std::string pid_;  // parent id.

  int64_t child_counter_ = 0;

 public:
  explicit Object(std::string id, std::string pid = "")
      : id_(std::move(id)), pid_(std::move(pid)) {}

  const std::string& id() const { return id_; }
  const std::string& pid() const { return pid_; }

  //
  std::unique_ptr<Object> fork();

  bool hasLowCostFork() const;

  void regKernel(std::string_view name, std::unique_ptr<Kernel> kernel);

  template <typename KernelT>
  void regKernel() {
    regKernel(KernelT::kBindName, std::make_unique<KernelT>());
  }

  template <typename KernelT>
  void regKernel(std::string_view name) {
    return regKernel(name, std::make_unique<KernelT>());
  }

  Kernel* getKernel(std::string_view name);
  bool hasKernel(std::string_view name) const;

  void addState(std::string_view name, std::unique_ptr<State> state) {
    const auto& itr = states_.find(name);
    YACL_ENFORCE(itr == states_.end(), "state={} already exist", name);
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
    YACL_ENFORCE(itr != states_.end(), "state={} not found", StateT::kBindName);
    return dynamic_cast<StateT*>(itr->second.get());
  }

  //
  std::vector<std::string_view> getKernelNames() const {
    std::vector<std::string_view> names;
    names.reserve(kernels_.size());
    for (auto const& itr : kernels_) {
      names.push_back(itr.first);
    }
    return names;
  }

  template <typename Ret = ArrayRef>
  Ret callImpl(Kernel* kernel, KernelEvalContext* ctx) {
    kernel->evaluate(ctx);
    return ctx->stealOutput<Ret>();
  }

  template <typename Ret = ArrayRef, typename First, typename... Args>
  Ret callImpl(Kernel* kernel, KernelEvalContext* ctx, First&& head,
               Args&&... tail) {
    ctx->bindParam(std::forward<First>(head));
    if constexpr (sizeof...(Args) == 0) {
      return callImpl<Ret>(kernel, ctx);
    } else {
      return callImpl<Ret>(kernel, ctx, std::forward<Args>(tail)...);
    }
  }

  template <typename Ret = ArrayRef, typename... Args>
  Ret call(std::string_view name, Args&&... args) {
    Kernel* kernel = getKernel(name);
    KernelEvalContext ctx(this);
    return callImpl<Ret>(kernel, &ctx, std::forward<Args>(args)...);
  }
};

}  // namespace spu::mpc
