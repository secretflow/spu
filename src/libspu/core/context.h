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

#include <memory>
#include <optional>
#include <variant>

#include "yacl/link/context.h"

#include "libspu/core/object.h"
#include "libspu/core/prelude.h"
#include "libspu/core/value.h"

#include "libspu/spu.pb.h"

namespace spu {

// The hal evaluation context for all spu operators.
class SPUContext final {
  RuntimeConfig config_;

  // A dynamic object for polymorphic(multi-stage) operations.
  std::unique_ptr<Object> prot_;

  // TODO(jint): do we really need a link here? how about a FHE context.
  std::shared_ptr<yacl::link::Context> lctx_;

  // Min number of cores in SPU cluster
  int32_t max_cluster_level_concurrency_;

 public:
  explicit SPUContext(const RuntimeConfig& config,
                      const std::shared_ptr<yacl::link::Context>& lctx);

  SPUContext(const SPUContext& other) = delete;
  SPUContext& operator=(const SPUContext& other) = delete;
  SPUContext(SPUContext&& other) = default;

  // all parties get a 'corresponding' hal context when forked.
  std::unique_ptr<SPUContext> fork() const;

  //
  const std::shared_ptr<yacl::link::Context>& lctx() const { return lctx_; }

  // Return current working fixed point fractional bits.
  size_t getFxpBits() const {
    const auto fbits = config_.fxp_fraction_bits();
    SPU_ENFORCE(fbits != 0);
    return fbits;
  }

  // Return current working field of MPC engine.
  FieldType getField() const { return config_.field(); }

  // Return current working runtime config.
  const RuntimeConfig& config() const { return config_; }

  const std::string& id() { return prot_->id(); }
  const std::string& pid() { return prot_->pid(); }

  Object* prot() { return prot_.get(); }

  // helper function, forward to caller
  bool hasKernel(const std::string& name) const {
    return prot_->hasKernel(name);
  }
  Kernel* getKernel(const std::string& name) const {
    return prot_->getKernel(name);
  }
  template <typename StateT>
  StateT* getState() {
    return prot_->template getState<StateT>();
  }

  // If any task assumes same level of parallelism across all instances,
  // this is the max number of tasks to launch at the same time.
  int32_t getClusterLevelMaxConcurrency() const {
    return max_cluster_level_concurrency_;
  }
};

class KernelEvalContext final {
  // Please keep param types as less as possible.
  using ParamType = std::variant<  //
      Value,                       // value type
      Shape,                       //
      size_t,                      // represent size(mmul), shift_bits(shift)
      bool,                        // binary flag
      Type,                        // type of type
      uint128_t,                   // ring constant
      int64_t,                     //
      SignType,                    //
      std::vector<Value>,          //
      Axes,                        //
      Index,                       //
      Strides,                     //
      Sizes                        //
      >;

  SPUContext* sctx_;

  std::vector<ParamType> params_;
  std::vector<ParamType> outputs_;

 public:
  explicit KernelEvalContext(SPUContext* sctx) : sctx_(sctx) {}

  SPUContext* sctx() { return sctx_; }

  const std::shared_ptr<yacl::link::Context>& lctx() const {
    return sctx_->lctx();
  }

  const std::string& id() { return sctx_->id(); }
  const std::string& pid() { return sctx_->pid(); }

  // helper function, forward to caller.
  template <typename StateT>
  StateT* getState() {
    return sctx_->prot()->template getState<StateT>();
  }

  // helper function, forward to caller
  bool hasKernel(const std::string& name) const {
    return sctx_->prot()->hasKernel(name);
  }

  size_t numParams() const { return params_.size(); }
  size_t numOutputs() const { return outputs_.size(); }

  // Steal the output from this evaluation context.
  //
  // * usually called by kernel caller.
  template <typename T = Value>
  T&& consumeOutput(size_t pos) {
    SPU_DEBUG_ONLY_ENFORCE(pos < outputs_.size(),
                           "pos={} exceed num of outputs={}", pos,
                           outputs_.size());
    return std::move(std::get<T>(outputs_[pos]));
  }

  // Bind an input to this evaluation context.
  //
  // * usually called by kernel caller.
  template <typename T>
  void pushParam(const T& in) {
    params_.emplace_back(in);
  }

  // Get the i'th parameter.
  //
  // * usually called by kernel callee.
  template <typename T>
  const T& getParam(size_t pos) const {
    SPU_DEBUG_ONLY_ENFORCE(pos < params_.size(),
                           "pos={} exceed num of inputs={}", pos,
                           params_.size());
    return std::get<T>(params_[pos]);
  }

  // Set the output.
  //
  // * usually called by kernel callee.
  template <typename T = Value>
  void pushOutput(T&& out) {
    outputs_.emplace_back(std::forward<T>(out));
  }
};

namespace detail {

template <typename First, typename... Args>
void bindParams(KernelEvalContext* ectx, First&& head, Args&&... tail) {
  ectx->pushParam(std::forward<First>(head));
  if constexpr (sizeof...(Args) > 0) {
    bindParams(ectx, std::forward<Args>(tail)...);
  }
}

}  // namespace detail

// Dynamic dispatch to a kernel according to a symbol name.
template <typename Ret = Value, typename... Args>
Ret dynDispatch(SPUContext* sctx, const std::string& name, Args&&... args) {
  /// Steps of dynamic dispatch.
  // 1. find a prop kernel.
  Kernel* kernel = sctx->prot()->getKernel(name);

  // 2. prep parameters (flatten it into an evaluation context).
  KernelEvalContext ectx(sctx);
  detail::bindParams(&ectx, std::forward<Args>(args)...);

  // 3. call a visitor, visit a kernel with params.
  // TODO: use a visitor to call different stage of a kernel
  kernel->evaluate(&ectx);

  // 4. steal the result and return it.
  if (ectx.numOutputs() > 0) {
    return ectx.consumeOutput<Ret>(0);
  }
  return Ret();
}

// helper class
template <typename T>
using OptionalAPI = std::optional<T>;
inline constexpr std::nullopt_t NotAvailable = std::nullopt;

void setupTrace(spu::SPUContext* sctx, const spu::RuntimeConfig& rt_config);

}  // namespace spu
