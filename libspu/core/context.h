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

#include "yacl/link/link.h"

#include "libspu/core/object.h"
#include "libspu/core/prelude.h"
#include "libspu/core/trace.h"  // TODO: bad reference, but implicitly include too much.
#include "libspu/core/value.h"

#include "libspu/spu.pb.h"

namespace spu {

// The hal evaluation context for all spu operators.
class SPUContext final {
  RuntimeConfig config_;

  // A dynamic object for polymophic(multi-stage) operations.
  std::unique_ptr<Object> prot_;

  // TODO(jint): do we really need a link here? how about a FHE context.
  std::shared_ptr<yacl::link::Context> lctx_;

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
};

using HalContext [[deprecated("Use SPUContext instead.")]] = SPUContext;

class KernelEvalContext final {
  // Please keep param types as less as possible.
  using ParamType = std::variant<  //
      Value,                       // value type
      Shape,                       //
      size_t,                      // represent size(mmul), shift_bits(shift)
      bool,                        // binary flag
      Type,                        // type of type
      uint128_t                    // ring constant
      >;

  SPUContext* sctx_;

  std::vector<ParamType> params_;
  ParamType output_;

 public:
  explicit KernelEvalContext(SPUContext* sctx) : sctx_(sctx) {}

  SPUContext* sctx() { return sctx_; }

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

  // Steal the output from this evaluation context.
  //
  // * usually called by kernel caller.
  template <typename T = Value>
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

  // Get the i'th parameter.
  //
  // * usually called by kernel callee.
  template <typename T>
  const T& getParam(size_t pos) const {
    SPU_ENFORCE(pos < params_.size(), "pos={} exceed num of inputs={}", pos,
                params_.size());
    return std::get<T>(params_[pos]);
  }

  // Set the output.
  //
  // * usually called by kernel callee.
  template <typename T = Value>
  void setOutput(T&& out) {
    output_ = std::forward<T>(out);
  }
};

namespace detail {

template <typename First, typename... Args>
void bindParams(KernelEvalContext* ectx, First&& head, Args&&... tail) {
  ectx->bindParam(std::forward<First>(head));
  if constexpr (sizeof...(Args) > 0) {
    return bindParams(ectx, std::forward<Args>(tail)...);
  }
}

}  // namespace detail

// Dynamic dispath to a kernel according to a symbol name.
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
  return ectx.stealOutput<Ret>();
}

// helper class
template <typename T>
using OptionalAPI = std::optional<T>;
inline constexpr std::nullopt_t NotAvailable = std::nullopt;

// TODO: currently unstable, statically config it.
// When it's stable move it to RuntimeConfig or even enable it by default.
// #define SPU_ENABLE_PRIVATE_TYPE

void setupTrace(spu::SPUContext* sctx, const spu::RuntimeConfig& rt_config);

}  // namespace spu
