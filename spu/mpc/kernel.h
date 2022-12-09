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

#include <thread>
#include <variant>

#include "yacl/base/exception.h"

#include "spu/core/array_ref.h"
#include "spu/core/type.h"
#include "spu/mpc/util/cexpr.h"

namespace spu::mpc {

class Object;

// Helper class to instantiate kernel calls.
class KernelEvalContext final {
  // Please keep param types as less as possible.
  using ParamType = std::variant<  //
      FieldType,  // TODO: drop this field when default field is added.
      size_t,     // for shift, mmul etc.
      ArrayRef,   // value type
      Type        // type of type
      >;

  Object* caller_;

  std::vector<ParamType> params_;
  ParamType output_;

 public:
  explicit KernelEvalContext(Object* caller) : caller_(caller) {}

  std::string name() const {
    return fmt::format("CTX:{}", std::this_thread::get_id());
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
    output_ = std::move(out);
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

class UnaryKernel : public Kernel {
 public:
  void evaluate(EvalContext* ctx) const override {
    ctx->setOutput(proc(ctx, ctx->getParam<ArrayRef>(0)));
  }
  virtual ArrayRef proc(EvalContext* ctx, const ArrayRef& in) const = 0;
};

class ShiftKernel : public Kernel {
 public:
  void evaluate(EvalContext* ctx) const override {
    ctx->setOutput(
        proc(ctx, ctx->getParam<ArrayRef>(0), ctx->getParam<size_t>(1)));
  }
  virtual ArrayRef proc(EvalContext* ctx, const ArrayRef& in,
                        size_t bits) const = 0;
};

class BinaryKernel : public Kernel {
 public:
  void evaluate(EvalContext* ctx) const override {
    ctx->setOutput(
        proc(ctx, ctx->getParam<ArrayRef>(0), ctx->getParam<ArrayRef>(1)));
  }
  virtual ArrayRef proc(EvalContext* ctx, const ArrayRef& lhs,
                        const ArrayRef& rhs) const = 0;
};

class MatmulKernel : public Kernel {
 public:
  void evaluate(KernelEvalContext* ctx) const override {
    ctx->setOutput(proc(ctx, ctx->getParam<ArrayRef>(0),
                        ctx->getParam<ArrayRef>(1), ctx->getParam<size_t>(2),
                        ctx->getParam<size_t>(3), ctx->getParam<size_t>(4)));
  }
  virtual ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& A,
                        const ArrayRef& B, size_t M, size_t N,
                        size_t K) const = 0;
};

class BitrevKernel : public Kernel {
 public:
  void evaluate(KernelEvalContext* ctx) const override {
    ctx->setOutput(proc(ctx, ctx->getParam<ArrayRef>(0),
                        ctx->getParam<size_t>(1), ctx->getParam<size_t>(2)));
  }
  virtual ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in,
                        size_t start, size_t end) const = 0;
};

class TruncPrAKernel : public ShiftKernel {
 public:
  virtual bool isPrecise() const = 0;
};

}  // namespace spu::mpc
