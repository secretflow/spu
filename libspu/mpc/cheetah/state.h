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
#include <mutex>

#include "libspu/core/context.h"
#include "libspu/core/ndarray_ref.h"
#include "libspu/core/object.h"
#include "libspu/mpc/cheetah/arith/cheetah_dot.h"
#include "libspu/mpc/cheetah/arith/cheetah_mul.h"
#include "libspu/mpc/cheetah/ot/basic_ot_prot.h"
#include "libspu/mpc/cheetah/rlwe/utils.h"

#include "libspu/spu.pb.h"

namespace spu::mpc::cheetah {

using OTUnaryFunc = std::function<NdArrayRef(
    const NdArrayRef& sub, const std::shared_ptr<BasicOTProtocols>& ot)>;

using OTBinaryFunc =
    std::function<NdArrayRef(const NdArrayRef& op0, const NdArrayRef& op1,
                             const std::shared_ptr<BasicOTProtocols>& ot)>;

using OTUnaryFuncWithU8 = std::function<NdArrayRef(
    absl::Span<const uint8_t> op, const std::shared_ptr<BasicOTProtocols>& ot)>;

using OTBinaryFuncWithU8 = std::function<NdArrayRef(
    const NdArrayRef& op0, absl::Span<const uint8_t> op1,
    const std::shared_ptr<BasicOTProtocols>& ot)>;

NdArrayRef TiledDispatchOTFunc(KernelEvalContext* ctx, const NdArrayRef& x,
                               OTUnaryFunc func);

NdArrayRef TiledDispatchOTFunc(KernelEvalContext* ctx, const NdArrayRef& x,
                               const NdArrayRef& y, OTBinaryFunc func);

NdArrayRef TiledDispatchOTFunc(KernelEvalContext* ctx,
                               absl::Span<const uint8_t> x,
                               OTUnaryFuncWithU8 func);

NdArrayRef TiledDispatchOTFunc(KernelEvalContext* ctx, const NdArrayRef& x,
                               absl::Span<const uint8_t> y,
                               OTBinaryFuncWithU8 func);

class CheetahMulState : public State {
 private:
  mutable std::mutex lock_;
  // a[2] = a[0] * a[1]
  mutable int64_t cached_sze_{0};
  FieldType field_{FT_INVALID};
  NdArrayRef cached_beaver_[3];

  std::unique_ptr<CheetahMul> mul_prot_;
  std::shared_ptr<yacl::link::Context> duplx_;

  // NOTE(juhou): make sure the lock is obtained
  void makeSureCacheSize(FieldType, int64_t numel);

  explicit CheetahMulState(std::unique_ptr<CheetahMul> mul_prot)
      : mul_prot_(std::move(mul_prot)) {}

 public:
  static constexpr const char* kBindName() { return "CheetahMul"; }

  explicit CheetahMulState(const std::shared_ptr<yacl::link::Context>& lctx,
                           bool enable_mul_lsb_error = false) {
    mul_prot_ = std::make_unique<CheetahMul>(lctx, enable_mul_lsb_error);
    duplx_ = lctx->Spawn();
  }

  ~CheetahMulState() override = default;

  CheetahMul* get() { return mul_prot_.get(); }

  std::shared_ptr<yacl::link::Context> duplx() { return duplx_; }

  std::array<NdArrayRef, 3> TakeCachedBeaver(FieldType field, int64_t num);
};

class CheetahDotState : public State {
 private:
  std::unique_ptr<CheetahDot> dot_prot_;

  explicit CheetahDotState(std::unique_ptr<CheetahDot> dot_prot)
      : dot_prot_(std::move(dot_prot)) {}

 public:
  static constexpr const char* kBindName() { return "CheetahDot"; }

  explicit CheetahDotState(const std::shared_ptr<yacl::link::Context>& lctx,
                           bool disable_matmul_pack = false) {
    dot_prot_ = std::make_unique<CheetahDot>(lctx, disable_matmul_pack);
  }

  ~CheetahDotState() override = default;

  CheetahDot* get() { return dot_prot_.get(); }
};

class CheetahOTState : public State {
 private:
  using ProtPtr = std::shared_ptr<BasicOTProtocols>;

  mutable std::mutex lock_;

  static constexpr size_t kMaxOTParallel = 48;

  size_t maximum_instances_ = 0;
  std::vector<ProtPtr> basic_ot_prot_;
  CheetahOtKind ot_kind_;

 public:
  static constexpr const char* kBindName() { return "CheetahOT"; }

  explicit CheetahOTState(size_t maximum_instances, CheetahOtKind ot_kind)
      : maximum_instances_(std::min(kMaxOTParallel, maximum_instances)),
        basic_ot_prot_(maximum_instances_),
        ot_kind_(ot_kind) {
    SPU_ENFORCE(maximum_instances_ > 0);
    std::string ot_type;
    switch (ot_kind_) {
      default:
      case CheetahOtKind::YACL_Ferret:
        ot_type = "yacl_ferret";
        break;
      case CheetahOtKind::EMP_Ferret:
        ot_type = "emp_ferret";
        break;
      case CheetahOtKind::YACL_Softspoken:
        ot_type = "yacl_softspoken";
        break;
    }
    SPDLOG_DEBUG("CHEETAH: Uses {} OT", ot_type);
  }

  ~CheetahOTState() override = default;

  size_t maximum_instances() const { return maximum_instances_; }

  void LazyInit(Communicator* comm, size_t idx = 0) {
    SPU_ENFORCE(idx < maximum_instances_, "idx={} out-of-bound", idx);
    std::lock_guard guard(lock_);
    if (basic_ot_prot_[idx]) {
      return;
    }
    // NOTE(lwj): create a separated link for OT
    // We **do not** block on the OT link since the message volume is small for
    // LPN-based OTe
    auto link = comm->lctx()->Spawn();
    link->SetThrottleWindowSize(0);
    auto _comm = std::make_shared<Communicator>(std::move(link));
    basic_ot_prot_[idx] =
        std::make_shared<BasicOTProtocols>(std::move(_comm), ot_kind_);
  }

  std::shared_ptr<BasicOTProtocols> get(size_t idx = 0) {
    SPU_ENFORCE(idx < maximum_instances_, "idx={} out-of-bound", idx);
    SPU_ENFORCE(basic_ot_prot_[idx], "call LazyInit first");
    return basic_ot_prot_[idx];
  }
};

}  // namespace spu::mpc::cheetah
