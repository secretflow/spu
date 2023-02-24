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
#include <shared_mutex>

#include "libspu/mpc/cheetah/arith/cheetah_dot.h"
#include "libspu/mpc/cheetah/arith/cheetah_mul.h"
#include "libspu/mpc/cheetah/ot/basic_ot_prot.h"
#include "libspu/mpc/object.h"

namespace spu::mpc::cheetah {

class CheetahMulState : public State {
 private:
  std::shared_mutex lock_;
  // a[2] = a[0] * a[1]
  mutable int64_t cached_sze_{0};
  FieldType field_{FT_INVALID};
  ArrayRef cached_beaver_[3];

  std::unique_ptr<CheetahMul> mul_prot_;

  // NOTE(juhou): make sure the lock is obtained
  void makeSureCacheSize(FieldType, int64_t numel);

  explicit CheetahMulState(std::unique_ptr<CheetahMul> mul_prot)
      : mul_prot_(std::move(mul_prot)) {}

 public:
  static constexpr char kBindName[] = "CheetahMul";

  explicit CheetahMulState(const std::shared_ptr<yacl::link::Context>& lctx) {
    mul_prot_ = std::make_unique<CheetahMul>(lctx);
  }

  ~CheetahMulState() override = default;

  CheetahMul* get() { return mul_prot_.get(); }

  bool hasLowCostFork() const override { return true; }

  std::unique_ptr<State> fork() override {
    auto ptr = new CheetahMulState(mul_prot_->Fork());
    return std::unique_ptr<State>(ptr);
  }

  std::array<ArrayRef, 3> TakeCachedBeaver(FieldType field, int64_t num);
};

class CheetahDotState : public State {
 private:
  std::unique_ptr<CheetahDot> dot_prot_;

  explicit CheetahDotState(std::unique_ptr<CheetahDot> dot_prot)
      : dot_prot_(std::move(dot_prot)) {}

 public:
  static constexpr char kBindName[] = "CheetahDot";

  explicit CheetahDotState(const std::shared_ptr<yacl::link::Context>& lctx) {
    dot_prot_ = std::make_unique<CheetahDot>(lctx);
  }

  ~CheetahDotState() override = default;

  CheetahDot* get() { return dot_prot_.get(); }

  bool hasLowCostFork() const override { return true; }

  std::unique_ptr<State> fork() override {
    auto ptr = new CheetahDotState(dot_prot_->Fork());
    return std::unique_ptr<State>(ptr);
  }
};

class CheetahOTState : public State {
 private:
  using ProtPtr = std::shared_ptr<BasicOTProtocols>;

  std::shared_mutex lock_;
  std::map<size_t, ProtPtr> basic_ot_prot_;

 public:
  static constexpr char kBindName[] = "CheetahOT";
  static constexpr size_t kParallel = 16;

  explicit CheetahOTState() {}

  ~CheetahOTState() override = default;

  size_t parallel_size() const { return kParallel; }

  void LazyInit(Communicator* comm, size_t idx = 0) {
    SPU_ENFORCE(idx < parallel_size());
    {
      std::shared_lock<std::shared_mutex> guard(lock_);
      auto kv = basic_ot_prot_.find(idx);
      if (kv != basic_ot_prot_.end()) {
        return;
      }
    }
    // double check
    std::unique_lock<std::shared_mutex> guard(lock_);
    auto kv = basic_ot_prot_.find(idx);
    if (kv != basic_ot_prot_.end()) {
      return;
    }

    // NOTE: create a seperated link for OT
    auto _comm = std::make_shared<Communicator>(comm->lctx()->Spawn());
    auto ptr = std::make_shared<BasicOTProtocols>(std::move(_comm));
    basic_ot_prot_.emplace(idx, ptr);
  }

  std::shared_ptr<BasicOTProtocols> get(size_t idx = 0) {
    SPU_ENFORCE(idx < parallel_size());
    std::shared_lock<std::shared_mutex> guard(lock_);
    auto kv = basic_ot_prot_.find(idx);
    SPU_ENFORCE(kv != basic_ot_prot_.end(), "call LazyInit first");
    return kv->second;
  }
};

}  // namespace spu::mpc::cheetah
