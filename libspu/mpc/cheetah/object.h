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

#include "libspu/mpc/cheetah/arith/cheetah_dot.h"
#include "libspu/mpc/cheetah/arith/cheetah_mul.h"
#include "libspu/mpc/cheetah/ot/basic_ot_prot.h"
#include "libspu/mpc/object.h"

namespace spu::mpc::cheetah {

class CheetahMulState : public State {
  std::unique_ptr<CheetahMul> mul_prot_;

 public:
  static constexpr char kBindName[] = "CheetahMul";

  explicit CheetahMulState(const std::shared_ptr<yacl::link::Context>& lctx) {
    mul_prot_ = std::make_unique<CheetahMul>(lctx);
  }

  ~CheetahMulState() override = default;

  CheetahMul* get() { return mul_prot_.get(); }
};

class CheetahDotState : public State {
  std::unique_ptr<CheetahDot> dot_prot_;

 public:
  static constexpr char kBindName[] = "CheetahDot";

  explicit CheetahDotState(const std::shared_ptr<yacl::link::Context>& lctx) {
    dot_prot_ = std::make_unique<CheetahDot>(lctx);
  }

  ~CheetahDotState() override = default;

  CheetahDot* get() { return dot_prot_.get(); }
};

class CheetahOTState : public State {
  std::shared_ptr<BasicOTProtocols> basic_ot_prot_;

 public:
  static constexpr char kBindName[] = "CheetahOT";

  explicit CheetahOTState(std::shared_ptr<yacl::link::Context> lctx) {
    basic_ot_prot_ = std::make_shared<BasicOTProtocols>(lctx);
  }

  ~CheetahOTState() override = default;

  std::shared_ptr<BasicOTProtocols> get() { return basic_ot_prot_; }
};

}  // namespace spu::mpc::cheetah
