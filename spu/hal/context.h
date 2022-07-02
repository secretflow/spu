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
#include <random>

#include "yasl/link/link.h"

#include "spu/core/profile.h"
#include "spu/mpc/object.h"

#include "spu/spu.pb.h"

namespace spu {

// The hal evaluation context for all spu operators.
class HalContext final : public ProfilingContext {
  const RuntimeConfig rt_config_;

  const std::shared_ptr<yasl::link::Context> lctx_;

  std::unique_ptr<mpc::Object> prot_;

  std::default_random_engine rand_engine_;

 public:
  explicit HalContext(RuntimeConfig config,
                      std::shared_ptr<yasl::link::Context> lctx);

  HalContext(const HalContext& other) = delete;
  HalContext& operator=(const HalContext& other) = delete;

  HalContext(HalContext&& other) = default;

  //
  const std::shared_ptr<yasl::link::Context>& lctx() const { return lctx_; }

  mpc::Object* prot() const { return prot_.get(); }

  // Return current working fixed point fractional bits.
  size_t getFxpBits() const { return getDefaultFxpBits(rt_config_); }

  // Return current working field of MPC engine.
  FieldType getField() const { return rt_config_.field(); }

  // Return current working runtime config.
  const RuntimeConfig& rt_config() const { return rt_config_; }

  //
  std::default_random_engine& rand_engine() { return rand_engine_; }
};

#define SPU_TRACE_HLO(...) __TRACE_OP("hlo", __func__, __VA_ARGS__)

#define SPU_TRACE_HAL(...) __TRACE_OP("hal", __func__, __VA_ARGS__)
#define SPU_PROFILE_OP(...) __PROFILE_OP("hal", __func__, __VA_ARGS__)

}  // namespace spu
