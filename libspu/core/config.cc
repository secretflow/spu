// Copyright 2023 Ant Group Co., Ltd.
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

#include "libspu/core/config.h"

#include "prelude.h"
#include "yacl/utils/parallel.h"

namespace spu {
namespace {

size_t defaultFxpBits(size_t field) {
  switch (field) {
    case 32: {
      return 8;
    }
    case 64: {
      return 18;
    }
    case 128: {
      return 26;
    }
    default: {
      SPU_THROW("unsupported field={}", field);
    }
  }
}

}  // namespace

void populateRuntimeConfig(RuntimeConfig& cfg) {
  // mandatory fields.
  SPU_ENFORCE(cfg.has_protocol(), "Protocol must be set");
  SPU_ENFORCE(cfg.protocol().kind() != ProtocolKind::PROT_INVALID);
  size_t field = cfg.protocol().field();
  SPU_ENFORCE(field == 32 || field == 64 || field == 128,
              "Only support 32/64/128 field now but got {}", field);
  if (cfg.max_concurrency() == 0) {
    cfg.set_max_concurrency(yacl::get_num_threads());
  }

  //
  if (cfg.fxp_fraction_bits() == 0) {
    cfg.set_fxp_fraction_bits(defaultFxpBits(field));
  }

  // inter op concurrency
  if (cfg.experimental_enable_inter_op_par()) {
    cfg.set_experimental_inter_op_concurrency(
        cfg.experimental_inter_op_concurrency() == 0
            ? 8
            : cfg.experimental_inter_op_concurrency());
  }

  // MPC related configurations
  // trunc_allow_msb_error           // by pass.
}

RuntimeConfig makeFullRuntimeConfig(const RuntimeConfig& cfg) {
  RuntimeConfig copy(cfg);
  populateRuntimeConfig(copy);
  return copy;
}

}  // namespace spu
