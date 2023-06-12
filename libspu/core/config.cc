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

namespace spu {
namespace {

size_t defaultFxpBits(FieldType field) {
  switch (field) {
    case FieldType::FM32: {
      return 8;
    }
    case FieldType::FM64: {
      return 18;
    }
    case FieldType::FM128: {
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
  SPU_ENFORCE(cfg.protocol() != ProtocolKind::PROT_INVALID);
  SPU_ENFORCE(cfg.field() != FieldType::FT_INVALID);

  //
  if (cfg.fxp_fraction_bits() == 0) {
    cfg.set_fxp_fraction_bits(defaultFxpBits(cfg.field()));
  }

  if (cfg.fxp_div_goldschmidt_iters() == 0) {
    cfg.set_fxp_div_goldschmidt_iters(2);
  }

  // fxp exponent config
  {
    if (cfg.fxp_exp_mode() == RuntimeConfig::EXP_DEFAULT) {
      cfg.set_fxp_exp_mode(RuntimeConfig::EXP_TAYLOR);
    }

    if (cfg.fxp_exp_iters() == 0) {
      cfg.set_fxp_exp_iters(8);
    }
  }

  // fxp log config
  {
    if (cfg.fxp_log_mode() == RuntimeConfig::LOG_DEFAULT) {
      cfg.set_fxp_log_mode(RuntimeConfig::LOG_PADE);
    }

    if (cfg.fxp_log_iters() == 0) {
      cfg.set_fxp_log_iters(3);
    }

    if (cfg.fxp_log_orders() == 0) {
      cfg.set_fxp_log_orders(8);
    }
  }

  if (cfg.sigmoid_mode() == RuntimeConfig::SIGMOID_DEFAULT) {
    cfg.set_sigmoid_mode(RuntimeConfig::SIGMOID_REAL);
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
