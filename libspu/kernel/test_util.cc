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

#include "libspu/kernel/test_util.h"

#include "libspu/core/config.h"
#include "libspu/core/encoding.h"
#include "libspu/kernel/hal/constants.h"  // bad reference
#include "libspu/mpc/factory.h"

namespace spu::kernel::test {

SPUContext makeSPUContext(RuntimeConfig config,
                          const std::shared_ptr<yacl::link::Context>& lctx) {
  populateRuntimeConfig(config);

  SPUContext ctx(config, lctx);
  mpc::Factory::RegisterProtocol(&ctx, lctx);

  return ctx;
}

SPUContext makeSPUContext(ProtocolKind prot_kind, FieldType field,
                          const std::shared_ptr<yacl::link::Context>& lctx) {
  RuntimeConfig cfg;
  cfg.set_protocol(prot_kind);
  cfg.set_field(field);
  cfg.set_enable_action_trace(false);

  return makeSPUContext(cfg, lctx);
}

Value makeValue(SPUContext* ctx, PtBufferView init, Visibility vtype,
                DataType dtype, ShapeView shape) {
  if (dtype == DT_INVALID) {
    dtype = getEncodeType(init.pt_type);
  }
  auto res = hal::constant(ctx, init, dtype, shape);
  switch (vtype) {
    case VIS_PUBLIC:
      return res;
    case VIS_SECRET:
      return hal::_p2s(ctx, res).setDtype(res.dtype());
    default:
      SPU_THROW("not supported vtype={}", vtype);
  }
}

}  // namespace spu::kernel::test
