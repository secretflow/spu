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

#include "libspu/kernel/hal/test_util.h"

#include "libspu/core/encoding.h"

namespace spu::kernel::hal::test {

HalContext makeRefHalContext(RuntimeConfig config) {
  // Note: we are testing the encoding and approximation method, not the
  // protocol itself, so ref2k is enough.
  config.set_enable_action_trace(false);
  HalContext ctx(config,  //
                 nullptr  // link context.
  );
  return ctx;
}

HalContext makeRefHalContext() {
  RuntimeConfig config;
  config.set_protocol(ProtocolKind::REF2K);
  config.set_field(FieldType::FM64);
  config.set_sigmoid_mode(RuntimeConfig::SIGMOID_REAL);
  return makeRefHalContext(config);
}

Value makeValue(HalContext* ctx, PtBufferView init, Visibility vtype,
                DataType dtype, ShapeView shape) {
  if (dtype == DT_INVALID) {
    dtype = getEncodeType(init.pt_type);
  }
  auto res = constant(ctx, init, dtype, shape);
  switch (vtype) {
    case VIS_PUBLIC:
      return res;
    case VIS_SECRET:
      return _p2s(ctx, res).setDtype(res.dtype());
    default:
      SPU_THROW("not supported vtype={}", vtype);
  }
}

}  // namespace spu::kernel::hal::test
