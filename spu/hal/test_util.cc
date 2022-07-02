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

#include "spu/hal/test_util.h"

namespace spu::hal::test {

HalContext makeRefHalContext(RuntimeConfig config) {
  // Note: we are testing the encoding and approxmation method, not the protocol
  // itself, so ref2k is enough.
  config.set_enable_action_trace(true);
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

}  // namespace spu::hal::test
