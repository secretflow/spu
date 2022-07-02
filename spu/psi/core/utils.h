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

#include "yasl/base/buffer.h"

#include "spu/psi/core/serializable.pb.h"

namespace spu::psi::utils {

inline yasl::Buffer SerializeSize(size_t size) {
  SizeProto proto;
  proto.set_input_size(size);
  yasl::Buffer b(proto.ByteSizeLong());
  proto.SerializePartialToArray(b.data(), b.size());
  return b;
}

inline size_t DeserializeSize(const yasl::Buffer& buf) {
  SizeProto proto;
  proto.ParseFromArray(buf.data(), buf.size());
  return proto.input_size();
}

}  // namespace spu::psi::utils
