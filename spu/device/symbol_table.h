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

#include <string>
#include <unordered_map>

#include "spu/hal/value.h"

#include "spu/device/device.pb.h"

namespace spu::device {

class SymbolTable {
  std::unordered_map<std::string, hal::Value> data_;

public:
  void setVar(const std::string &name, const hal::Value &val);
  hal::Value getVar(const std::string &name) const;
  bool hasVar(const std::string &name) const;
  void delVar(const std::string &name);
  void clear();

  SymbolTableProto toProto() const;
  static SymbolTable fromProto(const SymbolTableProto &proto);
};

} // namespace spu::device
