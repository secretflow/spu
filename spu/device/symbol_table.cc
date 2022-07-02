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

#include "spu/device/symbol_table.h"

namespace spu::device {

void SymbolTable::setVar(const std::string &name, const hal::Value &val) {
  data_[name] = val;
}

hal::Value SymbolTable::getVar(const std::string &name) const {
  const auto itr = data_.find(name);
  YASL_ENFORCE(itr != data_.end(), "symbol {} not found", name);
  return itr->second;
}

bool SymbolTable::hasVar(const std::string &name) const {
  return data_.find(name) != data_.end();
}

void SymbolTable::delVar(const std::string &name) { data_.erase(name); }

void SymbolTable::clear() { data_.clear(); }

SymbolTableProto SymbolTable::toProto() const {
  SymbolTableProto proto;
  for (const auto &[name, value] : data_) {
    proto.mutable_symbols()->insert({name, value.toProto()});
  }
  return proto;
}

SymbolTable SymbolTable::fromProto(const SymbolTableProto &proto) {
  SymbolTable st;
  for (const auto &[name, value_proto] : proto.symbols()) {
    st.setVar(name, hal::Value::fromProto(value_proto));
  }
  return st;
}

} // namespace spu::device
