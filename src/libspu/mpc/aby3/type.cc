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

#include "libspu/mpc/aby3/type.h"

#include <mutex>

#include "magic_enum.hpp"

#include "libspu/mpc/common/pv2k.h"
namespace spu::mpc::aby3 {

void registerTypes() {
  regPV2kTypes();

  static std::once_flag flag;
  std::call_once(flag, []() {
    TypeContext::getTypeContext()
        ->addTypes<AShrTy, BShrTy, OShrTy, OPShrTy, PShrTy>();
  });
}

void BShrTy::fromString(std::string_view detail) {
  auto comma = detail.find_first_of(',');
  auto back_type_str = detail.substr(0, comma);
  auto nbits_str = detail.substr(comma + 1);
  auto back_type = magic_enum::enum_cast<PtType>(back_type_str);
  SPU_ENFORCE(back_type.has_value(), "parse failed from={}", detail);
  back_type_ = back_type.value();
  nbits_ = std::stoul(std::string(nbits_str));
}

std::string BShrTy::toString() const {
  return fmt::format("{},{}", magic_enum::enum_name(back_type_), nbits_);
}

}  // namespace spu::mpc::aby3
