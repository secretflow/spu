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

#include "libspu/mpc/cheetah/env.h"

#include <algorithm>
#include <string>

namespace spu::mpc::cheetah {
static bool IsEnvOn(const char *name) {
  const char *str = std::getenv(name);
  if (str == nullptr) {
    return false;
  }

  std::string s(str);
  // to lower case
  std::transform(s.begin(), s.end(), s.begin(),
                 [](auto c) { return std::tolower(c); });
  return s == "1" or s == "on";
}

bool TestEnvFlag(EnvFlag g) {
  switch (g) {
    case EnvFlag::SPU_CTH_ENABLE_EMP_OT:
      return IsEnvOn("SPU_CTH_ENABLE_EMP_OT");
    default:
      return false;
  }
}

}  // namespace spu::mpc::cheetah
