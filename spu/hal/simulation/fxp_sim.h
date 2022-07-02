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

#include <vector>

#include "spu/spu.pb.h"

namespace spu::hal::simulation {

#define SIM_DECLARE_UNARY(FNAME) \
  std::vector<float> FNAME##_sim(const std::vector<float>& x, Visibility x_vis);

SIM_DECLARE_UNARY(exp)
SIM_DECLARE_UNARY(log)
SIM_DECLARE_UNARY(reciprocal)
SIM_DECLARE_UNARY(logistic)

#define SIM_DECLARE_BINARY(FNAME)                    \
  std::vector<float> FNAME##_sim(                    \
      const std::vector<float>& x, Visibility x_vis, \
      const std::vector<float>& y, Visibility y_vis);

SIM_DECLARE_BINARY(div)

}  // namespace spu::hal::simulation
