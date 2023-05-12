// Copyright 2022 Ant Group Co., Ltd.
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

#include <cstdint>

namespace spu {

extern bool hasAVX2();
extern bool hasBMI2();
extern bool hasAVX512ifma();

// bmi2 wrapper
uint64_t pdep_u64(uint64_t a, uint64_t b);
uint64_t pext_u64(uint64_t a, uint64_t b);

}  // namespace spu
