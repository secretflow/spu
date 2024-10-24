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
#include <string>
#include <vector>

#include "absl/strings/string_view.h"

namespace spu::psi {

// for big num
std::string EvalPolynomial(const std::vector<absl::string_view> &coeff,
                           absl::string_view X, std::string_view p);

std::string EvalPolynomial(const std::vector<std::string> &coeff,
                           absl::string_view X, std::string_view p);

std::vector<std::string> EvalPolynomial(
    const std::vector<absl::string_view> &coeff,
    const std::vector<absl::string_view> &poly_x, std::string_view p_str);

std::vector<std::string> EvalPolynomial(
    const std::vector<std::string> &coeff,
    const std::vector<absl::string_view> &poly_x, std::string_view p_str);

std::vector<std::string> InterpolatePolynomial(
    const std::vector<absl::string_view> &poly_x,
    const std::vector<absl::string_view> &poly_y, std::string_view p_str);

}  // namespace spu::psi
