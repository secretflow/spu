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

#include "spu/psi/cryptor/ecdh_oprf/ecdh_oprf.h"

#include "yasl/utils/parallel.h"

namespace spu {

std::vector<std::string> IEcdhOprfServer::Evaluate(
    absl::Span<const std::string> blinded_elements) const {
  std::vector<std::string> evaluated_elements(blinded_elements.size());

  yasl::parallel_for(
      0, blinded_elements.size(), 1, [&](int64_t begin, int64_t end) {
        for (int64_t idx = begin; idx < end; ++idx) {
          evaluated_elements[idx] = Evaluate(blinded_elements[idx]);
        }
      });

  return evaluated_elements;
}

std::vector<std::string> IEcdhOprfServer::FullEvaluate(
    absl::Span<const std::string> input) const {
  std::vector<std::string> output(input.size());

  yasl::parallel_for(0, input.size(), 1, [&](int64_t begin, int64_t end) {
    for (int64_t idx = begin; idx < end; ++idx) {
      output[idx] = FullEvaluate(input[idx]);
    }
  });

  return output;
}

std::vector<std::string> IEcdhOprfClient::Blind(
    absl::Span<const std::string> input) const {
  std::vector<std::string> blinded_elements(input.size());

  yasl::parallel_for(0, input.size(), 1, [&](int64_t begin, int64_t end) {
    for (int64_t idx = begin; idx < end; ++idx) {
      blinded_elements[idx] = Blind(input[idx]);
    }
  });

  return blinded_elements;
}

std::vector<std::string> IEcdhOprfClient::Finalize(
    absl::Span<const std::string> items,
    absl::Span<const std::string> evaluated_elements) const {
  std::vector<std::string> output(evaluated_elements.size());

  yasl::parallel_for(
      0, evaluated_elements.size(), 1, [&](int64_t begin, int64_t end) {
        for (int64_t idx = begin; idx < end; ++idx) {
          output[idx] = Finalize(items[idx], evaluated_elements[idx]);
        }
      });

  return output;
}
}  // namespace spu