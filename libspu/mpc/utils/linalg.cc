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

#include "libspu/mpc/utils/linalg.h"

namespace spu::mpc::linalg::detail {

void setEigenParallelLevel(int64_t expected_threads) {
  auto nproc = std::min(getNumberOfProc(), static_cast<int>(expected_threads));
  Eigen::setNbThreads(nproc);
}

}  // namespace spu::mpc::linalg::detail