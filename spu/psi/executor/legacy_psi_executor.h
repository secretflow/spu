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

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "yasl/link/link.h"

#include "spu/psi/executor/executor_base.h"

namespace spu::psi {

inline constexpr absl::string_view kPsiProtocolEcdh2PC = "ecdh";
inline constexpr absl::string_view kPsiProtocolKkrt = "kkrt";
inline constexpr absl::string_view kPsiProtocolEcdh = "ecdh-3pc";

struct LegacyPsiOptions {
  PsiExecBaseOptions base_options;

  std::string psi_protocol;
  size_t num_bins;
  bool broadcast_result = true;
};

// TODO: refactor
class LegacyPsiExecutor : public PsiExecutorBase {
 public:
  LegacyPsiExecutor(LegacyPsiOptions options);

  ~LegacyPsiExecutor() = default;

 protected:
  void OnRun(std::vector<unsigned> *indices) override;

  void OnStop() override;

  void OnInit() override;

 private:
  std::string psi_protocol_;
  size_t num_bins_;
  bool broadcast_result_;
};

}  // namespace spu::psi
