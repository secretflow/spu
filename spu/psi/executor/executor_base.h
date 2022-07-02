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

#include <any>
#include <memory>
#include <string>
#include <vector>

#include "yasl/link/link.h"

namespace spu::psi {

struct PsiReport {
  // 交集大小
  int64_t intersection_count;
  // 原始数据集大小
  int64_t original_count;
};

struct PsiExecBaseOptions {
  // Provides the link for the rank world.
  std::shared_ptr<yasl::link::Context> link_ctx;

  std::string in_path;
  std::vector<std::string> field_names;

  std::string out_path;
  bool should_sort;
};

class PsiExecutorBase {
 public:
  PsiExecutorBase(PsiExecBaseOptions options);

  virtual ~PsiExecutorBase() = default;

  void Init();

  void Run(PsiReport* report);

  void Stop();

 protected:
  virtual void OnInit() = 0;

  virtual void OnRun(std::vector<unsigned>* indices) = 0;

  virtual void OnStop() = 0;

 protected:
  PsiExecBaseOptions options_;

  size_t input_data_count_;
};

}  // namespace spu::psi
