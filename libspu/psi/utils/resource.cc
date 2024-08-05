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

#include "libspu/psi/utils/resource.h"

#include <fstream>
#include <string>

#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"

namespace spu::psi {

std::string ReadProcSelfStatusByKey(const std::string& key) {
  std::string ret;
  std::ifstream self_status("/proc/self/status");
  std::string line;
  while (std::getline(self_status, line)) {
    std::vector<absl::string_view> fields =
        absl::StrSplit(line, absl::ByChar(':'));
    if (fields.size() == 2 && key == absl::StripAsciiWhitespace(fields[0])) {
      ret = absl::StripAsciiWhitespace(fields[1]);
    }
  }
  return ret;
}

// only work with VmXXX in /proc/self/status
size_t ReadVMxFromProcSelfStatus(const std::string& key) {
  const std::string str_usage = ReadProcSelfStatusByKey(key);
  std::vector<absl::string_view> fields =
      absl::StrSplit(str_usage, absl::ByChar(' '));
  if (fields.size() == 2) {
    size_t ret = 0;
    SPU_ENFORCE(absl::SimpleAtoi(fields[0], &ret),
                "Fail to get {} in self status, {}", key, str_usage);
    return ret;
  }
  SPU_THROW("Fail to get {} in self status, {}", key, str_usage);
}

size_t GetPeakKbMemUsage() { return ReadVMxFromProcSelfStatus("VmHWM"); }

}  // namespace spu::psi
