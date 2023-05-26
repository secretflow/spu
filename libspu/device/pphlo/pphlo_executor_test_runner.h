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

#include "fmt/format.h"

#include "libspu/device/test_utils.h"

#include "libspu/spu.pb.h"

namespace spu::device::pphlo::test {

class Runner {
 public:
  Runner(size_t world_size, FieldType field, ProtocolKind protocol);

  auto &getConfig() { return config_; }

  template <typename T>
  void addInput(const T &input, Visibility vis = Visibility::VIS_PUBLIC) {
    const std::string name = fmt::format("input{}", input_idx_++);
    io_->InFeed(name, input, vis);
    executable_.add_input_names(name);
  }

  std::string compileMHlo(const std::string &mhlo,
                          const std::vector<spu::Visibility> &vis);

  void run(const std::string &mlir, size_t num_output = 1);

  template <typename T>
  void verifyOutput(const T *expected, size_t idx = 0) {
    const auto &out = io_->OutFeed(fmt::format("output{}", idx));

    size_t numel = out.numel();
    const auto *in_ptr = static_cast<const T *>(out.data());

    // TODO: handle strides
    for (size_t i = 0; i < numel; ++i) {
      if constexpr (std::is_integral_v<T>) {
        EXPECT_EQ(in_ptr[i], expected[i]) << "i = " << i << "\n";
      } else {
        EXPECT_TRUE(std::abs(in_ptr[i] - expected[i]) <= 1e-2)
            << "i = " << i << " in = " << in_ptr[i]
            << " expected = " << expected[i] << "\n";
      }
    }
  }

  template <typename T, std::enable_if_t<std::is_scalar_v<T>, bool> = true>
  void verifyScalarOutput(T expected, size_t idx = 0) {
    verifyOutput(&expected, idx);
  }

 private:
  size_t world_size_;
  RuntimeConfig config_;
  std::unique_ptr<LocalIo> io_;
  size_t input_idx_{0};
  ExecutableProto executable_;
};

}  // namespace spu::device::pphlo::test
