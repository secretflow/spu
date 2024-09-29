// Copyright 2024 Ant Group Co., Ltd.
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

#include "llvm/Support/Regex.h"
#include "mlir/Bytecode/BytecodeImplementation.h"
#include "mlir/Bytecode/Encoding.h"

namespace mlir::spu::ring {

struct RingDialectVersion : public mlir::DialectVersion {
  // major, minor, patch \in \{0, 1, ..., 999\}
  static constexpr uint64_t kSubVersionCarry = 1000;

  uint64_t major = 0;
  uint64_t minor = 0;
  uint64_t patch = 0;

  constexpr RingDialectVersion() = default;

  constexpr RingDialectVersion(uint64_t major_, uint64_t minor_,
                               uint64_t patch_) noexcept
      : major(major_), minor(minor_), patch(patch_) {}

  explicit RingDialectVersion(llvm::StringRef version) noexcept {
    llvm::SmallVector<llvm::StringRef> matches;
    if (!llvm::Regex("^([0-9]+)\\.([0-9]+)\\.([0-9]+)")
             .match(version, &matches)) {
      return;
    }

    matches[1].getAsInteger(/*radix=*/10, major);
    matches[2].getAsInteger(/*radix=*/10, minor);
    matches[3].getAsInteger(/*radix=*/10, patch);
  }

  constexpr uint64_t getVersionValue() const noexcept {
    return (major * kSubVersionCarry * kSubVersionCarry +
            minor * kSubVersionCarry + patch);
  }

  bool isValid() const noexcept {
    // 0.0.1 ~ 999.999.999
    static auto kMinValidVersionValue =
        RingDialectVersion(0, 0, 1).getVersionValue();
    static auto kMaxValidVersionValue =
        RingDialectVersion(kSubVersionCarry - 1, kSubVersionCarry - 1,
                           kSubVersionCarry - 1)
            .getVersionValue();

    auto value = getVersionValue();
    return (kMinValidVersionValue <= value && value <= kMaxValidVersionValue);
  }

  bool operator<(const RingDialectVersion &other) const noexcept {
    return (getVersionValue() < other.getVersionValue());
  }

  static bytecode::BytecodeVersion getBytecodeVersion() noexcept {
    return bytecode::BytecodeVersion::kVersion;
  }
};
}  // namespace mlir::spu::ring
