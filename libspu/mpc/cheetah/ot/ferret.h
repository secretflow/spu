//
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

#include <memory>

#include "absl/types/span.h"
#include "yacl/base/int128.h"

#include "libspu/mpc/common/communicator.h"

namespace spu::mpc::cheetah {

class FerretOT {
 private:
  struct Impl;
  std::shared_ptr<Impl> impl_;

 public:
  FerretOT(std::shared_ptr<Communicator> conn, bool is_sender);

  ~FerretOT();

  int Rank() const;

  void Flush();

  // One-of-N OT where msg_array is a Nxn array.
  // choice \in [0, N-1]
  void SendCMCC(absl::Span<const uint8_t> msg_array, size_t N,
                size_t bit_width = 0);
  void SendCMCC(absl::Span<const uint32_t> msg_array, size_t N,
                size_t bit_width = 0);
  void SendCMCC(absl::Span<const uint64_t> msg_array, size_t N,
                size_t bit_width = 0);
  void SendCMCC(absl::Span<const uint128_t> msg_array, size_t N,
                size_t bit_width = 0);

  void RecvCMCC(absl::Span<const uint8_t> one_oo_N_choices, size_t N,
                absl::Span<uint8_t> output, size_t bit_width = 0);
  void RecvCMCC(absl::Span<const uint8_t> one_oo_N_choices, size_t N,
                absl::Span<uint32_t> output, size_t bit_width = 0);
  void RecvCMCC(absl::Span<const uint8_t> one_oo_N_choices, size_t N,
                absl::Span<uint64_t> output, size_t bit_width = 0);
  void RecvCMCC(absl::Span<const uint8_t> one_oo_N_choices, size_t N,
                absl::Span<uint128_t> output, size_t bit_width = 0);

  // Random Message Random Choice
  void SendRMRC(absl::Span<uint8_t> output0, absl::Span<uint8_t> output1,
                size_t bit_width = 0);
  void SendRMRC(absl::Span<uint32_t> output0, absl::Span<uint32_t> output1,
                size_t bit_width = 0);
  void SendRMRC(absl::Span<uint64_t> output0, absl::Span<uint64_t> output1,
                size_t bit_width = 0);
  void SendRMRC(absl::Span<uint128_t> output0, absl::Span<uint128_t> output1,
                size_t bit_width = 0);

  void RecvRMRC(absl::Span<uint8_t> binary_choices, absl::Span<uint8_t> output,
                size_t bit_width = 0);
  void RecvRMRC(absl::Span<uint8_t> binary_choices, absl::Span<uint32_t> output,
                size_t bit_width = 0);
  void RecvRMRC(absl::Span<uint8_t> binary_choices, absl::Span<uint64_t> output,
                size_t bit_width = 0);
  void RecvRMRC(absl::Span<uint8_t> binary_choices,
                absl::Span<uint128_t> output, size_t bit_width = 0);

  // correlated additive message, chosen choice
  // (x, x + corr * choice) <- (corr, choice)
  // Can use bit_width to further indicate output ring. `bit_width = 0` means to
  // use the full range.
  void SendCAMCC(absl::Span<const uint8_t> corr, absl::Span<uint8_t> output,
                 int bit_width = 0);
  void SendCAMCC(absl::Span<const uint32_t> corr, absl::Span<uint32_t> output,
                 int bit_width = 0);
  void SendCAMCC(absl::Span<const uint64_t> corr, absl::Span<uint64_t> output,
                 int bit_width = 0);
  void SendCAMCC(absl::Span<const uint128_t> corr, absl::Span<uint128_t> output,
                 int bit_width = 0);

  void RecvCAMCC(absl::Span<const uint8_t> binary_choices,
                 absl::Span<uint8_t> output, int bit_width = 0);
  void RecvCAMCC(absl::Span<const uint8_t> binary_choices,
                 absl::Span<uint32_t> output, int bit_width = 0);
  void RecvCAMCC(absl::Span<const uint8_t> binary_choices,
                 absl::Span<uint64_t> output, int bit_width = 0);
  void RecvCAMCC(absl::Span<const uint8_t> binary_choices,
                 absl::Span<uint128_t> output, int bit_width = 0);
};

}  // namespace spu::mpc::cheetah
