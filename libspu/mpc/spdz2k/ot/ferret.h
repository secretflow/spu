//
// Copyright 2023 Ant Group Co., Ltd.
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
//
// This code is mostly from mpc/cheetah/ot/ferret.h, with two modifications:
// RMCC and VOLE

#pragma once

#include <memory>

#include "absl/types/span.h"
#include "yacl/base/int128.h"

#include "libspu/mpc/cheetah/ot/ferret.h"
#include "libspu/mpc/common/communicator.h"

namespace spu::mpc::spdz2k {

class FerretOT : public cheetah::FerretOT {
 private:
  struct Impl;
  std::shared_ptr<Impl> impl_;

 public:
  FerretOT(std::shared_ptr<Communicator> conn, bool is_sender,
           bool malicious = true);

  ~FerretOT();

  // VOLE, only for SPDZ2K
  // data[i] = data0[i] + a[i] * corr[i]
  // Sender: input corr, output random data0
  // Receiver: input a, output data
  template <typename T>
  void SendVole(absl::Span<const T> corr, absl::Span<T> data0);

  template <typename T>
  void RecvVole(absl::Span<const T> a, absl::Span<T> data);
};

FerretOT::FerretOT(std::shared_ptr<Communicator> conn, bool is_sender,
                   bool malicious)
    : cheetah::FerretOT::FerretOT(conn, is_sender, malicious) {}

FerretOT::~FerretOT() {}

// Refer to:
// Appendix C. Implementing Vector-OLE mod 2^l, P35
// SPDZ2k: Efficient MPC mod 2k for Dishonest Majority
// - https://eprint.iacr.org/2018/482.pdf
template <typename T>
void FerretOT::SendVole(absl::Span<const T> corr, absl::Span<T> data0) {
  SPU_ENFORCE(data0.size() == corr.size());
  size_t length = data0.size();
  constexpr size_t iters = sizeof(T) * 8 / 2;

  std::vector<T> t_data(length * iters, 0);
  std::vector<T> corrs;
  for (size_t i = 0; i < iters; ++i) {
    std::copy(corr.begin(), corr.end(), std::back_inserter(corrs));
  }

  // call parent class method
  SendCAMCC(absl::MakeSpan(corrs.data(), corrs.size()),
            absl::MakeSpan(t_data.data(), t_data.size()));
  Flush();

  for (size_t j = 0; j < length; ++j) {
    data0[j] = 0;
  }
  for (size_t i = 0; i < iters; ++i) {
    for (size_t j = 0; j < length; ++j) {
      data0[j] += t_data[i * length + j] << i;
    }
  }
}

template <typename T>
void FerretOT::RecvVole(absl::Span<const T> a, absl::Span<T> data) {
  SPU_ENFORCE(data.size() == a.size());

  size_t length = data.size();
  constexpr size_t iters = sizeof(T) * 8 / 2;

  std::vector<T> t_data(length * iters, 0);
  std::vector<uint8_t> b(length * iters, 0);

  for (size_t i = 0; i < iters; ++i) {
    for (size_t j = 0; j < length; ++j) {
      b[i * length + j] = (a[j] >> i) & 1;
    }
  }

  // call parent class method
  RecvCAMCC(absl::MakeSpan(b.data(), b.size()),
            absl::MakeSpan(t_data.data(), t_data.size()));

  for (size_t j = 0; j < length; ++j) {
    data[j] = 0;
  }
  for (size_t i = 0; i < iters; ++i) {
    for (size_t j = 0; j < length; ++j) {
      data[j] += t_data[i * length + j] << i;
    }
  }
}

}  // namespace spu::mpc::spdz2k
