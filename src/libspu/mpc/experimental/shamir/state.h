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

#include "yacl/crypto/tools/prg.h"

#include "libspu/core/context.h"
#include "libspu/core/object.h"
#include "libspu/mpc/utils/gfmp.h"

#include "libspu/spu.pb.h"

#define EIGEN_HAS_OPENMP

#include "Eigen/Core"

namespace spu::mpc::shamir {

template <typename T>
using GfmpMatrix =
    Eigen::Matrix<Gfmp<T>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

class ShamirPrecomputedState : public State {
 private:
  size_t world_size_;
  size_t threshold_;
  // FieldType field{FT_INVALID};
  GfmpMatrix<uint32_t> Vandermonde_n_by_n_minus_t_32;
  GfmpMatrix<uint64_t> Vandermonde_n_by_n_minus_t_64;
  GfmpMatrix<uint128_t> Vandermonde_n_by_n_minus_t_128;
  std::vector<uint32_t> reconstruct_t_32;
  std::vector<uint64_t> reconstruct_t_64;
  std::vector<uint128_t> reconstruct_t_128;
  std::vector<uint32_t> reconstruct_2t_32;
  std::vector<uint64_t> reconstruct_2t_64;
  std::vector<uint128_t> reconstruct_2t_128;

 public:
  static constexpr const char* kBindName() { return "ShamirPrecompute"; }

  explicit ShamirPrecomputedState(size_t _world_size, size_t _threshold);

  ~ShamirPrecomputedState() override = default;

  std::unique_ptr<State> fork() override;

  template <typename T,
            std::enable_if_t<
                yacl::crypto::IsSupportedMersennePrimeContainerType<T>::value,
                bool> = true>
  GfmpMatrix<T> get_vandermonde() {
    if constexpr (std::is_same_v<T, uint32_t>) {
      return Vandermonde_n_by_n_minus_t_32;
    } else if constexpr (std::is_same_v<T, uint64_t>) {
      return Vandermonde_n_by_n_minus_t_64;
    } else if constexpr (std::is_same_v<T, uint128_t>) {
      return Vandermonde_n_by_n_minus_t_128;
    } else {
      SPU_THROW("Type T is not supported");
    }
  }

  template <typename T,
            std::enable_if_t<
                yacl::crypto::IsSupportedMersennePrimeContainerType<T>::value,
                bool> = true>
  std::vector<T> get_recontruction(size_t n_shares) {
    if (n_shares == threshold_ + 1) {
      if constexpr (std::is_same_v<T, uint32_t>) {
        return reconstruct_t_32;
      } else if constexpr (std::is_same_v<T, uint64_t>) {
        return reconstruct_t_64;
      } else if constexpr (std::is_same_v<T, uint128_t>) {
        return reconstruct_t_128;
      } else {
        SPU_THROW("Type T is not supported");
      }
    } else if (n_shares == (threshold_ << 1) + 1) {
      if constexpr (std::is_same_v<T, uint32_t>) {
        return reconstruct_2t_32;
      } else if constexpr (std::is_same_v<T, uint64_t>) {
        return reconstruct_2t_64;
      } else if constexpr (std::is_same_v<T, uint128_t>) {
        return reconstruct_2t_128;
      } else {
        SPU_THROW("Type T is not supported");
      }
    } else {
      SPU_THROW("Degree is not supported");
    }
  }
};

}  // namespace spu::mpc::shamir