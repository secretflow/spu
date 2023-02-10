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

#include <cstddef>

#include "spdlog/spdlog.h"

#include "libspu/core/parallel_utils.h"
#include "libspu/core/prelude.h"

#define EIGEN_HAS_OPENMP

#include "Eigen/Core"

namespace spu::mpc::linalg {

namespace detail {

void setEigenParallelLevel(int64_t expected_threads);

}  // namespace detail

#define EIGEN_BINARY_FCN(NAME, OP)                                   \
  template <typename T>                                              \
  void NAME(int64_t numel, const T* A, int64_t stride_A, const T* B, \
            int64_t stride_B, T* C, int64_t stride_C) {              \
    pforeach(0, numel, [&](int64_t idx) {                            \
      C[idx * stride_C] = A[idx * stride_A] OP B[idx * stride_B];    \
    });                                                              \
  }

EIGEN_BINARY_FCN(mul, *)
EIGEN_BINARY_FCN(add, +)
EIGEN_BINARY_FCN(sub, -)
EIGEN_BINARY_FCN(equal, ==)
EIGEN_BINARY_FCN(bitwise_and, &)
EIGEN_BINARY_FCN(bitwise_xor, ^)

#undef EIGEN_BINARY_FCN

#define EIGEN_UNARY_FCN_WITH_OP(NAME, OP)                                     \
  template <typename T>                                                       \
  void NAME(int64_t numel, const T* A, int64_t stride_A, T* C,                \
            int64_t stride_C) {                                               \
    pforeach(0, numel,                                                        \
             [&](int64_t idx) { C[idx * stride_C] = OP A[idx * stride_A]; }); \
  }

EIGEN_UNARY_FCN_WITH_OP(bitwise_not, ~)
EIGEN_UNARY_FCN_WITH_OP(negate, -)

#undef EIGEN_UNARY_FCN_WITH_OP

#define EIGEN_SHIFT_FCN_WITH_OP(NAME, OP)                      \
  template <typename T>                                        \
  void NAME(int64_t numel, const T* A, int64_t stride_A, T* C, \
            int64_t stride_C, int64_t bits) {                  \
    pforeach(0, numel, [&](int64_t idx) {                      \
      C[idx * stride_C] = A[idx * stride_A] OP bits;           \
    });                                                        \
  }

EIGEN_SHIFT_FCN_WITH_OP(rshift, >>)
EIGEN_SHIFT_FCN_WITH_OP(lshift, <<)

#undef EIGEN_SHIFT_FCN_WITH_OP

template <typename T, typename OP>
void unaryWithOp(int64_t numel, const T* A, int64_t stride_A, T* C,
                 int64_t stride_C, const OP& op) {
  pforeach(0, numel, [&](int64_t idx) {  //
    C[idx * stride_C] = op(A[idx * stride_A]);
  });
}

template <typename T>
void assign(int64_t numel, const T* A, int64_t stride_A, T* C,
            int64_t stride_C) {
  pforeach(0, numel, [&](int64_t idx) {  //
    C[idx * stride_C] = A[idx * stride_A];
  });
}

template <typename T>
void setConstantValue(int64_t numel, T* A, int64_t stride_A, T value) {
  pforeach(0, numel, [&](int64_t idx) {  //
    A[idx * stride_A] = value;
  });
}

template <typename T>
void select(int64_t numel, const uint8_t* cond, const T* on_true,
            int64_t on_true_stride, const T* on_false, int64_t on_false_stride,
            T* ret, int64_t ret_stride) {
  pforeach(0, numel, [&](int64_t idx) {
    ret[idx * ret_stride] = (cond[idx] ? on_true[idx * on_true_stride]
                                       : on_false[idx * on_false_stride]);
  });
}

/**
 * @brief C := op( A )*op( B )
 *
 * @tparam T Type of A, B, C
 * @param M   Number of rows in A
 * @param N   Number of columns in B
 * @param K   Number of columns in A and number of rows in B
 * @param A   Pointer to A
 * @param LDA Leading dimension stride of A
 * @param IDA Inner dimension stride of A
 * @param B   Pointer to B
 * @param LDB Leading dimension stride of B
 * @param IDB Inner dimension stride of B
 * @param C   Pointer to C
 * @param LDC Leading dimension stride of C
 * @param IDC Inner dimension stride of C
 */
template <typename T>
void matmul(int64_t M, int64_t N, int64_t K, const T* A, int64_t LDA,
            int64_t IDA, const T* B, int64_t LDB, int64_t IDB, T* C,
            int64_t LDC, int64_t IDC) {
  using StrideT = Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>;
  using MapMatrixConstT = Eigen::Map<
      const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>,
      Eigen::Unaligned, StrideT>;
  using MapMatrixT = Eigen::Map<
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>,
      Eigen::Unaligned, StrideT>;

  MapMatrixConstT a(A, M, K, StrideT(LDA, IDA));
  MapMatrixConstT b(B, K, N, StrideT(LDB, IDB));
  MapMatrixT c(C, M, N, StrideT(LDC, IDC));

  // if (M == 1) {
  //   // GEMV case 1*K * K*N -> 1*N
  //   auto work_load_size = computeTaskSize(N);
  //   yacl::parallel_for(0, N, work_load_size, [&](int64_t begin, int64_t end)
  //   {
  //     auto block_size = end - begin;
  //     c.block(0, begin, 1, block_size) =
  //         a.row(0) * b.block(0, begin, K, block_size);
  //   });
  //   return;
  // } else if (N == 1) {
  //   // GEMV case M*K * K*1 -> M*1
  //   auto work_load_size = computeTaskSize(M);
  //   yacl::parallel_for(0, M, work_load_size, [&](int64_t begin, int64_t end)
  //   {
  //     auto block_size = end - begin;
  //     c.block(begin, 0, block_size, 1) =
  //         a.block(begin, 0, block_size, K) * b.col(0);
  //   });
  //   return;
  // }

  // If we don't limit # threads, eigen may overloading omp tasks (especially
  // under relative small tasks, MLP for example)
  //
  // FIXME: Investigate what can happen once we support ILP
  //        The performance is extremely bad when multi-process all tries to use
  //        num_cores.
  // auto expected_num_threads = std::max((M * K + kMinTaskSize) / kMinTaskSize,
  //                                     (N * K + kMinTaskSize) / kMinTaskSize);
  detail::setEigenParallelLevel(2);

  c.noalias() = a * b;
}

}  // namespace spu::mpc::linalg
