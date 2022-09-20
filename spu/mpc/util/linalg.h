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

#define EIGEN_USE_THREADS
#include "unsupported/Eigen/CXX11/Tensor"

namespace spu::mpc::linalg {

Eigen::ThreadPoolDevice* getEigenThreadPoolDevice();

#define EIGEN_BINARY_FCN(NAME, OP)                                           \
  template <typename T>                                                      \
  void NAME(int64_t numel, const T* A, int64_t stride_A, const T* B,         \
            int64_t stride_B, T* C, int64_t stride_C) {                      \
    Eigen::TensorMap<Eigen::Tensor<const T, 1, Eigen::RowMajor>,             \
                     Eigen::Unaligned>                                       \
    a(A, numel* stride_A);                                                   \
    Eigen::TensorMap<Eigen::Tensor<const T, 1, Eigen::RowMajor>,             \
                     Eigen::Unaligned>                                       \
    b(B, numel* stride_B);                                                   \
    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>, Eigen::Unaligned> \
    c(C, numel* stride_C);                                                   \
    c.stride(Eigen::array<Eigen::DenseIndex, 1>{stride_C})                   \
        .device(*getEigenThreadPoolDevice()) =                               \
        a.stride(Eigen::array<Eigen::DenseIndex, 1>{stride_A})               \
            OP b.stride(Eigen::array<Eigen::DenseIndex, 1>{stride_B});       \
  }

EIGEN_BINARY_FCN(mul, *)
EIGEN_BINARY_FCN(add, +)
EIGEN_BINARY_FCN(sub, -)

#undef EIGEN_BINARY_FCN

#define EIGEN_BINARY_FCN_WITH_OP(NAME, OP)                                   \
  template <typename T>                                                      \
  void NAME(int64_t numel, const T* A, int64_t stride_A, const T* B,         \
            int64_t stride_B, T* C, int64_t stride_C) {                      \
    Eigen::TensorMap<Eigen::Tensor<const T, 1, Eigen::RowMajor>,             \
                     Eigen::Unaligned>                                       \
    a(A, numel* stride_A);                                                   \
    Eigen::TensorMap<Eigen::Tensor<const T, 1, Eigen::RowMajor>,             \
                     Eigen::Unaligned>                                       \
    b(B, numel* stride_B);                                                   \
    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>, Eigen::Unaligned> \
    c(C, numel* stride_C);                                                   \
                                                                             \
    auto op = [](const T& lhs, const T& rhs) {                               \
      return static_cast<T>(lhs OP rhs);                                     \
    };                                                                       \
                                                                             \
    c.stride(Eigen::array<Eigen::DenseIndex, 1>{stride_C})                   \
        .device(*getEigenThreadPoolDevice()) =                               \
        a.stride(Eigen::array<Eigen::DenseIndex, 1>{stride_A})               \
            .binaryExpr(                                                     \
                b.stride(Eigen::array<Eigen::DenseIndex, 1>{stride_B}), op); \
  }

EIGEN_BINARY_FCN_WITH_OP(equal, ==)
EIGEN_BINARY_FCN_WITH_OP(bitwise_and, &)
EIGEN_BINARY_FCN_WITH_OP(bitwise_xor, ^)

#undef EIGEN_BINARY_FCN_WITH_OP

#define EIGEN_UNARY_FCN_WITH_OP(NAME, OP)                                     \
  template <typename T>                                                       \
  void NAME(int64_t numel, const T* A, int64_t stride_A, T* C,                \
            int64_t stride_C) {                                               \
    Eigen::TensorMap<Eigen::Tensor<const T, 1, Eigen::RowMajor>,              \
                     Eigen::Unaligned>                                        \
    a(A, numel* stride_A);                                                    \
    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>, Eigen::Unaligned>  \
    c(C, numel* stride_C);                                                    \
                                                                              \
    auto op = [](const T& in) { return OP in; };                              \
                                                                              \
    c.stride(Eigen::array<Eigen::DenseIndex, 1>{stride_C})                    \
        .device(*getEigenThreadPoolDevice()) =                                \
        a.stride(Eigen::array<Eigen::DenseIndex, 1>{stride_A}).unaryExpr(op); \
  }

EIGEN_UNARY_FCN_WITH_OP(bitwise_not, ~)
EIGEN_UNARY_FCN_WITH_OP(negate, -)

#undef EIGEN_UNARY_FCN_WITH_OP

#define EIGEN_SHIFT_FCN_WITH_OP(NAME, OP)                                     \
  template <typename T>                                                       \
  void NAME(int64_t numel, const T* A, int64_t stride_A, T* C,                \
            int64_t stride_C, int64_t bits) {                                 \
    Eigen::TensorMap<Eigen::Tensor<const T, 1, Eigen::RowMajor>,              \
                     Eigen::Unaligned>                                        \
    a(A, numel* stride_A);                                                    \
    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>, Eigen::Unaligned>  \
    c(C, numel* stride_C);                                                    \
                                                                              \
    auto op = [&](const T& in) { return in OP bits; };                        \
                                                                              \
    c.stride(Eigen::array<Eigen::DenseIndex, 1>{stride_C})                    \
        .device(*getEigenThreadPoolDevice()) =                                \
        a.stride(Eigen::array<Eigen::DenseIndex, 1>{stride_A}).unaryExpr(op); \
  }

EIGEN_SHIFT_FCN_WITH_OP(rshift, >>)
EIGEN_SHIFT_FCN_WITH_OP(lshift, <<)

#undef EIGEN_SHIFT_FCN_WITH_OP

template <typename T, typename OP>
void unaryWithOp(int64_t numel, const T* A, int64_t stride_A, T* C,
                 int64_t stride_C, const OP& op) {
  Eigen::TensorMap<Eigen::Tensor<const T, 1, Eigen::RowMajor>, Eigen::Unaligned>
      a(A, numel * stride_A);
  Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>, Eigen::Unaligned> c(
      C, numel * stride_C);

  c.stride(Eigen::array<Eigen::DenseIndex, 1>{stride_C})
      .device(*getEigenThreadPoolDevice()) =
      a.stride(Eigen::array<Eigen::DenseIndex, 1>{stride_A}).unaryExpr(op);
}

template <typename T>
void assign(int64_t numel, const T* A, int64_t stride_A, T* C,
            int64_t stride_C) {
  Eigen::TensorMap<Eigen::Tensor<const T, 1, Eigen::RowMajor>, Eigen::Unaligned>
      a(A, numel * stride_A);
  Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>, Eigen::Unaligned> c(
      C, numel * stride_C);
  c.stride(Eigen::array<Eigen::DenseIndex, 1>{stride_C})
      .device(*getEigenThreadPoolDevice()) =
      a.stride(Eigen::array<Eigen::DenseIndex, 1>{stride_A});
}

template <typename T>
void setConstantValue(int64_t numel, T* A, int64_t stride_A, T value) {
  Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>, Eigen::Unaligned> a(
      A, numel * stride_A);
  a.stride(Eigen::array<Eigen::DenseIndex, 1>{stride_A})
      .setConstant(value)
      .device(*getEigenThreadPoolDevice());
}

template <typename T>
void select(int64_t numel, const uint8_t* cond, const T* on_true,
            int64_t on_true_stride, const T* on_false, int64_t on_false_stride,
            T* ret, int64_t ret_stride) {
  Eigen::TensorMap<Eigen::Tensor<const bool, 1, Eigen::RowMajor>,
                   Eigen::Unaligned>
      a(reinterpret_cast<const bool*>(cond), numel);
  Eigen::TensorMap<Eigen::Tensor<const T, 1, Eigen::RowMajor>, Eigen::Unaligned>
      t(on_true, numel * on_true_stride);
  Eigen::TensorMap<Eigen::Tensor<const T, 1, Eigen::RowMajor>, Eigen::Unaligned>
      f(on_false, numel * on_false_stride);
  Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>, Eigen::Unaligned> r(
      ret, numel * ret_stride);

  r.stride(Eigen::array<Eigen::DenseIndex, 1>{ret_stride})
      .device(*getEigenThreadPoolDevice()) =
      a.select(t.stride(Eigen::array<Eigen::DenseIndex, 1>{on_true_stride}),
               f.stride(Eigen::array<Eigen::DenseIndex, 1>{on_false_stride}));
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
  int64_t unstrided_lhs_cols = K * IDA;
  int64_t normalized_lda = LDA / unstrided_lhs_cols;
  int64_t unstrided_lhs_rows = M * normalized_lda;

  int64_t unstrided_rhs_cols = N * IDB;
  int64_t normalized_ldb = LDB / unstrided_rhs_cols;
  int64_t unstrided_rhs_rows = K * normalized_ldb;

  int64_t unstrided_ret_cols = N * IDC;
  int64_t normalized_ldc = LDC / unstrided_ret_cols;
  int64_t unstrided_ret_rows = M * normalized_ldc;

  Eigen::TensorMap<Eigen::Tensor<const T, 2, Eigen::RowMajor>, Eigen::Unaligned>
      a(A, unstrided_lhs_rows, unstrided_lhs_cols);
  Eigen::TensorMap<Eigen::Tensor<const T, 2, Eigen::RowMajor>, Eigen::Unaligned>
      b(B, unstrided_rhs_rows, unstrided_rhs_cols);
  Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>, Eigen::Unaligned> c(
      C, unstrided_ret_rows, unstrided_ret_cols);

  using DimPair = typename Eigen::Tensor<T, 2>::DimensionPair;
  const Eigen::array<DimPair, 1> dims({DimPair(1, 0)});

  c.stride(Eigen::array<Eigen::DenseIndex, 2>{normalized_ldc, IDC})
      .device(*getEigenThreadPoolDevice()) =
      a.stride(Eigen::array<Eigen::DenseIndex, 2>{normalized_lda, IDA})
          .contract(
              b.stride(Eigen::array<Eigen::DenseIndex, 2>{normalized_ldb, IDB}),
              dims);
}

}  // namespace spu::mpc::linalg
