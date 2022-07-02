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

#include "yasl/utils/parallel.h"

namespace spu::mpc::linalg {

template <typename IndexType, typename X, typename Y, typename Result>
void dotu_generic(IndexType n, const X* x, IndexType incX, const Y* y,
                  IndexType incY, Result& result) {
  result = Result(0);
  for (IndexType i = 0, iX = 0, iY = 0; i < n; ++i, iX += incX, iY += incY) {
    result += Result(x[iX]) * Result(y[iY]);
  }
}

template <typename IndexType, typename MA, typename VX, typename VY>
void gemv_generic(IndexType m, IndexType n, const MA* A, IndexType LDA,
                  IndexType IDA, const VX* x, IndexType incX, VY* y,
                  IndexType incY) {
  if (incX < 0) {
    x -= incX * (n - 1);
  }
  if (incY < 0) {
    y -= incY * (m - 1);
  }

  for (IndexType i = 0, iY = 0; i < m; ++i, iY += incY) {
    dotu_generic(n, A + i * LDA, IDA, x, incX, y[iY]);
  }
}

template <typename IndexType, typename MA, typename MB, typename MC>
void gemm_generic(IndexType m, IndexType n, IndexType k, const MA* A,
                  IndexType LDA, IndexType IDA, const MB* B, IndexType LDB,
                  IndexType IDB, MC* C, IndexType LDC, IndexType IDC) {
  if ((m == 0) || (n == 0)) {
    return;
  }

  // for (IndexType l = 0; l < n; ++l) {
  //   gemv_generic(m, k, A, LDA, IDA, B + l * IDB, LDB, C + l * IDC, LDC);
  // }
  yasl::parallel_for(0, n, 1, [&](size_t begin, size_t end) {
    for (IndexType l = begin; l < end; ++l) {
      gemv_generic(m, k, A, LDA, IDA, B + l * IDB, LDB, C + l * IDC, LDC);
    }
  });
}

/**
 * @brief C := op( A )*op( B )
 *
 * @tparam TA Type of A
 * @tparam TB Type of B
 * @tparam TC Type of C
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
template <typename TA, typename TB, typename TC>
void matmul(size_t M, size_t N, size_t K, const TA* A, size_t LDA, size_t IDA,
            const TB* B, size_t LDB, size_t IDB, TC* C, size_t LDC,
            size_t IDC) {
  gemm_generic(M, N, K, A, LDA, IDA, B, LDB, IDB, C, LDC, IDC);
}

}  // namespace spu::mpc::linalg
