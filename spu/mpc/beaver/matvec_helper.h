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

#include "seal/batchencoder.h"
#include "seal/context.h"

#include "spu/core/array_ref.h"
#include "spu/mpc/util/seal_help.h"

namespace spu::mpc {

class MatVecHelper {
 public:
  // Meta information defines a submatrix view.
  struct MatViewMeta {
    bool is_transposed;
    size_t num_rows;
    size_t num_cols;

    size_t row_start;   // row_start \in [0, num_rows)
    size_t row_extent;  // row_extent \in [1, min(num_slots, num_rows)]

    size_t col_start;   // col_start \in [0, num_cols)
    size_t col_extent;  // col_extent \in [1, min(num_slots, num_cols)]
  };

  explicit MatVecHelper(size_t num_slots, ArrayRef mat, MatViewMeta meta);

  // Return a diagnoal vector from the specified sub-matrix.
  // Rotate the vector right-hand-side when `rhs_rot > 0` (rhs_rot \in [0,
  // num_slots/2)) Basically, the number of diagnoals is Next2Pow(min(nrows,
  // ncols)) Example
  //     [[a0, a1, a2, a3],
  //      [b0, b1, b2, b3],
  //      [c0, c1, c2, c3],
  //      [d0, d1, d2, d3]]
  //   The 0-th diagnoal is [a0, b1, c2, d3]
  //   The 1-th diagnoal is [a1, b2, c3, d0]
  //   The 2-th diagnoal is [a2, b3, c0, d1]
  //   The 3-th diagnoal is [a3, b0, c1, d2]
  //   RHS rotate the 1-th diagnoal by 1-unit gives [d0, a1, b2, c0] when
  //   num_slots > 4
  //
  //   However, RHS rotate the 1-th diagnoal by 1-unit gives [b2, a1, d0, c0]
  //   for num_slots = 4 This is an optimization used to handle the case
  //   num_diagnoals > num_slots / 2.
  //
  // For non-square matrix, we will pack multiple columns/rows into one diagnoal
  // For example (num_slots >= 4):
  //     [[a0, a1, a2, a3],
  //      [b0, b1, b2, b3]]
  //   The 0-th diagnoal is [a0, b1, a2, b3]
  //   The 1-th diagnoal is [a1, b2, a3, b0]
  //
  // For example:
  //     [[a0, a1],
  //      [b0, b1],
  //      [c0, c1],
  //      [d0, d1]]
  //   The 0-th diagnoal is [a0, b1, c0, d1]
  //   The 1-th diagnoal is [a1, b0, c1, d0]
  ArrayRef GetRotatedDiagnoal(size_t diag_index, size_t rhs_rot = 0) const;

  bool IsMetaValid(const MatViewMeta& meta) const {
    if (meta.num_rows <= 0 || meta.num_cols <= 0) {
      return false;
    }
    if (meta.row_extent < 1 ||
        meta.row_start + meta.row_extent > meta.num_rows) {
      return false;
    }
    if (meta.col_extent < 1 ||
        meta.col_start + meta.col_extent > meta.num_cols) {
      return false;
    }
    if (NumDiagnoals() > num_slots_ / 2) {
      return false;
    }

    return true;
  }

  size_t NumDiagnoals() const {
    auto row = Next2Pow(meta_.row_extent);
    auto col = Next2Pow(meta_.col_extent);
    return std::min<size_t>(row, col);
  }

  size_t row_extent() const { return meta_.row_extent; }

  size_t col_extent() const { return meta_.col_extent; }

 private:
  size_t num_slots_;
  ArrayRef mat_;
  MatViewMeta meta_;
};

}  // namespace spu::mpc
