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

// Author: Wen-jie Lu(juhou)

#include "spu/mpc/beaver/matvec_helper.h"

#include "xtensor/xview.hpp"

#include "spu/core/xt_helper.h"
#include "spu/mpc/util/ring_ops.h"
#include "spu/mpc/util/seal_help.h"

namespace spu::mpc {

MatVecHelper::MatVecHelper(size_t num_slots, ArrayRef mat, MatViewMeta meta)
    : num_slots_(num_slots), mat_(mat), meta_(meta) {
  YASL_ENFORCE(num_slots_ > 0 && IsTwoPower(num_slots_));
  YASL_ENFORCE((size_t)mat_.numel() == meta_.num_rows * meta_.num_cols);
  YASL_ENFORCE(IsMetaValid(meta_));
}

ArrayRef MatVecHelper::GetRotatedDiagnoal(size_t diag_index,
                                          size_t rhs_rot) const {
  YASL_ENFORCE(diag_index < NumDiagnoals());
  const size_t row_ext = Next2Pow(meta_.row_extent);
  const size_t col_ext = Next2Pow(meta_.col_extent);
  const size_t max_dim = std::max(row_ext, col_ext);
  const size_t min_dim = std::min(row_ext, col_ext);
  YASL_ENFORCE(max_dim <= num_slots_ && min_dim < num_slots_);

  const Type& type = mat_.eltype();
  YASL_ENFORCE(type.isa<RingTy>(), "source must be ring_type, got={}", type);
  auto field = type.as<Ring2k>()->field();
  auto out = ring_zeros(field, max_dim);

  DISPATCH_FM3264(field, "EncodeMatrixDiagnoal", [&, this]() {
    auto xmat = xt_adapt<ring2k_t>(mat_);
    auto xout = xt_mutable_adapt<ring2k_t>(out);
    xmat.reshape({meta_.num_rows, meta_.num_cols});
    auto submat = xt::view(
        xmat, xt::range(meta_.row_start, meta_.row_start + meta_.row_extent),
        xt::range(meta_.col_start, meta_.col_start + meta_.col_extent));

    auto mat_getter = [&](size_t r, size_t c) {
      if (r < meta_.row_extent && c < meta_.col_extent) {
        return submat(r, c);
      } else {
        return static_cast<ring2k_t>(0);
      }
    };

    size_t row_mask = row_ext - 1;
    size_t col_mask = col_ext - 1;
    size_t half_slot = num_slots_ / 2;

    size_t row_wrap_mask =
        (meta_.is_transposed ? half_slot - 1 : row_mask) & row_mask;
    size_t col_wrap_mask =
        (meta_.is_transposed ? col_mask : half_slot - 1) & col_mask;

    size_t row_offset = meta_.is_transposed ? diag_index : 0;
    size_t col_offset = meta_.is_transposed ? 0 : diag_index;

    if (max_dim <= half_slot) {
      for (size_t idx = 0; idx < max_dim; ++idx) {
        xout[idx] = mat_getter((idx + row_offset) & row_mask,
                               (idx + col_offset) & col_mask);
      }
    } else {
      // NOTE(juhou): initutively we handle multiple submatrix blocks
      size_t row_skip = row_ext <= half_slot ? 0 : half_slot;
      size_t col_skip = col_ext <= half_slot ? 0 : half_slot;

      for (size_t idx = 0; idx < half_slot; ++idx) {
        size_t row = (idx + row_offset) & row_wrap_mask;
        size_t col = (idx + col_offset) & col_wrap_mask;
        xout[idx] = mat_getter(row, col);
        xout[idx + half_slot] = mat_getter(row + row_skip, col + col_skip);
      }
    }

    rhs_rot %= max_dim;
    if (rhs_rot == 0) return;

    if (xout.size() < num_slots_) {
      xout = xt::roll(xout, rhs_rot, 0);
    } else {
      // NOTE(juhou): optimization for BSGS algorithm
      // RHS rotate xout[0, half_slot)
      // RHS rotate xout[half_slot, num_slots_)
      std::rotate(xout.begin(), xout.begin() + half_slot - rhs_rot,
                  xout.begin() + half_slot);
      std::rotate(xout.begin() + half_slot, xout.begin() + num_slots_ - rhs_rot,
                  xout.end());
    }

    return;
  });

  return out;
}

}  // namespace spu::mpc
