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
#include <iterator>
#include <unordered_set>

#include "Eigen/Sparse"

#include "libspu/core/context.h"
#include "libspu/core/value.h"

namespace squirrel {

// A simple binary matrix from std::unordered_set.
// In this XGB demo, we define the binary matrix as num_sample x num_buckets.
// Thus we need to iterate the binary matrix row-by-row.
//
// We can also use Eigen::SparseMatrix<Eigen::RowMajor, XX>.
struct StlSparseMatrix {
  using SparseRow = std::unordered_set<size_t>;

  static StlSparseMatrix Initialize(const std::vector<SparseRow>& data,
                                    size_t cols) {
    for (const auto& row : data) {
      SPU_ENFORCE(std::all_of(row.cbegin(), row.cend(),
                              [&](size_t c) { return c < cols; }));
    }

    StlSparseMatrix mat;
    mat.rows_data_ = data;
    mat.cols_ = cols;
    return mat;
  }

  int64_t rows() const { return rows_data_.size(); }

  int64_t cols() const { return cols_; }

  auto iterate_row_begin(size_t row) const {
    return rows_data_.at(row).cbegin();
  }

  auto iterate_row_end(size_t row) const { return rows_data_.at(row).cend(); }

  int64_t cols_ = 0;
  std::vector<SparseRow> rows_data_;
};

// REF: Squirrel: A Scalable Secure Two-Party Computation Framework for Training
// Gradient Boosting Decision Tree
//
// https://eprint.iacr.org/2023/527
//
// Inputs:
//   Sender: v0 \in Zk^{n}
//     Recv: v1 \n Zk^{n}, M \in {0,1}^{mxn}
//
// Outputs:
//   Sender: z0 \in Zk^{m}
//     Recv: z1 \in Zk^{m}
// such that z0 + z1 = M * (v0 + v1) mod Zk
class BinMatVecProtocol {
 public:
  BinMatVecProtocol(size_t ring_bitwidth,
                    std::shared_ptr<yacl::link::Context> conn);

  ~BinMatVecProtocol() = default;

  // Compute BinMat * vec
  // This function is called by the vector holder.
  //
  // NOTE: throw error if vec.numel() != dim_in
  spu::NdArrayRef Send(const spu::NdArrayRef& vec_in, int64_t dim_out,
                       int64_t dim_in);

  // Compute BinMat * vec
  // This function is called by the matrix holder.
  // TODO(lwj): maybe we can also have a function that only take matrix as
  // input.
  //
  // NOTE: throw error if mat.nrows() != dim_out or mat.ncols() != dim_in or
  // vec.size() != dim_in
  spu::NdArrayRef Recv(const spu::NdArrayRef& vec_in, int64_t dim_out,
                       int64_t dim_in, const StlSparseMatrix& prv_bin_mat) {
    std::vector<uint8_t> dummy;
    return Recv(vec_in, dim_out, dim_in, prv_bin_mat,
                absl::MakeConstSpan(dummy));
  }

  // Compute (BinMat * diag(indicator)) * vec
  // That is, indicator[i] = 0 means the i-th row of the BinMat is ignored.
  // During the XGB training process, we need to exclude some data samples.
  // It seems better to use the binary indicator than constructing a "new"
  // sparse matrix.
  //
  // NOTE: throw error when indicator.size() != dim_in
  spu::NdArrayRef Recv(const spu::NdArrayRef& vec_in, int64_t dim_out,
                       int64_t dim_in, const StlSparseMatrix& priv_bin_mat,
                       absl::Span<const uint8_t> indicator);

 private:
  size_t ring_bitwidth_;
  struct Impl;
  std::shared_ptr<Impl> impl_;
};

}  // namespace squirrel
