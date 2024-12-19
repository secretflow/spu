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
#include "experimental/squirrel/tree_build_worker.h"

#include "experimental/squirrel/bin_matvec_prot.h"
#include "experimental/squirrel/objectives.h"

#include "libspu/core/ndarray_ref.h"
#include "libspu/core/prelude.h"
#include "libspu/core/trace.h"
#include "libspu/kernel/hlo/basic_binary.h"
#include "libspu/kernel/hlo/geometrical.h"

namespace squirrel {

static void AccumulateHistogram(spu::NdArrayRef buckets_share, size_t nfeatures,
                                size_t bucket_size);

XGBTreeBuildWorker::XGBTreeBuildWorker(size_t bucket_size, size_t nfeatures,
                                       size_t peer_nfeatures)
    : bucket_size_(bucket_size),
      nfeatures_(nfeatures),
      peer_nfeatures_(peer_nfeatures) {}

void XGBTreeBuildWorker::Setup(size_t ring_bitwidth,
                               std::shared_ptr<yacl::link::Context> conn) {
  SPU_ENFORCE(ring_bitwidth >= 32 and ring_bitwidth <= 128);
  SPU_ENFORCE(conn != nullptr);
  std::shared_ptr<yacl::link::Context> dupx = conn->Spawn();
  if (0 == conn->Rank()) {
    matvec_prot_send_ =
        std::make_unique<BinMatVecProtocol>(ring_bitwidth, conn);
    matvec_prot_recv_ =
        std::make_unique<BinMatVecProtocol>(ring_bitwidth, dupx);
  } else {
    matvec_prot_recv_ =
        std::make_unique<BinMatVecProtocol>(ring_bitwidth, conn);
    matvec_prot_send_ =
        std::make_unique<BinMatVecProtocol>(ring_bitwidth, dupx);
  }
  rank_ = conn->Rank();
}

void XGBTreeBuildWorker::SetUpBucketMap(const StlSparseMatrix& bucket_map,
                                        const std::vector<Binning>& binnings) {
  SPU_ENFORCE_EQ(binnings.size(), nfeatures_);
  for (const auto& bin : binnings) {
    SPU_ENFORCE_EQ(bin.nbuckets(), bucket_size_);
  }

  SPU_ENFORCE_EQ((size_t)bucket_map.rows(), bucket_size_ * nfeatures_);

  bucket_map_ = bucket_map;
  binnings_ = binnings;
}

void XGBTreeBuildWorker::BuildMap(const xt::xarray<double>& dframe) {
  size_t nsamples = dframe.shape()[0];
  size_t nfeatures = dframe.shape()[1];
  SPU_ENFORCE_EQ(nfeatures, nfeatures_);
  binnings_.resize(nfeatures, Binning(bucket_size_));

  std::vector<StlSparseMatrix::SparseRow> nzero_position(
      /*nrows*/ bucket_size_ * nfeatures);

  for (size_t f = 0; f < nfeatures; ++f) {
    const size_t feature_bucket_pos = f * bucket_size_;

    xt::xarray<double> fd =
        xt::ravel<xt::layout_type::column_major>(xt::col(dframe, f));
    absl::Span<const double> _fd = {fd.data(), fd.size()};

    binnings_[f].Fit(_fd);
    std::vector<uint16_t> bucket_indices = binnings_[f].Transform(_fd);

    SPU_ENFORCE_EQ(nsamples, bucket_indices.size());

    // Put samples in bucket[0], ..., bucket[bucket_size-1]
    for (size_t i = 0; i < nsamples; ++i) {
      size_t bucket = bucket_indices[i];
      SPU_ENFORCE(bucket < bucket_size_);
      int row = static_cast<int>(feature_bucket_pos + bucket);
      int col = static_cast<int>(i);
      nzero_position[row].insert(col);
    }
  }

  bucket_map_ = StlSparseMatrix::Initialize(nzero_position, /*ncols*/ nsamples);
}

std::pair<spu::Value, spu::Value> XGBTreeBuildWorker::ComputeGradientSums(
    spu::SPUContext* ctx, const spu::Value& gradient, const spu::Value& hessian,
    absl::Span<const uint8_t> sample_indicator) {
  using namespace spu;
  SPU_TRACE_HAL_LEAF(ctx, gradient, hessian);
  SPU_ENFORCE(matvec_prot_send_ != nullptr && matvec_prot_recv_ != nullptr,
              "Call Setup() first");
  size_t dim_in = bucket_map_.cols();
  size_t dim_out = bucket_map_.rows();
  SPU_ENFORCE_EQ(gradient.numel(), hessian.numel());
  SPU_ENFORCE_EQ(dim_in, (size_t)gradient.numel());
  SPU_ENFORCE(sample_indicator.empty() || dim_in == sample_indicator.size());

  // Need 1D tensor
  auto grad = gradient.data().reshape({gradient.numel()});
  auto hess = hessian.data().reshape({hessian.numel()});
  size_t peer_dim_out = peer_nfeatures_ * bucket_size_;

  // parallel rank0 -> rank1 BinMatVec
  auto subtask =
      std::async(std::launch::async, [&]() -> std::array<spu::NdArrayRef, 2> {
        if (rank_ == 0) {
          auto G = matvec_prot_send_->Send(grad, peer_dim_out, dim_in);
          auto H = matvec_prot_send_->Send(hess, peer_dim_out, dim_in);
          return {G, H};
        }
        auto G = matvec_prot_recv_->Recv(grad, dim_out, dim_in, bucket_map_,
                                         sample_indicator);
        auto H = matvec_prot_recv_->Recv(hess, dim_out, dim_in, bucket_map_,
                                         sample_indicator);
        return {G, H};
      });

  // parallel rank1 -> rank0 BinMatVec
  spu::NdArrayRef G0;
  spu::NdArrayRef H0;
  if (rank_ == 1) {
    // From rank1, peer_total_buckets = #rows of matrix 0
    G0 = matvec_prot_send_->Send(grad, peer_dim_out, dim_in);
    H0 = matvec_prot_send_->Send(hess, peer_dim_out, dim_in);
  } else {
    G0 = matvec_prot_recv_->Recv(grad, dim_out, dim_in, bucket_map_,
                                 sample_indicator);
    H0 = matvec_prot_recv_->Recv(hess, dim_out, dim_in, bucket_map_,
                                 sample_indicator);
  }

  auto [G1, H1] = subtask.get();
  size_t nfeatures_0 = rank_ == 0 ? nfeatures_ : peer_nfeatures_;
  size_t nfeatures_1 = rank_ == 1 ? nfeatures_ : peer_nfeatures_;

  // We compute: G[i] = sum_{j <= i} bin[j]
  //
  // That is G[0] = bin[0]
  //         G[1] = bin[0] + bin[1]
  //         G[2] = bin[0] + bin[1] + bin[2]
  //             ...
  //       G[B-1] = bin[0] + bin[1] ... bin[B-1] (The sum all bins)
  //
  // Using this partial sum format, we can simplify the computation of split
  // gain. where
  //      gain[i] = \sum_{j <= i} bin[j] + \sum{j > i} bin[j] - \sum_{k} bin[k]
  AccumulateHistogram(G0, nfeatures_0, bucket_size_);
  AccumulateHistogram(H0, nfeatures_0, bucket_size_);
  AccumulateHistogram(G1, nfeatures_1, bucket_size_);
  AccumulateHistogram(H1, nfeatures_1, bucket_size_);

  spu::Shape shape0 = {1L, G0.numel()};
  spu::Shape shape1 = {1L, G1.numel()};
  G0 = G0.reshape(shape0);
  H0 = H0.reshape(shape0);
  G1 = G1.reshape(shape1);
  H1 = H1.reshape(shape1);

  spu::Value G0_(G0, gradient.dtype());
  spu::Value G1_(G1, gradient.dtype());

  spu::Value H0_(H0, hessian.dtype());
  spu::Value H1_(H1, hessian.dtype());

  // We let rank0's buckets come before rank1's buckets.
  // G = G0 || G1
  // H = H0 || H1
  auto G_ = spu::kernel::hlo::Concatenate(ctx, {G0_, G1_}, 1);
  auto H_ = spu::kernel::hlo::Concatenate(ctx, {H0_, H1_}, 1);
  return {G_, H_};
}

std::pair<size_t, double> XGBTreeBuildWorker::SplitInfo(
    size_t target_bucket_index) const {
  SPU_ENFORCE(target_bucket_index < nfeatures_ * bucket_size_,
              "invalid target_bucket_index={}", target_bucket_index);
  size_t feature_index = target_bucket_index / bucket_size_;
  size_t bin_index = target_bucket_index % bucket_size_;
  double threshold = binnings_[feature_index].bin_thresholds().at(bin_index);
  return {feature_index, threshold};
}

std::vector<uint8_t> XGBTreeBuildWorker::PotentialLeftIndicator(
    size_t target_bucket_index) {
  size_t nsamples = this->num_samples();
  SPU_ENFORCE(target_bucket_index < nfeatures_ * bucket_size_,
              "invalid target_bucket_index={}", target_bucket_index);

  std::vector<uint8_t> indicator(nsamples, 0);

  size_t feature_index = target_bucket_index / bucket_size_;
  size_t bucket_bgn = feature_index * bucket_size_;

  // bucket_map_: (bucket_size * nfeatures) x nsample mapping
  for (size_t k = bucket_bgn; k <= target_bucket_index; ++k) {
    for (auto col_iter = bucket_map_.iterate_row_begin(k);
         col_iter != bucket_map_.iterate_row_end(k); ++col_iter) {
      size_t sample_index = *col_iter;
      SPU_ENFORCE(sample_index < nsamples);
      // hit once
      SPU_ENFORCE(indicator[sample_index] == 0);
      indicator[sample_index] = 1;
    }
  }
  return indicator;
}

void AccumulateHistogram(spu::NdArrayRef buckets_share, size_t nfeatures,
                         size_t bucket_size) {
  using namespace spu;
  SPU_ENFORCE(buckets_share.ndim() == 1);
  SPU_ENFORCE_EQ((size_t)buckets_share.numel(), nfeatures * bucket_size);
  // The buckets belong to the i-th feature is
  // `buckets[i*bucket_size:(i+1)*bucket_size]`
  auto field = buckets_share.eltype().as<RingTy>()->field();
  DISPATCH_ALL_FIELDS(field, [&]() {
    NdArrayView<ring2k_t> histogram(buckets_share);
    for (size_t j = 0; j < nfeatures; ++j) {
      size_t start = j * bucket_size;
      size_t end = start + bucket_size;
      for (size_t k = start + 1; k < end; ++k) {
        histogram[k] += histogram[k - 1];
      }
    }
  });
}

}  // namespace squirrel
