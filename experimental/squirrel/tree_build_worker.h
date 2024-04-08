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
#include <future>
#include <random>

#include "experimental/squirrel/bin_matvec_prot.h"
#include "experimental/squirrel/binning.h"
#include "xtensor/xarray.hpp"
#include "xtensor/xview.hpp"

#include "libspu/core/context.h"
#include "libspu/core/prelude.h"
#include "libspu/core/type.h"
#include "libspu/core/value.h"

namespace squirrel {

class XGBTreeBuildWorker {
 public:
  XGBTreeBuildWorker(size_t bucket_size, size_t nfeatures,
                     size_t peer_nfeatures);

  void Setup(size_t ring_bitwidth, std::shared_ptr<yacl::link::Context> conn);

  // Partition data according to feature's percentiles.
  // Build a mapping from feature_index -> bucket_index
  //   bucket_map \in {0, 1}^{(nfeatures * bucket_size) x nsamples}
  //   bucket_map[j * bucket_size + k, i] = 1 <=> Sample `i` is putted into the
  //   k-th bucket of the j-th feature.
  //
  // Sum of gradients: G = bucket_map * g \in RR^{nfeatures * bucket_size}
  void BuildMap(const xt::xarray<double>& dframe);

  // Directly setup the bucketing maps.
  // Indeed, for the XGB training, we only need the bucketing maps
  // instead of the dataframe itself.
  void SetUpBucketMap(const StlSparseMatrix& bucket_map,
                      const std::vector<Binning>& binnings);

  // Ignore sample_indicator[i] = 0
  // Empty indicator is regarded as all 1s indicator.
  std::pair<spu::Value, spu::Value> ComputeGradientSums(
      spu::SPUContext* ctx, const spu::Value& gradient,
      const spu::Value& hessian,
      absl::Span<const uint8_t> sample_indicator = {nullptr, 0});

  // The b* indicator. b*[i] = 1 <=> sample i feature[fidx] >
  // buckets[target_bucket_index] where fidx = target_bucket_index / bucket_size
  std::vector<uint8_t> PotentialLeftIndicator(size_t target_bucket_index);

  // target_bucket_index -> feature index and threshold to go left
  std::pair<size_t, double> SplitInfo(size_t target_bucket_index) const;

  size_t num_samples() const { return bucket_map_.cols(); }

  int rank() const { return rank_; }

  int bucket_belongs_to(size_t bucket) const {
    size_t nbuckets_0 =
        (rank_ == 0 ? nfeatures_ : peer_nfeatures_) * bucket_size_;
    return bucket >= nbuckets_0 ? 1 : 0;
  }

  size_t map_bucket(size_t bucket) const {
    size_t nbuckets_0 =
        (rank_ == 0 ? nfeatures_ : peer_nfeatures_) * bucket_size_;
    return bucket >= nbuckets_0 ? bucket - nbuckets_0 : bucket;
  }

  size_t nfeatures(int rank) const {
    return rank == rank_ ? nfeatures_ : peer_nfeatures_;
  }

  size_t bucket_size() const { return bucket_size_; }

 private:
  int rank_ = -1;
  // NOTE(lwj): we use two instances for simutaneously
  // computing the bin_matmul for 1st order gradient and 2nd order.
  std::unique_ptr<BinMatVecProtocol> matvec_prot_send_;
  std::unique_ptr<BinMatVecProtocol> matvec_prot_recv_;

  size_t bucket_size_;
  size_t nfeatures_;       // number of self's features
  size_t peer_nfeatures_;  // number of peer's features

  StlSparseMatrix bucket_map_;  // (bucket_size * nfeatures) x nsamples mapping
  std::vector<Binning> binnings_;  // feature -> partition percentiles
};

}  // namespace squirrel
