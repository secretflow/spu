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
//
#include <chrono>
#include <unordered_map>

#include "libspu/core/context.h"
#include "libspu/core/value.h"

namespace squirrel {

class XGBTreeBuildWorker;

using TimeCommuStat = std::pair<std::chrono::nanoseconds, size_t>;

enum class ActivationType {
  Logistic,  // 1/(1 + exp(-x))
  Sigmoid,   // 0.5+0.5x/sqrt(1+x^2)
};

class XGBTreeBuilder {
 public:
  XGBTreeBuilder(int max_depth, double reg_lambda, size_t nsamples,
                 double subsample = 1.);

  void InitGradients(const spu::Value& gradient, const spu::Value& hessian);

  void BuildTree(spu::SPUContext* ctx,
                 const std::shared_ptr<XGBTreeBuildWorker>& worker,
                 double learn_rate = 1.);

  spu::Value UpdatePrediction(spu::SPUContext* ctx,
                              const spu::Value& predictions, size_t tree_index);

  spu::Value Inference(spu::SPUContext* ctx,
                       const std::shared_ptr<XGBTreeBuildWorker>& worker,
                       absl::Span<const double> x);

  std::pair<spu::Value, spu::Value> BinaryClassificationGradients(
      spu::SPUContext* ctx, const spu::Value& pred, const spu::Value& label,
      ActivationType act = ActivationType::Logistic);

  void PrintProfilingData() const;

  void PrintTreesStructure() const;

  double DEBUG_OpenObjects(
      spu::SPUContext* ctx,
      const std::shared_ptr<XGBTreeBuildWorker>& worker) const;

  double DEBUG_OpenLoss(spu::SPUContext* ctx, const spu::Value& pred,
                        const spu::Value& label) const;

 private:
  void TrainLevel(spu::SPUContext* ctx,
                  const std::shared_ptr<XGBTreeBuildWorker>& worker,
                  size_t level);

  // NOTE: empty indicator indicates all 1s.
  spu::Value UpdateGradient(spu::SPUContext* ctx, const spu::Value& gradient,
                            absl::Span<const uint8_t> indicator = {nullptr, 0});

  void SplitLevel(spu::SPUContext* ctx,
                  std::shared_ptr<XGBTreeBuildWorker> worker, size_t level,
                  const spu::Value& max_gains_index,
                  const spu::Value& plain_greater_bits);

  void ComputeLeafWeights(spu::SPUContext* ctx,
                          const std::shared_ptr<XGBTreeBuildWorker>& worker,
                          double learn_rate);

 private:
  int max_depth_ = -1;
  double reg_lambda_ = 1.0;  // regularizer > 0
  size_t nsamples_ = 0;
  double subsample_ = 1.;  // subsampling rate \in (0., 1.]

  // Node index starts with 1.
  using split_identifier_t = std::vector<std::tuple<bool, size_t, double>>;
  // split_identifiers_[i][j] indicates the j-th split info for the i-th tree
  // i \in [0, ntrees)
  // j \in [1, 2^{max_depth_}]
  std::vector<split_identifier_t> split_identifiers_;
  // leaf_weights_[i] indicates the leaf weights on the i-th tree
  // leaf_weights_[i][j] is the j-th leaf for j \in [0, 2^{max_depth_ - 1})
  std::vector<spu::Value> leaf_weights_;
  // NodeIdx -> GH pair
  std::unordered_map<size_t, std::pair<spu::Value, spu::Value>> cached_gh_;
  // NodeIdx -> GH pair
  std::unordered_map<size_t, std::pair<spu::Value, spu::Value>> cached_GHs_;
  // NodeIdx -> Sample Indicator
  std::unordered_map<size_t, std::vector<uint8_t>> sample_indicators_;
  // Timing Statistics
  std::unordered_map<std::string, TimeCommuStat> stats_;
};

}  // namespace squirrel
