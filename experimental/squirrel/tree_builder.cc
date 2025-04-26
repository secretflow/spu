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
#include "experimental/squirrel/tree_builder.h"

#include <chrono>
#include <future>
#include <random>

#include "experimental/squirrel/objectives.h"
#include "experimental/squirrel/tree_build_worker.h"
#include "experimental/squirrel/utils.h"

#include "libspu/core/prelude.h"
#include "libspu/core/type_util.h"
#include "libspu/kernel/hal/fxp_base.h"
#include "libspu/kernel/hal/type_cast.h"
#include "libspu/kernel/hlo/basic_binary.h"
#include "libspu/kernel/hlo/casting.h"
#include "libspu/kernel/hlo/const.h"
#include "libspu/kernel/hlo/geometrical.h"
#include "libspu/mpc/cheetah/state.h"
#include "libspu/mpc/cheetah/type.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace squirrel {

#define RECORD_STATS(NAME, ctx)             \
  if (stats_.find(NAME) == stats_.end()) {  \
    stats_.insert({NAME, {Duration(), 0}}); \
  }                                         \
  auto __kv = stats_.find(NAME);            \
  StatsGuard __guard(__kv->second, ctx->lctx());

namespace {
using TimePoint = std::chrono::time_point<std::chrono::high_resolution_clock>;
using Duration = std::chrono::nanoseconds;

// copy from libspu/device/api.cc
class StatsGuard {
  TimePoint start_;
  size_t send_bytes_ = 0;

  TimeCommuStat& stat_;
  const std::shared_ptr<yacl::link::Context>& lctx_;

 public:
  explicit StatsGuard(TimeCommuStat& stat,
                      const std::shared_ptr<yacl::link::Context>& lctx)
      : stat_(stat), lctx_(lctx) {
    start_ = std::chrono::high_resolution_clock::now();
    send_bytes_ = lctx_->GetStats()->sent_bytes;
  }

  ~StatsGuard() {
    stat_.first += (std::chrono::high_resolution_clock::now() - start_);
    stat_.second += (lctx_->GetStats()->sent_bytes - send_bytes_);
  }
};

[[maybe_unused]] double getSeconds(const Duration& dur) {
  return std::chrono::duration_cast<std::chrono::duration<double>>(dur).count();
}

[[maybe_unused]] double getSeconds(const TimePoint& start,
                                   const TimePoint& end) {
  return std::chrono::duration_cast<std::chrono::duration<double>>(end - start)
      .count();
}

struct ExecutionStats {
  Duration total_time() const {
    return infeed_time + execution_time + outfeed_time;
  }
  Duration infeed_time;
  Duration execution_time;
  Duration outfeed_time;
};

}  // namespace

XGBTreeBuilder::XGBTreeBuilder(int max_depth, double reg_lambda,
                               size_t nsamples, double subsample)
    : max_depth_(max_depth),
      reg_lambda_(reg_lambda),
      nsamples_(nsamples),
      subsample_(subsample) {
  SPU_ENFORCE(max_depth_ > 1);
  SPU_ENFORCE(reg_lambda_ > 0.);
  SPU_ENFORCE(nsamples_ >= 1);
  SPU_ENFORCE(subsample_ > 0. && subsample_ <= 1.);
}

void XGBTreeBuilder::BuildTree(
    spu::SPUContext* ctx, const std::shared_ptr<XGBTreeBuildWorker>& worker,
    double learn_rate) {
  SPU_ENFORCE(worker != nullptr);
  SPU_ENFORCE_EQ(nsamples_, worker->num_samples());
  split_identifier_t dummy;
  // Node index starts from 1, so we add a dummy
  dummy.push_back({false, 0, 0.});
  split_identifiers_.push_back(dummy);
  for (int depth = 1; depth <= max_depth_; ++depth) {
    TrainLevel(ctx, worker, depth);
  }

  SPDLOG_DEBUG("ComputeLeafWeights ...");
  ComputeLeafWeights(ctx, worker, learn_rate);
  SPDLOG_DEBUG("ComputeLeafWeights done");
}

// pred[i] <- pred[i] + sum_k w_k * 1{sample_i in leaf_k}
spu::Value XGBTreeBuilder::UpdatePrediction(spu::SPUContext* ctx,
                                            const spu::Value& predictions,
                                            size_t tree_index) {
  RECORD_STATS("update_prediction", ctx);
  using namespace spu;
  using namespace spu::kernel;
  SPU_ENFORCE_EQ(predictions.numel(), (int64_t)nsamples_);
  SPU_ENFORCE(tree_index < leaf_weights_.size());
  size_t leaf_idx_start = 1UL << (max_depth_ - 1);
  const auto& weights = leaf_weights_.at(tree_index);

  // Iterate the indicator in all leafs
  // For each leaf, one AShr weight multiplies `nsamples` bits.
  // Thus, for total k leafs, `k` AShr x `k*nsamples` bits
  size_t num_leafs = sample_indicators_.size();
  SPU_ENFORCE_EQ(num_leafs, (size_t)weights.numel());

  // concat all the indicators in the leafs
  std::vector<uint8_t> indicators(nsamples_ * num_leafs);
  for (auto& kv : sample_indicators_) {
    int64_t leaf_pos = kv.first - leaf_idx_start;
    if (kv.second.empty()) {
      // empty indicator means all 1s
      std::fill_n(indicators.data() + leaf_pos * nsamples_, nsamples_, 1);
    } else {
      SPU_ENFORCE_EQ(kv.second.size(), nsamples_);
      std::copy_n(kv.second.data(), nsamples_,
                  indicators.data() + leaf_pos * nsamples_);
    }
  }

  auto updated =
      BatchMulArithShareWithANDBoolShare(ctx, weights, nsamples_, indicators);

  auto pred = hlo::Reshape(ctx, predictions, {predictions.numel()});

  for (int64_t j = 0; j < updated.numel(); j += nsamples_) {
    int64_t end = j + nsamples_;
    pred = hlo::Add(ctx, pred, hlo::Slice(ctx, updated, {j}, {end}, {1}));
  }
  return hlo::Reshape(ctx, pred, predictions.shape());
}

void XGBTreeBuilder::ComputeLeafWeights(
    spu::SPUContext* ctx, const std::shared_ptr<XGBTreeBuildWorker>& worker,
    double learn_rate) {
  RECORD_STATS("compute_weight", ctx);
  using namespace spu::kernel;
  SPU_ENFORCE(learn_rate > 0. && learn_rate < 10.);

  const int64_t bucket_size = worker->bucket_size();
  std::vector<spu::Value> _Gsum;
  std::vector<spu::Value> _Hsum;
  // G shape (1, nfeatures * bucket_size)
  size_t leaf_idx_start = 1UL << (max_depth_ - 1);
  size_t leaf_idx_end = 1UL << (max_depth_);

  for (size_t idx = leaf_idx_start; idx < leaf_idx_end; ++idx) {
    auto kv = cached_GHs_.find(idx);
    SPU_ENFORCE(kv != cached_GHs_.end(), "Leaf {} is missing", idx);
    const auto& G = kv->second.first;
    const auto& H = kv->second.second;
    // last bucket is the sum of all gradients on the node
    _Gsum.push_back(
        hlo::Slice(ctx, G, {0, bucket_size - 1}, {1, bucket_size}, {1, 1}));
    _Hsum.push_back(
        hlo::Slice(ctx, H, {0, bucket_size - 1}, {1, bucket_size}, {1, 1}));
  }

  // shape (#leafs, 1)
  auto Gsum = hlo::Concatenate(ctx, _Gsum, 0);
  auto Hsum = hlo::Concatenate(ctx, _Hsum, 0);

  // w = -((G / (H + reg_lambda)) * learning_rate)
  Hsum = hlo::Add(ctx, Hsum, hlo::Constant(ctx, reg_lambda_, Hsum.shape()));
  // NOTE H + reg_lambda is always positive
  Hsum = hal::detail::reciprocal_goldschmidt_positive(ctx, Hsum);
  auto w = hlo::Mul(ctx, Gsum, Hsum);
  leaf_weights_.push_back(
      hlo::Mul(ctx, w, hlo::Constant(ctx, -learn_rate, w.shape())));

#if 0  // debug
  spu::NdArrayRef _indicator =
      spu::mpc::ring_zeros(spu::FM32, {(int64_t)nsamples_});
  spu::NdArrayView<uint32_t> indicator(_indicator);
  std::vector<uint8_t> ground(nsamples_, 0);

  auto* comm = ctx->prot()->getState<spu::mpc::Communicator>();
  for (auto& kv : sample_indicators_) {
    if (kv.second.empty()) {
      std::fill_n(&indicator[0], nsamples_, 1);
    } else {
      spu::pforeach(0, nsamples_,
                    [&](int64_t i) { indicator[i] = (uint32_t)kv.second[i]; });
    }
    auto _peer_indicator = comm->rotate(_indicator, "indicator");

    if (worker->rank() == 0) {
      spu::NdArrayView<uint32_t> peer_indicator(_peer_indicator);

      size_t count_sample = 0;
      for (size_t i = 0; i < nsamples_; ++i) {
        uint8_t g = indicator[i] * peer_indicator[i];
        if (g) {
          SPU_ENFORCE(ground[i] == 0);
          ground[i] = g;
        }
        count_sample += g;
      }

      printf("Node %zd %zd util #sample %d\n", kv.first, count_sample,
             std::accumulate(ground.begin(), ground.end(), 0));
    }
  }
#endif
}

spu::Value XGBTreeBuilder::UpdateGradient(spu::SPUContext* ctx,
                                          const spu::Value& gradient,
                                          absl::Span<const uint8_t> indicator) {
  RECORD_STATS("update_gradient", ctx);
  // Compute <grad>_A * indicator where indicator is privately hold by one
  // party.
  if (indicator.empty()) {
    return MulArithShareWithPrivateBoolean(ctx, gradient);
  }

  SPU_ENFORCE_EQ((size_t)gradient.numel(), indicator.size());
  return MulArithShareWithPrivateBoolean(ctx, gradient, indicator);
}

void XGBTreeBuilder::TrainLevel(
    spu::SPUContext* ctx, const std::shared_ptr<XGBTreeBuildWorker>& worker,
    size_t level) {
  using namespace spu::kernel;
  SPU_ENFORCE(level > 0 && level <= (size_t)max_depth_);
  // Level 1: node 1
  // Level 2: node 2, 3
  // Level 3: node 4, 5, 6 7
  // Level i: node 2^{i-1} ... 2^{i}-1
  size_t node_idx_bgn = 1UL << (level - 1);
  size_t node_idx_end = 1UL << level;
  for (size_t idx = node_idx_bgn; idx < node_idx_end; ++idx) {
    SPU_ENFORCE(cached_gh_.find(idx) != cached_gh_.end(),
                "gradients on node {} are missing", idx);
    SPU_ENFORCE(sample_indicators_.find(idx) != sample_indicators_.end(),
                "indicator on node {} is missing", idx);
  }

  const bool is_leaf_level = level == static_cast<size_t>(max_depth_);
  std::vector<spu::Value> current_G;
  std::vector<spu::Value> current_H;
  for (size_t idx = node_idx_bgn; idx < node_idx_end; idx += 2) {
    RECORD_STATS("gradient_sum", ctx);
    const auto& indicator = sample_indicators_.find(idx)->second;
    const auto& [gradient, hessian] = cached_gh_.find(idx)->second;

    // Left child
    SPDLOG_DEBUG("ComputeGradientSums on Node {} ...", idx);
    auto [GL, HL] = worker->ComputeGradientSums(ctx, gradient, hessian,
                                                absl::MakeSpan(indicator));
    SPDLOG_DEBUG("ComputeGradientSums on Node {} done", idx);

    if (not is_leaf_level) {
      current_G.push_back(GL);
      current_H.push_back(HL);
    }
    cached_GHs_.insert({idx, {GL, HL}});

    size_t parent_idx = idx / 2;
    auto parent_GH = cached_GHs_.find(parent_idx);
    if (parent_GH == cached_GHs_.end()) {
      // level = 1
      continue;
    }

    // Histgram subtraction to obtain the siblings
    const auto& Gp = parent_GH->second.first;
    const auto& Hp = parent_GH->second.second;
    auto GR = hlo::Sub(ctx, Gp, GL);
    auto HR = hlo::Sub(ctx, Hp, HL);

    if (not is_leaf_level) {
      current_G.push_back(GR);
      current_H.push_back(HR);
    }
    cached_GHs_.erase(parent_idx);
    cached_GHs_.insert({idx + 1, {GR, HR}});
  }

  if (is_leaf_level) {
    // already a leaf, no need to split
    return;
  }

  // Level-wise growth. Concat all the G and H in this level.
  spu::Value Gs = hlo::Concatenate(ctx, current_G, 0);
  spu::Value Hs = hlo::Concatenate(ctx, current_H, 0);

  SPDLOG_DEBUG("Max gain on level {} ...", level);
  spu::Value max_gains_index;
  spu::Value greater_bits;
  {
    RECORD_STATS("find_best_split", ctx);
    max_gains_index = MaxGainOnLevel(ctx, Gs, Hs, reg_lambda_);
    // To see 1{max index >= #buckets on rank0}
    // 1{max_index >= #buckets on rank0} => 1{max_index > #buckets on rank0 - 1}
    // If max_index >= #buckets on rank0 => the split feature belongs to rank1
    // If max_index < #buckets on rank0 => the split feature belongs to rank0
    int64_t nbuckets_rank0 =
        worker->nfeatures(/*rank*/ 0) * worker->bucket_size();
    greater_bits = hlo::Greater(
        ctx, max_gains_index,
        hlo::Constant(ctx, nbuckets_rank0 - 1, max_gains_index.shape()));
    // Reveal to both.
    greater_bits = hal::reveal(ctx, greater_bits);
  }
  SPDLOG_DEBUG("Max gain on level {} done", level);

  SPDLOG_DEBUG("Update gradient share on level {} ...", level);
  SplitLevel(ctx, worker, level, max_gains_index, greater_bits);
  SPDLOG_DEBUG("Update gradient share on level {} done", level);
}

void XGBTreeBuilder::SplitLevel(spu::SPUContext* ctx,
                                std::shared_ptr<XGBTreeBuildWorker> worker,
                                size_t level, const spu::Value& max_gains_index,
                                const spu::Value& plain_greater_bits) {
  using namespace spu;
  using namespace spu::kernel;
  SPU_ENFORCE(level > 0);
  size_t node_idx_bgn = 1UL << (level - 1);
  size_t node_idx_end = 1UL << level;
  int64_t num_nodes = node_idx_end - node_idx_bgn;
  SPU_ENFORCE_EQ(num_nodes, max_gains_index.shape()[0]);

  // TODO: Should we combine multiple updates into one to reduce the
  // communication round.
  for (int64_t i = 0; i < plain_greater_bits.numel(); ++i) {
    auto max_gain_index = hlo::Slice(ctx, max_gains_index, {i}, {i + 1}, {1});

    // greater_bit = 1 => split by rank1
    // greater_bit = 0 => split by rank0
    int32_t belongs_to = plain_greater_bits.data().at<int32_t>(i);
    SPU_ENFORCE(belongs_to == 0 || belongs_to == 1);

    size_t nidx = node_idx_bgn + i;
    std::vector<uint8_t> p = sample_indicators_.find(nidx)->second;
    sample_indicators_.erase(nidx);  // drop this
    const auto& [grad, hess] = cached_gh_.find(nidx)->second;

    spu::Value grad_L, hess_L;
    spu::Value grad_R, hess_R;

    SPDLOG_DEBUG("Node {} will be splitted by Rank={}", nidx, belongs_to);

    // NOTE: split_info = { is_holding_the_feature, feature_index, threshold }
    std::tuple<bool, size_t, double> split_info = {false, 0, 0.};
    if (belongs_to == worker->rank()) {
      // FIXME(lwj): use reveal_to to reveal the `max_gains_index` to this
      // rank. However, current spu does not supoort reveal_to yet.
      max_gain_index = hal::reveal(ctx, max_gain_index);
      int32_t bucket_index = max_gain_index.data().at<int64_t>(0);
      int32_t target_bucket_idx = worker->map_bucket(bucket_index);
      const auto& [split_feature, split_threshold] =
          worker->SplitInfo(target_bucket_idx);
      split_info = {true, split_feature, split_threshold};
      // b* indicator
      auto b_ = worker->PotentialLeftIndicator(target_bucket_idx);
      std::vector<uint8_t> lchild(b_.size(), 0);
      std::vector<uint8_t> rchild(b_.size(), 0);
      for (size_t i = 0; i < b_.size(); ++i) {
        uint8_t pi = p.empty() ? 1 : (p[i] & 1);
        lchild[i] = pi & b_[i];
        rchild[i] = pi - lchild[i];
      }

      sample_indicators_.insert({nidx * 2, lchild});
      sample_indicators_.insert({nidx * 2 + 1, rchild});

      grad_L = UpdateGradient(ctx, grad, b_);
      hess_L = UpdateGradient(ctx, hess, b_);
    } else {
      // FIXME(lwj): needs reveal_to
      (void)hal::reveal(ctx, max_gain_index);

      // keep the indicators unchanged
      sample_indicators_.insert({nidx * 2, p});
      sample_indicators_.insert({nidx * 2 + 1, p});

      grad_L = UpdateGradient(ctx, grad);
      hess_L = UpdateGradient(ctx, hess);
    }

    // right gradient = parent gradient - left gradient
    grad_R = hlo::Sub(ctx, grad, grad_L);
    hess_R = hlo::Sub(ctx, hess, hess_L);

    cached_gh_.erase(nidx);  // clean up
    cached_gh_.insert({2 * nidx, {grad_L, hess_L}});
    cached_gh_.insert({2 * nidx + 1, {grad_R, hess_R}});
    // back() point to the current tree
    split_identifiers_.back().push_back(split_info);
  }
}

void XGBTreeBuilder::InitGradients(const spu::Value& gradient,
                                   const spu::Value& hessian) {
  SPU_ENFORCE_EQ(gradient.numel(), hessian.numel());
  SPU_ENFORCE_EQ(nsamples_, (size_t)gradient.numel());

  cached_gh_.clear();
  cached_gh_.insert({1, {gradient, hessian}});
  cached_GHs_.clear();

  sample_indicators_.clear();
  constexpr size_t root = 1;
  std::vector<uint8_t> all;  // empty indicates all 1s

  if (subsample_ < 1.) {
    all.resize(nsamples_, 1);
    std::default_random_engine rdv(std::time(0));
    std::uniform_real_distribution<double> uniform(0., 1.);
    for (size_t i = 0; i < nsamples_; ++i) {
      all[i] = uniform(rdv) < subsample_ ? 1 : 0;
    }
  }
  sample_indicators_.insert({root, all});
}

spu::Value XGBTreeBuilder::Inference(
    spu::SPUContext* ctx, const std::shared_ptr<XGBTreeBuildWorker>& worker,
    const absl::Span<const double> x) {
  using namespace spu;
  RECORD_STATS("inference", ctx);

  size_t ntrees = split_identifiers_.size();
  size_t nfeature = x.size();
  SPU_ENFORCE_EQ(ntrees, leaf_weights_.size());
  SPU_ENFORCE_EQ(nfeature, worker->nfeatures(worker->rank()));

  const size_t leaf_idx_start = 1UL << (max_depth_ - 1);
  const size_t num_leafs = (1UL << max_depth_) - leaf_idx_start;

  std::vector<uint8_t> leaf_indicators(num_leafs * ntrees);
  for (size_t t = 0; t < ntrees; ++t) {
    const auto& split_identifiers = split_identifiers_[t];
    size_t num_nodes = split_identifiers.size();
    std::vector<uint8_t> path_indicator(1UL << max_depth_, 0);
    path_indicator[1] = 1;  // init root
    for (size_t nidx = 1; nidx < num_nodes; ++nidx) {
      const auto& [self, feature, threshold] = split_identifiers[nidx];
      size_t lchild = 2 * nidx;
      size_t rchild = 2 * nidx + 1;
      if (self) {
        SPU_ENFORCE(feature < nfeature);
        if (x[feature] <= threshold) {
          // go left
          path_indicator[lchild] = path_indicator[nidx];
          path_indicator[rchild] = 0;
        } else {
          // go right
          path_indicator[lchild] = 0;
          path_indicator[rchild] = path_indicator[nidx];
        }
      } else {
        // not holding the split feature
        path_indicator[lchild] = path_indicator[nidx];
        path_indicator[rchild] = path_indicator[nidx];
      }
    }

    std::copy_n(path_indicator.data() + leaf_idx_start, num_leafs,
                leaf_indicators.data() + t * num_leafs);
  }

  // ntrees * num_leafs
  auto leaf_weights = kernel::hlo::Concatenate(ctx, leaf_weights_, 0);

  auto updated =
      MulArithShareWithANDBoolShare(ctx, leaf_weights, leaf_indicators);

  return ReduceSum(ctx, updated, 0);
}

std::pair<spu::Value, spu::Value> XGBTreeBuilder::BinaryClassificationGradients(
    spu::SPUContext* ctx, const spu::Value& pred, const spu::Value& label,
    ActivationType act) {
  using namespace spu::kernel;
  RECORD_STATS("binary_gradient", ctx);
  spu::Value prob;
  switch (act) {
    case ActivationType::Logistic: {
      prob = Logistic(ctx, pred);
      break;
    }
    case ActivationType::Sigmoid: {
      prob = Sigmoid(ctx, pred);
      break;
    }
    default:
      SPU_THROW("Unknown activation type");
  }

  // gradient = prob - y
  spu::Value g = hlo::Sub(ctx, prob, label);
  // hessian = (1 - prob) * prob = prob - prob^2 (square is faster)
  spu::Value h = hlo::Sub(ctx, prob, hal::f_square(ctx, prob));
  return {g, h};
}

void XGBTreeBuilder::PrintProfilingData() const {
  for (auto kv : stats_) {
    SPDLOG_INFO("- {}, duration {:4f}s, total send {:4f}MB", kv.first,
                getSeconds(kv.second.first), kv.second.second / 1024. / 1024.);
  }
}

void XGBTreeBuilder::PrintTreesStructure() const {
  size_t ntrees = split_identifiers_.size();
  for (size_t t_idx = 0; t_idx < ntrees; ++t_idx) {
    size_t nnodes = split_identifiers_[t_idx].size();
    for (size_t n_idx = 1; n_idx < nnodes; ++n_idx) {
      auto& [self, fidx, thld] = split_identifiers_[t_idx][n_idx];
      if (not self) {
        printf("Tree %zd Node %zd Self 0 Feature 0 Threshold 0\n", t_idx,
               n_idx);
      } else {
        printf("Tree %zd Node %zd Self 1 Feature %zd Threshold %f\n", t_idx,
               n_idx, fidx, thld);
      }
    }
  }
}

double XGBTreeBuilder::DEBUG_OpenObjects(
    spu::SPUContext* ctx,
    const std::shared_ptr<XGBTreeBuildWorker>& worker) const {
  using namespace spu::kernel;

  const int64_t bucket_size = worker->bucket_size();
  std::vector<spu::Value> _Gsum;
  std::vector<spu::Value> _Hsum;
  // G shape (1, nfeatures * bucket_size)
  size_t leaf_idx_start = 1UL << (max_depth_ - 1);
  size_t leaf_idx_end = 1UL << (max_depth_);

  for (size_t idx = leaf_idx_start; idx < leaf_idx_end; ++idx) {
    auto kv = cached_GHs_.find(idx);
    SPU_ENFORCE(kv != cached_GHs_.end(), "Leaf {} is missing", idx);
    const auto& G = kv->second.first;
    const auto& H = kv->second.second;
    // last bucket is the sum of all gradients on the node
    _Gsum.push_back(
        hlo::Slice(ctx, G, {0, bucket_size - 1}, {1, bucket_size}, {1, 1}));
    _Hsum.push_back(
        hlo::Slice(ctx, H, {0, bucket_size - 1}, {1, bucket_size}, {1, 1}));
  }

  // shape (#leafs, 1)
  auto Gsum = hlo::Concatenate(ctx, _Gsum, 0);
  auto Hsum = hlo::Concatenate(ctx, _Hsum, 0);
  auto weights = leaf_weights_.back();
  SPU_ENFORCE_EQ(Gsum.numel(), weights.numel());

  Gsum = hal::reveal(ctx, Gsum);
  Hsum = hal::reveal(ctx, Hsum);
  weights = hal::reveal(ctx, weights);
  const double fxp = std::pow(2., ctx->config().fxp_fraction_bits);
  double object = 0.0;
  for (int64_t i = 0; i < weights.numel(); ++i) {
    double G = Gsum.data().at<int64_t>(i) / fxp;
    double H = Hsum.data().at<int64_t>(i) / fxp;
    double w = weights.data().at<int64_t>(i) / fxp;
    object += w * G + 0.5 * (H + reg_lambda_) * w * w;
  }
  return object;
}

double XGBTreeBuilder::DEBUG_OpenLoss(spu::SPUContext* ctx,
                                      const spu::Value& pred,
                                      const spu::Value& label) const {
  using namespace spu::kernel;
  auto _pred = hal::reveal(ctx, pred);
  auto _label = hal::reveal(ctx, label);
  double fxp = std::pow(2., ctx->config().fxp_fraction_bits);
  double loss = 0.;
  for (int64_t i = 0; i < _pred.numel(); ++i) {
    double y = _label.data().at<int64_t>(i) / fxp;
    double yhat = _pred.data().at<int64_t>(i) / fxp;
    loss += y * std::log(1 + std::exp(-yhat)) +
            (1. - y) * std::log(1 + std::exp(yhat));
  }
  return loss;
}

}  // namespace squirrel
