// Copyright 2023 Ant Group Co., Ltd.
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

#include <fstream>
#include <future>
#include <memory>
#include <random>

#include "absl/strings/match.h"
#include "absl/strings/str_split.h"
#include "experimental/squirrel/tree_build_worker.h"
#include "experimental/squirrel/tree_builder.h"
#include "llvm/Support/CommandLine.h"
#include "xtensor/xarray.hpp"
#include "xtensor/xcsv.hpp"
#include "xtensor/xview.hpp"
#include "yacl/link/link.h"

#include "libspu/core/context.h"
#include "libspu/core/prelude.h"
#include "libspu/core/type.h"
#include "libspu/core/value.h"
#include "libspu/device/io.h"
#include "libspu/kernel/hal/type_cast.h"
#include "libspu/kernel/hlo/basic_binary.h"
#include "libspu/kernel/hlo/const.h"
#include "libspu/mpc/factory.h"

// Boosting releated parameters
llvm::cl::opt<uint32_t> BucketSize("bucket_size", llvm::cl::init(10),
                                   llvm::cl::desc("bucket size"));

llvm::cl::opt<uint32_t> MaxDepth("max_depth", llvm::cl::init(5),
                                 llvm::cl::desc("max_depth"));

llvm::cl::opt<uint32_t> NumTrees("num_tree", llvm::cl::init(10),
                                 llvm::cl::desc("the number of trees"));

llvm::cl::opt<double> RegLambda("lambda", llvm::cl::init(1.0),
                                llvm::cl::desc("reg lambda"));

llvm::cl::opt<double> LearningRate("lr", llvm::cl::init(0.3),
                                   llvm::cl::desc("learning rate"));

llvm::cl::opt<double> BaseScore("base_score", llvm::cl::init(0.0),
                                llvm::cl::desc("base score"));

llvm::cl::opt<double> Subsample("subsample", llvm::cl::init(1.0),
                                llvm::cl::desc("subsampling rate (0., 1]"));

llvm::cl::opt<std::string> Activation(
    "activation", llvm::cl::init("log"),
    llvm::cl::desc("activation type: log|sig"));

// Data
llvm::cl::opt<bool> Standalone("standalone", llvm::cl::init(true),
                               llvm::cl::desc("standalone dataset"));

llvm::cl::opt<std::string> Dataset("train", llvm::cl::init("data.csv"),
                                   llvm::cl::desc("only csv is supported"));

llvm::cl::opt<std::string> TestDataset("test", llvm::cl::init("NONE"),
                                       llvm::cl::desc("only csv is supported"));

llvm::cl::opt<uint32_t> Rank0NumFeatures(
    "rank0_nfeatures", llvm::cl::init(10),
    llvm::cl::desc("the number of features held by player0"));

llvm::cl::opt<uint32_t> Rank1NumFeatures(
    "rank1_nfeatures", llvm::cl::init(10),
    llvm::cl::desc("the number of features held by player1"));

llvm::cl::opt<bool> HasLabel(
    "has_label", llvm::cl::init(false),
    llvm::cl::desc("if true, label is the last column of dataset"));

// MPC releated parameters
llvm::cl::opt<std::string> Parties(
    "parties", llvm::cl::init("127.0.0.1:9530,127.0.0.1:9531"),
    llvm::cl::desc("server list, format: host1:port1[,host2:port2, ...]"));

llvm::cl::opt<uint32_t> Rank("rank", llvm::cl::init(0),
                             llvm::cl::desc("self rank"));

llvm::cl::opt<uint32_t> Field(
    "field", llvm::cl::init(2),
    llvm::cl::desc("1 for Ring32, 2 for Ring64, 3 for Ring128"));

llvm::cl::opt<bool> EngineTrace("engine_trace", llvm::cl::init(false),
                                llvm::cl::desc("Enable trace info"));

std::shared_ptr<yacl::link::Context> MakeLink(const std::string& parties,
                                              size_t rank) {
  yacl::link::ContextDesc lctx_desc;
  std::vector<std::string> hosts = absl::StrSplit(parties, ',');
  for (size_t rank = 0; rank < hosts.size(); rank++) {
    const auto id = fmt::format("party{}", rank);
    lctx_desc.parties.push_back({id, hosts[rank]});
  }
  auto lctx = yacl::link::FactoryBrpc().CreateContext(lctx_desc, rank);
  lctx->ConnectToMesh();
  return lctx;
}

std::unique_ptr<spu::SPUContext> MakeSPUContext() {
  auto lctx = MakeLink(Parties.getValue(), Rank.getValue());

  spu::RuntimeConfig config;
  config.set_protocol(spu::ProtocolKind::CHEETAH);
  config.set_field(static_cast<spu::FieldType>(Field.getValue()));
  config.set_fxp_fraction_bits(18);
  config.set_fxp_div_goldschmidt_iters(1);
  config.set_enable_hal_profile(EngineTrace.getValue());
  auto hctx = std::make_unique<spu::SPUContext>(config, lctx);
  spu::mpc::Factory::RegisterProtocol(hctx.get(), lctx);
  return hctx;
}

spu::Value InfeedInitPred(spu::SPUContext* hctx, const xt::xarray<double>& pred,
                          bool self_has_label) {
  spu::device::ColocatedIo cio(hctx);
  // NOTE(lwj): We let the label holder to set the init prediction vector just
  // for convience.
  if (self_has_label) {
    cio.hostSetVar("init_pred", pred);
  }
  cio.sync();
  return cio.deviceGetVar("init_pred");
}

spu::Value InfeedLabel(spu::SPUContext* hctx, const xt::xarray<double>& ds,
                       bool self_has_label) {
  spu::device::ColocatedIo cio(hctx);
  if (self_has_label) {
    // the last column is label.
    using namespace xt::placeholders;  // required for `_` to work
    xt::xarray<double> dy =
        xt::view(ds, xt::all(), xt::range(ds.shape(1) - 1, _));
    cio.hostSetVar("label", dy);
  }
  cio.sync();
  return cio.deviceGetVar("label");
}

xt::xarray<double> ReadDataSet(const std::string& fileName) {
  // read dataset.
  xt::xarray<double> dframe;
  {
    std::ifstream file(fileName);
    if (!file) {
      spdlog::error("open file={} failed", fileName);
      exit(-1);
    }
    dframe = xt::load_csv<double>(file, ',', /*skip*/ 1);
  }

  size_t ncols = dframe.shape(1);
  size_t rank0_nfeatures = Rank0NumFeatures.getValue();
  size_t rank1_nfeatures = Rank1NumFeatures.getValue();
  SPU_ENFORCE(rank0_nfeatures > 0);
  SPU_ENFORCE(rank1_nfeatures > 0);

  if (Standalone.getValue()) {
    SPU_ENFORCE_EQ(ncols, rank0_nfeatures + rank1_nfeatures + 1);
    if (Rank.getValue() == 0) {
      dframe = xt::view(dframe, xt::all(), xt::range(0, rank0_nfeatures));
    } else {
      // rank1 takes the label
      dframe = xt::view(dframe, xt::all(), xt::range(rank0_nfeatures, ncols));
    }
  } else {
    int extra = HasLabel.getValue() ? 1 : 0;
    if (Rank.getValue() == 0) {
      SPU_ENFORCE_EQ(rank0_nfeatures + extra, ncols);
    } else {
      SPU_ENFORCE_EQ(rank1_nfeatures + extra, ncols);
    }
  }

  return dframe;
}

void RunTest(spu::SPUContext* hctx, squirrel::XGBTreeBuilder& builder,
             std::shared_ptr<squirrel::XGBTreeBuildWorker>& worker,
             size_t tree_idx) {
  auto test_file = TestDataset.getValue();
  if (test_file == "NONE") {
    return;
  }

  // read dataset.
  xt::xarray<double> dframe = ReadDataSet(TestDataset.getValue());
  bool has_label =
      Standalone.getValue() ? Rank.getValue() == 1 : HasLabel.getValue();

  squirrel::ActivationType act_t = Activation.getValue() == "log"
                                       ? squirrel::ActivationType::Logistic
                                       : squirrel::ActivationType::Sigmoid;

  xt::xarray<double> dy;
  if (has_label) {
    using namespace xt::placeholders;
    dy = xt::view(dframe, xt::all(), xt::range(dframe.shape(1) - 1, _));
    // remove the label column
    dframe = xt::view(dframe, xt::all(), xt::range(_, dframe.shape(1) - 1));
  }

  const int64_t nsamples = dframe.shape(0);
  const double fxp = std::pow(2., hctx->config().fxp_fraction_bits());

  SPDLOG_DEBUG("Computing inference on testing set ...");

  std::vector<double> probs(nsamples);
  for (int64_t i = 0; i < nsamples; ++i) {
    xt::xarray<double> row = xt::row(dframe, i);
    auto pred = builder.Inference(hctx, worker, {row.data(), row.size()});
    pred = spu::kernel::hal::reveal(hctx, pred);

    if (has_label) {
      double p = pred.data().at<int64_t>(0) / fxp;
      double prob;
      if (act_t == squirrel::ActivationType::Logistic) {
        prob = 1. / (1. + std::exp(-p));
      } else {
        prob = 0.5 + 0.5 * p / std::sqrt(1.0 + p * p);
      }
      probs[i] = prob;
    }
  }
  SPDLOG_DEBUG("Computing inference on test set done");

  if (has_label) {
    std::string fname = fmt::format("test-label-prob-{}.csv", tree_idx);
    std::ofstream fout(fname);
    for (int64_t i = 0; i < nsamples; ++i) {
      fout << probs[i] << "," << dy(i, 0) << std::endl;
    }
    fout.close();
    SPDLOG_INFO("Saving to prediction to {}", fname);
  }
}

#define DEBUG_SHOW_LOSS 1

int main(int argc, char** argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv);
  // read dataset.
  xt::xarray<double> dframe = ReadDataSet(Dataset.getValue());
  auto hctx = MakeSPUContext();

  bool has_label =
      Standalone.getValue() ? Rank.getValue() == 1 : HasLabel.getValue();

  spu::Value label = InfeedLabel(hctx.get(), dframe, has_label);
  xt::xarray<double> dy;
  if (has_label) {
    using namespace xt::placeholders;
    dy = xt::view(dframe, xt::all(), xt::range(dframe.shape(1) - 1, _));
    // remove the label column
    dframe = xt::view(dframe, xt::all(), xt::range(_, dframe.shape(1) - 1));
  }

  std::vector<size_t> shape = {dframe.shape(0), 1};
  xt::xarray<double> init_pred(shape);

  if (has_label) {
    // init predicition for 0-th tree
    std::fill_n(init_pred.data(), init_pred.size(), BaseScore.getValue());
  }

  spu::Value pred = InfeedInitPred(hctx.get(), init_pred, has_label);

  size_t bucket_size = BucketSize.getValue();
  size_t nsamples = dframe.shape(0);
  size_t nfeatures = dframe.shape(1);
  size_t peer_nfeatures = Rank.getValue() == 0 ? Rank1NumFeatures.getValue()
                                               : Rank0NumFeatures.getValue();
  double subsample = Subsample.getValue();
  SPU_ENFORCE(subsample > 0 && subsample <= 1., "subsample={}", subsample);
  auto worker = std::make_shared<squirrel::XGBTreeBuildWorker>(
      bucket_size, nfeatures, peer_nfeatures);

  worker->BuildMap(dframe);
  worker->Setup(8 * spu::SizeOf(hctx->config().field()), hctx->lctx());

  std::string act = Activation.getValue();
  SPU_ENFORCE(act == "log" or act == "sig", "invalid activation type={}", act);
  squirrel::ActivationType act_t = act == "log"
                                       ? squirrel::ActivationType::Logistic
                                       : squirrel::ActivationType::Sigmoid;
  double reg_lambda = RegLambda.getValue();
  double rate = LearningRate.getValue();
  size_t ntrees = NumTrees.getValue();
  int max_depth = MaxDepth.getValue();
  SPU_ENFORCE(reg_lambda > 0 && reg_lambda <= 1000., "reg_lambda={}",
              reg_lambda);
  squirrel::XGBTreeBuilder builder(max_depth, reg_lambda, nsamples, subsample);

#if DEBUG_SHOW_LOSS
  [[maybe_unused]] double objective = 0.;
  [[maybe_unused]] double loss = 0.;
#endif

  for (size_t tree_idx = 0; tree_idx < ntrees; ++tree_idx) {
    SPDLOG_DEBUG("Computing gradients for Tree {} ...", tree_idx);
    auto [g, h] =
        builder.BinaryClassificationGradients(hctx.get(), pred, label, act_t);
    SPDLOG_DEBUG("Computing gradients for Tree {} done", tree_idx);
    builder.InitGradients(g, h);
    builder.BuildTree(hctx.get(), worker, rate);

    SPDLOG_DEBUG("Updating predictions after Tree {} ...", tree_idx);
    pred = builder.UpdatePrediction(hctx.get(), pred, tree_idx);
    SPDLOG_DEBUG("Updating predictions after Tree {} done", tree_idx);

#if DEBUG_SHOW_LOSS
    double lss = builder.DEBUG_OpenLoss(hctx.get(), pred, label);
    SPDLOG_INFO("Loss {} -> {} on Tree {}", loss, lss, tree_idx);
    loss = lss;
#endif

    RunTest(hctx.get(), builder, worker, tree_idx);
  }

  // Test on train set
  double fxp = std::pow(2., hctx->config().fxp_fraction_bits());
  int32_t correct = 0;
  SPDLOG_DEBUG("Computing inference on training set ...");
  for (int64_t i = 0; i < (int64_t)nsamples; ++i) {
    xt::xarray<double> row = xt::row(dframe, i);
    auto pred = builder.Inference(hctx.get(), worker, {row.data(), row.size()});
    pred = spu::kernel::hal::reveal(hctx.get(), pred);
    if (has_label) {
      double p = pred.data().at<int64_t>(0) / fxp;
      double prob;
      if (act_t == squirrel::ActivationType::Logistic) {
        prob = 1. / (1. + std::exp(-p));
      } else {
        prob = 0.5 + 0.5 * p / std::sqrt(1 + p * p);
      }
      // NOTE(lwj): should tune this threshold
      double got = prob >= 0.5;
      if (got == dy(i, 0)) {
        correct += 1;
      }
    }
  }

  SPDLOG_DEBUG("Computing inference on training set done");

  if (has_label) {
    SPDLOG_INFO("Correct {} out-of {} in the training set", correct, nsamples);
  }

  // builder.PrintTreesStructure();
  builder.PrintProfilingData();
  return 0;
}
