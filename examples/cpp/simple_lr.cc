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

// clang-format off
// To run the example, start two terminals:
// > bazel run //examples/cpp:simple_lr -- --dataset=examples/data/perfect_logit_a.csv --has_label=true
// > bazel run //examples/cpp:simple_lr -- --dataset=examples/data/perfect_logit_b.csv --rank=1
// clang-format on

#include <fstream>
#include <iostream>
#include <vector>

#include "examples/cpp/utils.h"
#include "spdlog/spdlog.h"
#include "xtensor/xarray.hpp"
#include "xtensor/xcsv.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"

#include "libspu/device/io.h"
#include "libspu/kernel/hal/hal.h"
#include "libspu/kernel/hal/type_cast.h"
#include "libspu/mpc/factory.h"

using namespace spu::kernel;

spu::Value train_step(spu::SPUContext* ctx, const spu::Value& x,
                      const spu::Value& y, const spu::Value& w) {
  // Padding x
  auto padding = hal::constant(ctx, 1.0F, spu::DT_F32, {x.shape()[0], 1});
  auto padded_x = hal::concatenate(ctx, {x, hal::seal(ctx, padding)}, 1);
  auto pred = hal::logistic(ctx, hal::matmul(ctx, padded_x, w));

  SPDLOG_DEBUG("[SSLR] Err = Pred - Y");
  auto err = hal::sub(ctx, pred, y);

  SPDLOG_DEBUG("[SSLR] Grad = X.t * Err");
  auto grad = hal::matmul(ctx, hal::transpose(ctx, padded_x), err);

  SPDLOG_DEBUG("[SSLR] Step = LR / B * Grad");
  auto lr = hal::constant(ctx, 0.0001F, spu::DT_F32);
  auto msize =
      hal::constant(ctx, static_cast<float>(y.shape()[0]), spu::DT_F32);
  auto p1 = hal::mul(ctx, lr, hal::reciprocal(ctx, msize));
  auto step = hal::mul(ctx, hal::broadcast_to(ctx, p1, grad.shape()), grad);

  SPDLOG_DEBUG("[SSLR] W = W - Step");
  auto new_w = hal::sub(ctx, w, step);

  return new_w;
}

spu::Value train(spu::SPUContext* ctx, const spu::Value& x, const spu::Value& y,
                 size_t num_epoch, size_t bsize) {
  const size_t num_iter = x.shape()[0] / bsize;
  auto w = hal::constant(ctx, 0.0F, spu::DT_F32, {x.shape()[1] + 1, 1});

  // Run train loop
  for (size_t epoch = 0; epoch < num_epoch; ++epoch) {
    for (size_t iter = 0; iter < num_iter; ++iter) {
      SPDLOG_INFO("Running train iteration {}", iter);

      const int64_t rows_beg = iter * bsize;
      const int64_t rows_end = rows_beg + bsize;

      const auto x_slice =
          hal::slice(ctx, x, {rows_beg, 0}, {rows_end, x.shape()[1]}, {});

      const auto y_slice =
          hal::slice(ctx, y, {rows_beg, 0}, {rows_end, y.shape()[1]}, {});

      w = train_step(ctx, x_slice, y_slice, w);
    }
  }

  return w;
}

spu::Value inference(spu::SPUContext* ctx, const spu::Value& x,
                     const spu::Value& weight) {
  auto padding = hal::constant(ctx, 1.0F, spu::DT_F32, {x.shape()[0], 1});
  auto padded_x = hal::concatenate(ctx, {x, hal::seal(ctx, padding)}, 1);
  return hal::matmul(ctx, padded_x, weight);
}

float SSE(const xt::xarray<float>& y_true, const xt::xarray<float>& y_pred) {
  float sse = 0;

  for (auto y_true_iter = y_true.begin(), y_pred_iter = y_pred.begin();
       y_true_iter != y_true.end() && y_pred_iter != y_pred.end();
       ++y_pred_iter, ++y_true_iter) {
    sse += std::pow(*y_true_iter - *y_pred_iter, 2);
  }
  return sse;
}

float MSE(const xt::xarray<float>& y_true, const xt::xarray<float>& y_pred) {
  auto sse = SSE(y_true, y_pred);

  return sse / static_cast<float>(y_true.size());
}

llvm::cl::opt<std::string> Dataset("dataset", llvm::cl::init("data.csv"),
                                   llvm::cl::desc("only csv is supported"));
llvm::cl::opt<uint32_t> SkipRows(
    "skip_rows", llvm::cl::init(1),
    llvm::cl::desc("skip number of rows from dataset"));
llvm::cl::opt<bool> HasLabel(
    "has_label", llvm::cl::init(false),
    llvm::cl::desc("if true, label is the last column of dataset"));
llvm::cl::opt<uint32_t> BatchSize("batch_size", llvm::cl::init(21),
                                  llvm::cl::desc("size of each batch"));
llvm::cl::opt<uint32_t> NumEpoch("num_epoch", llvm::cl::init(1),
                                 llvm::cl::desc("number of epoch"));

std::pair<spu::Value, spu::Value> infeed(spu::SPUContext* sctx,
                                         const xt::xarray<float>& ds,
                                         bool self_has_label) {
  spu::device::ColocatedIo cio(sctx);
  if (self_has_label) {
    // the last column is label.
    using namespace xt::placeholders;  // required for `_` to work
    xt::xarray<float> dx =
        xt::view(ds, xt::all(), xt::range(_, ds.shape(1) - 1));
    xt::xarray<float> dy =
        xt::view(ds, xt::all(), xt::range(ds.shape(1) - 1, _));
    cio.hostSetVar(fmt::format("x-{}", sctx->lctx()->Rank()), dx);
    cio.hostSetVar("label", dy);
  } else {
    cio.hostSetVar(fmt::format("x-{}", sctx->lctx()->Rank()), ds);
  }
  cio.sync();

  auto x = cio.deviceGetVar("x-0");
  // Concatenate all slices
  for (size_t idx = 1; idx < cio.getWorldSize(); ++idx) {
    x = hal::concatenate(sctx, {x, cio.deviceGetVar(fmt::format("x-{}", idx))},
                         1);
  }
  auto y = cio.deviceGetVar("label");

  return std::make_pair(x, y);
}

int main(int argc, char** argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv);
  // read dataset.
  xt::xarray<float> ds;
  {
    std::ifstream file(Dataset.getValue());
    if (!file) {
      spdlog::error("open file={} failed", Dataset.getValue());
      exit(-1);
    }
    ds = xt::load_csv<float>(file, ',', SkipRows.getValue());
  }

  auto sctx = MakeSPUContext();

  spu::mpc::Factory::RegisterProtocol(sctx.get(), sctx->lctx());

  const auto& [x, y] = infeed(sctx.get(), ds, HasLabel.getValue());

  const auto w =
      train(sctx.get(), x, y, NumEpoch.getValue(), BatchSize.getValue());

  const auto scores = inference(sctx.get(), x, w);

  xt::xarray<float> revealed_labels =
      hal::dump_public_as<float>(sctx.get(), hal::reveal(sctx.get(), y));
  xt::xarray<float> revealed_scores =
      hal::dump_public_as<float>(sctx.get(), hal::reveal(sctx.get(), scores));

  auto mse = MSE(revealed_labels, revealed_scores);
  std::cout << "MSE = " << mse << "\n";

  return 0;
}
