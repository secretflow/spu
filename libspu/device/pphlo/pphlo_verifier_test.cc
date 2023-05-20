// Copyright 2022 Ant Group Co., Ltd.
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

#include "libspu/device/pphlo/pphlo_verifier.h"

#include "gtest/gtest.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "xtensor/xarray.hpp"

#include "libspu/device/test_utils.h"
#include "libspu/kernel/test_util.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::device::pphlo {

void failed(bool v) {
  if (!v) {
    SPU_THROW("Equal check failed");
  }
}

template <typename InT, typename OutT, typename OpFcn>
void runner(const OpFcn &f, absl::Span<const xt::xarray<InT>> inputs,
            absl::Span<const xt::xarray<OutT>> positives,
            absl::Span<const xt::xarray<OutT>> negatives) {
  RuntimeConfig conf;
  conf.set_field(FM64);
  conf.set_protocol(SEMI2K);
  std::unique_ptr<LocalIo> io_ = std::make_unique<LocalIo>(2, conf);
  for (size_t idx = 0; idx < inputs.size(); ++idx) {
    io_->InFeed(fmt::format("in{}", idx), inputs[idx], VIS_SECRET);
  }
  for (size_t idx = 0; idx < positives.size(); ++idx) {
    io_->InFeed(fmt::format("pout{}", idx), positives[idx], VIS_SECRET);
    io_->InFeed(fmt::format("nout{}", idx), negatives[idx], VIS_SECRET);
  }

  ::spu::mpc::utils::simulate(
      2, [&](const std::shared_ptr<yacl::link::Context> &lctx) {
        SPUContext sctx = kernel::test::makeSPUContext(conf, lctx);
        PPHloVerifier verifier(&sctx);
        verifier.setMismatchHandler(failed);
        auto *table = io_->GetSymbolTable(lctx->Rank());

        std::vector<spu::Value> in(inputs.size());
        std::vector<spu::Value> pout(positives.size());
        std::vector<spu::Value> nout(negatives.size());
        for (size_t idx = 0; idx < in.size(); ++idx) {
          in[idx] = table->getVar(fmt::format("in{}", idx));
        }
        for (size_t idx = 0; idx < pout.size(); ++idx) {
          pout[idx] = table->getVar(fmt::format("pout{}", idx));
          nout[idx] = table->getVar(fmt::format("nout{}", idx));
        }

        // positive case, should pass
        verifier.verify(f(), in, pout);

        // negative case, should throw
        EXPECT_THROW(verifier.verify(f(), in, nout), yacl::RuntimeError);
      });
}

TEST(Verify, DISABLED_Reciprocal) {
  runner<float, float>([] { return mlir::pphlo::ReciprocalOp{}; },
                       {xt::xarray<float>{1, 2, 4, 8}},
                       {xt::xarray<float>{1, 0.5, 0.25, 0.125}},
                       {xt::xarray<float>{1, 2, 2, 1}});
}

TEST(Verify, Neg) {
  runner<int32_t, int32_t>(
      [] { return mlir::pphlo::NegOp{}; }, {xt::xarray<int32_t>{0, 0, 1, 1}},
      {xt::xarray<int32_t>{0, 0, -1, -1}}, {xt::xarray<int32_t>{0, 0, 0, 0}});
}

TEST(Verify, Log) {
  runner<float, float>(
      [] { return mlir::pphlo::LogOp{}; }, {xt::xarray<float>{1, 2, 3, 4}},
      {xt::xarray<float>{0, 0.6931472, 1.09861229, 1.38629436}},
      {xt::xarray<float>{1, 1, 1, 1}});
}

TEST(Verify, DISABLED_Log1p) {
  runner<float, float>(
      [] { return mlir::pphlo::Log1pOp{}; }, {xt::xarray<float>{0, 1, 2, 3}},
      {xt::xarray<float>{0, 0.6931472, 1.09861229, 1.38629436}},
      {xt::xarray<float>{1, 1, 1, 1}});
}

TEST(Verify, Floor) {
  runner<float, float>([] { return mlir::pphlo::FloorOp{}; },
                       {xt::xarray<float>{-1.1, 2.2, -3.3, 4.4}},
                       {xt::xarray<float>{-2, 2, -4, 4}},
                       {xt::xarray<float>{1, 2, 2, 1}});
}

TEST(Verify, Ceil) {
  runner<float, float>([] { return mlir::pphlo::CeilOp{}; },
                       {xt::xarray<float>{-1.1, 2.2, -3.3, 4.4}},
                       {xt::xarray<float>{-1, 3, -3, 5}},
                       {xt::xarray<float>{1, 2, 2, 1}});
}

TEST(Verify, Abs) {
  runner<int64_t, int64_t>([] { return mlir::pphlo::AbsOp{}; },
                           {xt::xarray<int64_t>{-1, -1, -1, -1}},
                           {xt::xarray<int64_t>{1, 1, 1, 1}},
                           {xt::xarray<int64_t>{1, 2, 2, 1}});
}

TEST(Verify, Logisitc) {
  runner<float, float>(
      [] { return mlir::pphlo::LogisticOp{}; }, {xt::xarray<float>{1, 2, 3, 4}},
      {xt::xarray<float>{0.73105858, 0.88079708, 0.95257413, 0.98201379}},
      {xt::xarray<float>{1, 1, 1, 1}});
}

TEST(Verify, Tanh) {
  runner<float, float>(
      [] { return mlir::pphlo::TanhOp{}; }, {xt::xarray<float>{1, 2, 3, 4}},
      {xt::xarray<float>{0.76159416, 0.96402758, 0.99505475, 0.9993293}},
      {xt::xarray<float>{1, 1, 1, 1}});
}

TEST(Verify, Not) {
  runner<uint8_t, uint8_t>([] { return mlir::pphlo::NotOp{}; },
                           {xt::xarray<uint8_t>{2, 0, 1, 0}},
                           {xt::xarray<uint8_t>{253, 255, 254, 255}},
                           {xt::xarray<uint8_t>{1, 2, 2, 1}});
}

TEST(Verify, Add) {
  runner<int32_t, int32_t>(
      [] { return mlir::pphlo::AddOp{}; },
      {xt::xarray<int32_t>{1, 2, 3, 4}, xt::xarray<int32_t>{5, 6, 7, 8}},
      {xt::xarray<int32_t>{6, 8, 10, 12}}, {xt::xarray<int32_t>{1, 2, 2, 1}});
}

TEST(Verify, Sub) {
  runner<int32_t, int32_t>(
      [] { return mlir::pphlo::SubtractOp{}; },
      {xt::xarray<int32_t>{1, 2, 3, 4}, xt::xarray<int32_t>{5, 6, 7, 8}},
      {xt::xarray<int32_t>{-4, -4, -4, -4}}, {xt::xarray<int32_t>{1, 2, 2, 1}});
}

TEST(Verify, Mul) {
  runner<int32_t, int32_t>(
      [] { return mlir::pphlo::MulOp{}; },
      {xt::xarray<int32_t>{1, 2, 3, 4}, xt::xarray<int32_t>{5, 6, 7, 8}},
      {xt::xarray<int32_t>{5, 12, 21, 32}}, {xt::xarray<int32_t>{1, 2, 2, 1}});
}

TEST(Verify, Pow) {
  runner<int32_t, int32_t>(
      [] { return mlir::pphlo::PowOp{}; },
      {xt::xarray<int32_t>{1, 2, 3, 4}, xt::xarray<int32_t>{5, 6, 7, 8}},
      {xt::xarray<int32_t>{1, 64, 2187, 65536}},
      {xt::xarray<int32_t>{1, 2, 2, 1}});
}

TEST(Verify, Max) {
  runner<int32_t, int32_t>(
      [] { return mlir::pphlo::MaxOp{}; },
      {xt::xarray<int32_t>{10, 9, 8, 7}, xt::xarray<int32_t>{5, 6, 7, 8}},
      {xt::xarray<int32_t>{10, 9, 8, 8}}, {xt::xarray<int32_t>{1, 2, 2, 1}});
}

TEST(Verify, Min) {
  runner<int32_t, int32_t>(
      [] { return mlir::pphlo::MinOp{}; },
      {xt::xarray<int32_t>{10, 9, 8, 7}, xt::xarray<int32_t>{5, 6, 7, 8}},
      {xt::xarray<int32_t>{5, 6, 7, 7}}, {xt::xarray<int32_t>{1, 2, 2, 1}});
}

TEST(Verify, And) {
  runner<int8_t, int8_t>(
      [] { return mlir::pphlo::AndOp{}; },
      {xt::xarray<int8_t>{1, 15, 33, 60}, xt::xarray<int8_t>{12, 13, 17, 19}},
      {xt::xarray<int8_t>{0, 13, 1, 16}}, {xt::xarray<int8_t>{1, 2, 2, 1}});
}

TEST(Verify, Or) {
  runner<int8_t, int8_t>(
      [] { return mlir::pphlo::OrOp{}; },
      {xt::xarray<int8_t>{1, 15, 33, 60}, xt::xarray<int8_t>{12, 13, 17, 19}},
      {xt::xarray<int8_t>{13, 15, 49, 63}}, {xt::xarray<int8_t>{1, 2, 2, 1}});
}

TEST(Verify, Xor) {
  runner<int8_t, int8_t>(
      [] { return mlir::pphlo::XorOp{}; },
      {xt::xarray<int8_t>{1, 15, 33, 60}, xt::xarray<int8_t>{12, 13, 17, 19}},
      {xt::xarray<int8_t>{13, 2, 48, 47}}, {xt::xarray<int8_t>{1, 2, 2, 1}});
}

TEST(Verify, Div) {
  runner<float, float>(
      [] { return mlir::pphlo::DivOp{}; },
      {xt::xarray<float>{1, 15, 33, 60}, xt::xarray<float>{2, 3, 3, 5}},
      {xt::xarray<float>{0.5, 5, 11, 12}}, {xt::xarray<float>{1, 2, 2, 1}});
}

TEST(Verify, Rem) {
  runner<int32_t, int32_t>(
      [] { return mlir::pphlo::RemOp{}; },
      {xt::xarray<int32_t>{1, 15, 33, 60}, xt::xarray<int32_t>{2, 3, 10, 7}},
      {xt::xarray<int32_t>{1, 0, 3, 4}}, {xt::xarray<int32_t>{1, 2, 2, 1}});
}

TEST(Verify, DISABLED_Dot) {
  runner<int32_t, int32_t>(
      [] { return mlir::pphlo::DotOp{}; },
      {xt::xarray<int32_t>{{1, 3, 5}, {2, 4, 7}},
       xt::xarray<int32_t>{{-5, 8, 11}, {3, 9, 21}, {4, 0, 8}}},
      {xt::xarray<int32_t>{{24, 35, 114}, {30, 52, 162}}},
      {xt::xarray<int32_t>{{1, 2, 2}, {1, 2, 2}}});
}

TEST(Verify, Equal) {
  runner<int32_t, bool>(
      [] { return mlir::pphlo::EqualOp{}; },
      {xt::xarray<int32_t>{1, 2, 3, 4}, xt::xarray<int32_t>{5, 2, 7, 4}},
      {xt::xarray<bool>{false, true, false, true}},
      {xt::xarray<bool>{false, false, false, false}});
}

TEST(Verify, Less) {
  runner<int32_t, bool>(
      [] { return mlir::pphlo::LessOp{}; },
      {xt::xarray<int32_t>{1, 2, 3, 4}, xt::xarray<int32_t>{5, 2, 7, 4}},
      {xt::xarray<bool>{true, false, true, false}},
      {xt::xarray<bool>{false, false, false, false}});
}

TEST(Verify, Greater) {
  runner<int32_t, bool>(
      [] { return mlir::pphlo::GreaterOp{}; },
      {xt::xarray<int32_t>{1, 2, 3, 4}, xt::xarray<int32_t>{5, 2, 7, 4}},
      {xt::xarray<bool>{false, false, false, false}},
      {xt::xarray<bool>{true, true, false, false}});
}

TEST(Verify, Select) {
  RuntimeConfig conf;
  conf.set_field(FM64);
  conf.set_protocol(SEMI2K);
  std::unique_ptr<LocalIo> io_ = std::make_unique<LocalIo>(2, conf);
  io_->InFeed("in0", xt::xarray<bool>{false, true, true, false}, VIS_SECRET);
  io_->InFeed("in1", xt::xarray<int32_t>{5, 2, 7, 4}, VIS_SECRET);
  io_->InFeed("in2", xt::xarray<int32_t>{1, 2, 3, 8}, VIS_SECRET);

  io_->InFeed("pout", xt::xarray<int32_t>{1, 2, 7, 8}, VIS_SECRET);
  io_->InFeed("nout", xt::xarray<int32_t>{0, 0, 0, 0}, VIS_SECRET);

  ::spu::mpc::utils::simulate(
      2, [&](const std::shared_ptr<yacl::link::Context> &lctx) {
        SPUContext sctx = kernel::test::makeSPUContext(conf, lctx);
        PPHloVerifier verifier(&sctx);
        verifier.setMismatchHandler(failed);
        auto *table = io_->GetSymbolTable(lctx->Rank());

        // positive case, should pass
        verifier.verify(
            mlir::pphlo::SelectOp{},
            {table->getVar("in0"), table->getVar("in1"), table->getVar("in2")},
            {table->getVar("pout")});

        // negative case, should throw
        EXPECT_THROW(
            verifier.verify(mlir::pphlo::SelectOp{},
                            {table->getVar("in0"), table->getVar("in1"),
                             table->getVar("in2")},
                            {table->getVar("nout")}),
            yacl::RuntimeError);
      });
}

TEST(Verify, Clamp) {
  runner<int32_t, int32_t>(
      [] { return mlir::pphlo::ClampOp{}; },
      {xt::xarray<int32_t>{1, 2, 3, 4}, xt::xarray<int32_t>{10, 1, -40, 5},
       xt::xarray<int32_t>{5, 6, 7, 8}},
      {xt::xarray<int32_t>{5, 2, 3, 5}}, {xt::xarray<int32_t>{1, 1, 1, 1}});
}

TEST(Verify, DynamicSlice) {
  std::string mlir = R"(
func.func @main(%arg0: tensor<5x!pphlo.pub<i32>>, %arg1: tensor<!pphlo.pub<i32>>) -> tensor<2x!pphlo.pub<i32>> {
  %0 = "pphlo.dynamic-slice"(%arg0, %arg1) {slice_sizes = dense<2> : tensor<1xi64>} : (tensor<5x!pphlo.pub<i32>>, tensor<!pphlo.pub<i32>>) -> tensor<2x!pphlo.pub<i32>>
  return %0 : tensor<2x!pphlo.pub<i32>>
})";

  mlir::DialectRegistry registry;
  registry.insert<mlir::pphlo::PPHloDialect, mlir::func::FuncDialect>();
  auto mlir_ctx = std::make_unique<mlir::MLIRContext>(registry);

  auto moduleOpRef =
      mlir::parseSourceString<mlir::ModuleOp>(mlir, mlir_ctx.get());

  auto entry_function = moduleOpRef->lookupSymbol<mlir::func::FuncOp>("main");
  runner<int32_t, int32_t>(
      [&] {
        return mlir::dyn_cast<mlir::pphlo::DynamicSliceOp>(
            entry_function.getBody().front().front());
      },
      {xt::xarray<int32_t>{0, 1, 2, 3, 4}, 2}, {xt::xarray<int32_t>{2, 3}},
      {xt::xarray<int32_t>{1, 1}});
}

TEST(Verify, DynamicUpdateSlice) {
  runner<int32_t, int32_t>(
      [] { return mlir::pphlo::DynamicUpdateSliceOp{}; },
      {xt::xarray<int32_t>{0, 1, 2, 3, 4}, xt::xarray<int32_t>{5, 6}, 2},
      {xt::xarray<int32_t>{0, 1, 5, 6, 4}},
      {xt::xarray<int32_t>{1, 1, 1, 1, 1}});
}

}  // namespace spu::device::pphlo
