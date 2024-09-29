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

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xtensor/xio.hpp"

#include "libspu/compiler/common/compilation_context.h"
#include "libspu/compiler/front_end/fe.h"
#include "libspu/core/prelude.h"
#include "libspu/device/api.h"
#include "libspu/device/intrinsic_table.h"
#include "libspu/dialect/pphlo/IR/dialect.h"
#include "libspu/dialect/ring/IR/dialect.h"
#include "libspu/dialect/ring/IR/ops.h"
#include "libspu/dialect/utils/utils.h"
#include "libspu/kernel/test_util.h"
#include "libspu/mpc/utils/simulate.h"

#include "libspu/spu.pb.h"
#define EXPOSE_PIPELINE_BUILDER
#include "libspu/compiler/core/core.h"
#undef EXPOSE_PIPELINE_BUILDER

template <typename T>
struct fmt::formatter<xt::xarray<T>> : ostream_formatter {};

llvm::cl::opt<uint32_t> ProtocolKind(
    "protocol_kind", llvm::cl::init(1),
    llvm::cl::desc("1 for REF2k, 2 for SEMI2k, 3 for ABY3, 4 for Cheetah"));

namespace mlir {

namespace {

void runPasses(ModuleOp module) {
  ::spu::CompilerOptions options;
  // ---- tweak compation options ---- //
  // --------------------------------- //
  ::spu::compiler::CompilationContext ccontext(options, module->getContext());
  auto &engine = ccontext.getMLIRContext()->getDiagEngine();
  engine.registerHandler([&](mlir::Diagnostic &diag) {
    SPDLOG_ERROR(diag.str());
    for (const auto &node : diag.getNotes()) {
      SPDLOG_ERROR(node.str());
    }
  });

  ::spu::compiler::FE fe(&ccontext);
  ::spu::compiler::Core c(&ccontext);
  {
    mlir::PassManager pm(module->getContext());
    c.buildFixedPointPipeline(&pm);

    SPU_ENFORCE(pm.run(module).succeeded());
    SPDLOG_INFO("Fxp IR\n {}", spu::mlirObjectToString(module));
  }

  {
    mlir::PassManager pm(module->getContext());
    c.buildRingPipeline(&pm);

    SPU_ENFORCE(pm.run(module).succeeded());
    SPDLOG_INFO("Ring IR\n {}", spu::mlirObjectToString(module));
  }
}

template <typename T>
void isEqual(const xt::xarray<T> &lhs, const xt::xarray<T> &rhs) {
  SPDLOG_INFO("lhs = {}", lhs);
  SPDLOG_INFO("rhs = {}", rhs);

  auto error = lhs - rhs;

  for (T v : error) {
    if (v != 0) {
      llvm::report_fatal_error(fmt::format("Diff = {}", v).c_str());
    }
  }
}

bool testOpHandler(::spu::SPUContext *sctx, mlir::Operation *op,
                   absl::Span<const ::spu::MemRef> inputs) {
  auto callOp = mlir::dyn_cast<func::CallOp>(op);

  auto callee = ::spu::device::demangle_fcn_name(callOp.getCallee());

  if (callee == "expect_almost_eq") {
    const ::spu::MemRef &runtimeLhs = inputs[0];
    ::spu::MemRef runtimeRhs = inputs[1];

    if (!runtimeRhs.isPublic()) {
      runtimeRhs = ::spu::kernel::hal::_s2p(sctx, runtimeRhs);
    }

    SPU_ENFORCE(runtimeRhs.isPublic());

    auto lhs = ::spu::kernel::hal::dump_public_as<double>(sctx, runtimeLhs, 18);
    auto rhs = ::spu::kernel::hal::dump_public_as<double>(sctx, runtimeRhs, 18);

    SPDLOG_INFO("lhs = {}", lhs);
    SPDLOG_INFO("rhs = {}", rhs);

    double tol = 0.1;
    if (auto tol_attr = callOp->getAttr("tol")) {
      tol = mlir::dyn_cast<FloatAttr>(tol_attr).getValueAsDouble();
    }

    auto error = xt::fabs(lhs - rhs);

    for (double v : error) {
      if (v > tol) {
        llvm::report_fatal_error(
            fmt::format("Diff {} greater than tol {}", v, tol).c_str());
      }
    }

    return true;
  }

  if (callee == "expect_eq") {
    const ::spu::MemRef &runtimeLhs = inputs[0];
    ::spu::MemRef runtimeRhs = inputs[1];

    if (!runtimeRhs.isPublic()) {
      runtimeRhs = ::spu::kernel::hal::_s2p(sctx, runtimeRhs);
    }

    SPU_ENFORCE(runtimeRhs.isPublic());

    auto it = mlir::dyn_cast<IntegerType>(
        getElementTypeOrSelf(callOp->getOperand(0).getType()));
    auto width = it.getWidth();
    auto unsign = it.isUnsigned();

    switch (width) {
    case 1: {
      auto lhs = ::spu::kernel::hal::dump_public_as<bool>(sctx, runtimeLhs);
      auto rhs = ::spu::kernel::hal::dump_public_as<bool>(sctx, runtimeRhs);
      isEqual(lhs, rhs);
      break;
    }
    case 8: {
      if (unsign) {
        auto lhs =
            ::spu::kernel::hal::dump_public_as<uint8_t>(sctx, runtimeLhs);
        auto rhs =
            ::spu::kernel::hal::dump_public_as<uint8_t>(sctx, runtimeRhs);
        isEqual(lhs, rhs);
      } else {
        auto lhs = ::spu::kernel::hal::dump_public_as<int8_t>(sctx, runtimeLhs);
        auto rhs = ::spu::kernel::hal::dump_public_as<int8_t>(sctx, runtimeRhs);
        isEqual(lhs, rhs);
      }
      break;
    }
    case 16: {
      if (unsign) {
        auto lhs =
            ::spu::kernel::hal::dump_public_as<uint16_t>(sctx, runtimeLhs);
        auto rhs =
            ::spu::kernel::hal::dump_public_as<uint16_t>(sctx, runtimeRhs);
        isEqual(lhs, rhs);
      } else {
        auto lhs =
            ::spu::kernel::hal::dump_public_as<int16_t>(sctx, runtimeLhs);
        auto rhs =
            ::spu::kernel::hal::dump_public_as<int16_t>(sctx, runtimeRhs);
        isEqual(lhs, rhs);
      }
      break;
    }
    case 32: {
      if (unsign) {
        auto lhs =
            ::spu::kernel::hal::dump_public_as<uint32_t>(sctx, runtimeLhs);
        auto rhs =
            ::spu::kernel::hal::dump_public_as<uint32_t>(sctx, runtimeRhs);
        isEqual(lhs, rhs);
      } else {
        auto lhs =
            ::spu::kernel::hal::dump_public_as<int32_t>(sctx, runtimeLhs);
        auto rhs =
            ::spu::kernel::hal::dump_public_as<int32_t>(sctx, runtimeRhs);
        isEqual(lhs, rhs);
      }
      break;
    }
    case 64: {
      if (unsign) {
        auto lhs =
            ::spu::kernel::hal::dump_public_as<uint64_t>(sctx, runtimeLhs);
        auto rhs =
            ::spu::kernel::hal::dump_public_as<uint64_t>(sctx, runtimeRhs);
        isEqual(lhs, rhs);
      } else {
        auto lhs =
            ::spu::kernel::hal::dump_public_as<int64_t>(sctx, runtimeLhs);
        auto rhs =
            ::spu::kernel::hal::dump_public_as<int64_t>(sctx, runtimeRhs);
        isEqual(lhs, rhs);
      }
      break;
    }
    }

    return true;
  }

  return false;
}

void evalModule(ModuleOp module) {
  // Run passes
  runPasses(module);

  ::spu::RuntimeConfig conf;
  conf.set_enable_type_checker(true);
  conf.mutable_protocol()->set_field(64);
  int numParties = 1;

  switch (ProtocolKind.getValue()) {
  case 1: {
    conf.mutable_protocol()->set_kind(::spu::REF2K);
    numParties = 1;
    break;
  }
  case 2: {
    conf.mutable_protocol()->set_kind(::spu::SEMI2K);
    numParties = 2;
    break;
  }
  case 3: {
    conf.mutable_protocol()->set_kind(::spu::ABY3);
    numParties = 3;
    break;
  }
  case 4: {
    conf.mutable_protocol()->set_kind(::spu::CHEETAH);
    numParties = 2;
    break;
  }
  case 5: {
    conf.mutable_protocol()->set_kind(::spu::SECURENN);
    numParties = 3;
    break;
  }
  }

  SPDLOG_INFO(conf.DebugString());

  auto entry_function = spu::get_entrypoint(module);
  SPU_ENFORCE(entry_function, "main module not found");

  ::spu::device::ConcreteExecutor executor;
  executor.setExtraIntrinsicHandler(testOpHandler);
  ::spu::device::ExecutionOptions opts;

  ::spu::mpc::utils::simulate(
      numParties, [&](const std::shared_ptr<yacl::link::Context> &lctx) {
        auto sctx = ::spu::kernel::test::makeSPUContext(conf, lctx);

        runRegion(&executor, &sctx, nullptr, entry_function.getBody(), {},
                  opts);
        return;
      });
}

} // namespace

TranslateFromMLIRRegistration interpretRegistration(
    "interpret", "Interpreter for SPU",
    [](Operation *op, raw_ostream &os) -> LogicalResult {
      auto module = mlir::dyn_cast<ModuleOp>(op);
      evalModule(module);

      return success();
    },
    [](DialectRegistry &registry) {
      registry.insert<stablehlo::StablehloDialect, spu::pphlo::PPHloDialect,
                      spu::ring::RingDialect, mlir::func::FuncDialect,
                      mlir::tensor::TensorDialect, mlir::arith::ArithDialect>();
    });

} //  namespace mlir

int main(int argc, char **argv) {
  return static_cast<int>(
      failed(mlir::mlirTranslateMain(argc, argv, "SPU interpreter driver\n")));
}
