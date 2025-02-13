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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xtensor/xio.hpp"

#include "libspu/compiler/common/compilation_context.h"
#include "libspu/core/prelude.h"
#include "libspu/device/pphlo/pphlo_executor.h"
#include "libspu/dialect/pphlo/IR/dialect.h"
#include "libspu/dialect/pphlo/IR/ops.h"
#include "libspu/dialect/pphlo/transforms/passes.h"
#include "libspu/dialect/utils/utils.h"
#include "libspu/kernel/test_util.h"
#include "libspu/mpc/utils/simulate.h"
#include "libspu/spu.h"

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
  ::spu::compiler::CompilationContext ccontext(options);

  ::spu::compiler::Core c(&ccontext);
  mlir::PassManager pm(module->getContext());

  c.buildPipeline(&pm);

  SPU_ENFORCE(pm.run(module).succeeded());

  SPDLOG_INFO("IR\n {}", spu::mlirObjectToString(module));
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
                   absl::Span<const ::spu::Value> inputs) {
  auto callOp = mlir::dyn_cast<spu::pphlo::CustomCallOp>(op);
  if (callOp.getCallTargetName() == "expect_almost_eq") {
    ::spu::Value runtimeLhs = inputs[0];
    const ::spu::Value &runtimeRhs = inputs[1];

    if (!runtimeLhs.isPublic()) {
      runtimeLhs = ::spu::kernel::hal::_s2p(sctx, runtimeLhs)
                       .setDtype(runtimeLhs.dtype());
    }

    SPU_ENFORCE(runtimeRhs.isPublic());

    auto lhs = ::spu::kernel::hal::dump_public_as<double>(sctx, runtimeLhs);
    auto rhs = ::spu::kernel::hal::dump_public_as<double>(sctx, runtimeRhs);

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

  if (callOp.getCallTargetName() == "expect_eq") {
    ::spu::Value runtimeLhs = inputs[0];
    const ::spu::Value &runtimeRhs = inputs[1];

    if (!runtimeLhs.isPublic()) {
      runtimeLhs = ::spu::kernel::hal::_s2p(sctx, runtimeLhs)
                       .setDtype(runtimeLhs.dtype());
    }

    SPU_ENFORCE(runtimeRhs.isPublic());

    auto it = mlir::dyn_cast<IntegerType>(
        getElementTypeOrSelf(callOp->getOperand(1).getType()));
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
  conf.field = ::spu::FM64;
  conf.enable_type_checker = true;
  int numParties = 1;

  switch (ProtocolKind.getValue()) {
  case 1: {
    conf.protocol = ::spu::REF2K;
    numParties = 1;
    break;
  }
  case 2: {
    conf.protocol = ::spu::SEMI2K;
    numParties = 2;
    break;
  }
  case 3: {
    conf.protocol = ::spu::ABY3;
    numParties = 3;
    break;
  }
  case 4: {
    conf.protocol = ::spu::CHEETAH;
    numParties = 2;
    break;
  }
  case 5: {
    conf.protocol = ::spu::SECURENN;
    numParties = 3;
    break;
  }
  case 6: {
    conf.protocol = ::spu::SWIFT;
    numParties = 3;
    break;
  }
  }

  SPDLOG_INFO(conf.DebugString());

  auto entry_function = spu::get_entrypoint(module);
  SPU_ENFORCE(entry_function, "main module not found");

  ::spu::device::pphlo::PPHloExecutor executor;
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
      registry.insert<func::FuncDialect, stablehlo::StablehloDialect,
                      spu::pphlo::PPHloDialect>();
    });

} //  namespace mlir

int main(int argc, char **argv) {
  return static_cast<int>(
      failed(mlir::mlirTranslateMain(argc, argv, "SPU interpreter driver\n")));
}
