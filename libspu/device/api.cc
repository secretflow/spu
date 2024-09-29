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

#include "libspu/device/api.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <vector>

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "spdlog/spdlog.h"

#include "libspu/kernel/hal/debug.h"

// Depending dialects
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "libspu/core/trace.h"
#include "libspu/core/type_util.h"
#include "libspu/device/utils/debug_dump_constant.h"
#include "libspu/dialect/ring/IR/dialect.h"
#include "libspu/dialect/utils/utils.h"
#include "libspu/version.h"

// Dispatcher
#include "libspu/device/arith/dispatcher.h"
#include "libspu/device/func/dispatcher.h"
#include "libspu/device/linalg/dispatcher.h"
#include "libspu/device/math/dispatcher.h"
#include "libspu/device/ring/dispatcher.h"
#include "libspu/device/scf/dispatcher.h"
#include "libspu/device/tensor/dispatcher.h"

namespace spu::device {
namespace {

class TimeitGuard {
  TimePoint start_;
  Duration &duration_;

 public:
  explicit TimeitGuard(Duration &dur) : duration_(dur) {
    start_ = std::chrono::high_resolution_clock::now();
  }

  ~TimeitGuard() {
    duration_ = std::chrono::high_resolution_clock::now() - start_;
  }
};

double getSeconds(const Duration &dur) {
  return std::chrono::duration_cast<std::chrono::duration<double>>(dur).count();
}

[[maybe_unused]] double getSeconds(const TimePoint &start,
                                   const TimePoint &end) {
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

struct CommunicationStats {
  size_t send_bytes = 0;
  size_t recv_bytes = 0;
  size_t send_actions = 0;
  size_t recv_actions = 0;

  void reset(const std::shared_ptr<yacl::link::Context> &lctx) {
    if (!lctx) {
      return;
    }
    send_actions = lctx->GetStats()->sent_actions;
    recv_actions = lctx->GetStats()->recv_actions;
    send_bytes = lctx->GetStats()->sent_bytes;
    recv_bytes = lctx->GetStats()->recv_bytes;
  }

  void diff(const std::shared_ptr<yacl::link::Context> &lctx) {
    if (!lctx) {
      return;
    }
    send_bytes = lctx->GetStats()->sent_bytes - send_bytes;
    recv_bytes = lctx->GetStats()->recv_bytes - recv_bytes;
    send_actions = lctx->GetStats()->sent_actions - send_actions;
    recv_actions = lctx->GetStats()->recv_actions - recv_actions;
  }
};

struct ActionKey {
  std::string_view name;
  int64_t flag;
  bool operator<(const ActionKey &other) const {
    return std::tie(name, flag) < std::tie(other.name, other.flag);
  }
};

struct ActionStats {
  // number of actions executed.
  size_t count = 0;
  // total duration time.
  Duration total_time = {};
  // total send bytes.
  size_t send_bytes = 0;
  // total recv bytes.
  size_t recv_bytes = 0;
  // total send actions.
  size_t send_actions = 0;
  // total recv actions.
  size_t recv_actions = 0;

  inline double getTotalTimeInSecond() const {
    return std::chrono::duration_cast<std::chrono::duration<double>>(total_time)
        .count();
  }
};

void takeSnapshot(size_t rank, const RuntimeConfig &rt_config,
                  const ExecutableProto &executable, const SymbolTable &env) {
  const std::string &dump_dir = rt_config.snapshot_dump_dir();
  // Naming convention for dumped files must align with debug runner.
  std::filesystem::path dump_folder(dump_dir);
  std::filesystem::create_directories(dump_folder);

  // Dump executable
  {
    std::ofstream config_file(getConfigFilePath(dump_folder),
                              std::ios::binary | std::ios::out);
    config_file << rt_config.SerializeAsString();
  }

  // Dump executable
  {
    std::ofstream main_file(getCodeFilePath(dump_folder),
                            std::ios::binary | std::ios::out);
    main_file << executable.SerializeAsString();
  }

  auto value_dump_dir = getRankFolder(dump_folder, rank);
  std::filesystem::create_directories(value_dump_dir);

  // Dump inputs
  for (const auto &[name, var_dtype_pair] : env) {
    auto serialized =
        var_dtype_pair.first.toProto(std::numeric_limits<int>::max());
    {
      std::ofstream meta_file(getMetaFilePath(dump_folder, rank, name),
                              std::ios::binary | std::ios::out);
      meta_file << serialized.meta.SerializeAsString();
    }
    {
      for (const auto &chunk : llvm::enumerate(serialized.chunks)) {
        std::ofstream chunk_file(
            getValueChunkFilePath(dump_folder, rank, name, chunk.index()),
            std::ios::binary | std::ios::out);
        chunk_file << chunk.value().SerializeAsString();
      }
    }
  }
}

void printProfilingData(spu::SPUContext *sctx, const std::string &name,
                        const ExecutionStats &exec_stats,
                        const CommunicationStats &comm_stats) {
  // print overall information
  SPDLOG_INFO(
      "[Profiling] SPU execution {} completed, input processing took {}s, "
      "execution took {}s, output processing took {}s, total time {}s.",
      name, getSeconds(exec_stats.infeed_time),
      getSeconds(exec_stats.execution_time),
      getSeconds(exec_stats.outfeed_time), getSeconds(exec_stats.total_time()));

  // print action trace information
  {
    std::map<ActionKey, ActionStats> stats;

    const auto &tracer = GET_TRACER(sctx);
    const auto &records = tracer->getProfState()->getRecords();

    for (const auto &rec : records) {
      auto &stat = stats[{rec.name, rec.flag}];
      stat.count++;
      stat.total_time +=
          std::chrono::duration_cast<Duration>(rec.end - rec.start);
      stat.send_bytes += (rec.send_bytes_end - rec.send_bytes_start);
      stat.recv_bytes += (rec.recv_bytes_end - rec.recv_bytes_start);
      stat.send_actions += (rec.send_actions_end - rec.send_actions_start);
      stat.recv_actions += (rec.recv_actions_end - rec.recv_actions_start);
    }

    static std::map<int64_t, std::string> kModules = {
        {TR_HLO, "HLO"}, {TR_HAL, "HAL"}, {TR_MPC, "MPC"}};

    for (const auto &[mod_flag, mod_name] : kModules) {
      if ((tracer->getFlag() & mod_flag) == 0) {
        continue;
      }

      double total_time = 0.0;
      std::vector<ActionKey> sorted_by_time;
      for (const auto &[key, stat] : stats) {
        if ((key.flag & mod_flag) != 0) {
          total_time += stat.getTotalTimeInSecond();
          sorted_by_time.emplace_back(key);
        }
      }

      std::sort(sorted_by_time.begin(), sorted_by_time.end(),
                [&](const auto &k0, const auto &k1) {
                  return stats.find(k0)->second.getTotalTimeInSecond() >
                         stats.find(k1)->second.getTotalTimeInSecond();
                });

      SPDLOG_INFO("{} profiling: total time {}", mod_name, total_time);
      size_t instruction_counter = 0;
      for (const auto &key : sorted_by_time) {
        const auto &stat = stats.find(key)->second;
        instruction_counter += stat.count;
        SPDLOG_INFO(
            "- {}, executed {} times, duration {}s, send bytes {} recv "
            "bytes {}, send actions {}, recv actions {}",
            key.name, stat.count, stat.getTotalTimeInSecond(), stat.send_bytes,
            stat.recv_bytes, stat.send_actions, stat.recv_actions);
      }
      SPDLOG_INFO("number of instructions executed = {}", instruction_counter);
    }
  }

  // print link statistics
  SPDLOG_INFO(
      "Link details: total send bytes {}, recv bytes {}, send actions {}, recv "
      "actions {}",
      comm_stats.send_bytes, comm_stats.recv_bytes, comm_stats.send_actions,
      comm_stats.recv_actions);
}

void SPUErrorHandler(void *use_data, const char *reason, bool gen_crash_diag) {
  (void)use_data;
  (void)gen_crash_diag;
  SPU_THROW("{}", reason);
}

std::mutex ErrorHandlerMutex;
void installLLVMErrorHandler() {
  std::lock_guard<std::mutex> guard(ErrorHandlerMutex);
  llvm::remove_fatal_error_handler();
  llvm::install_fatal_error_handler(SPUErrorHandler);
}

[[maybe_unused]] void removeLLVMErrorHandler() {
  std::lock_guard<std::mutex> guard(ErrorHandlerMutex);
  llvm::remove_fatal_error_handler();
}

}  // namespace

void executeImpl(OpExecutor *executor, spu::SPUContext *sctx,
                 const ExecutableProto &executable, SymbolTable *env) {
  setupTrace(sctx, sctx->config());
  installLLVMErrorHandler();

  CommunicationStats comm_stats;
  comm_stats.reset(sctx->lctx());
  ExecutionStats exec_stats;

  // prepare inputs from environment.
  std::vector<spu::MemRef> inputs;
  {
    TimeitGuard timeit(exec_stats.infeed_time);
    inputs.reserve(executable.input_names_size());
    for (int32_t idx = 0; idx < executable.input_names_size(); idx++) {
      inputs.emplace_back(env->getVar(executable.input_names(idx)).first);
    }
  }

  const RuntimeConfig rt_config = sctx->config();

  if (rt_config.enable_runtime_snapshot()) {
    const bool isRefHal = sctx->lctx() == nullptr;
    const size_t rank = isRefHal ? 0 : sctx->lctx()->Rank();
    takeSnapshot(rank, rt_config, executable, *env);
  }

  // execution
  std::vector<spu::MemRef> outputs;
  {
    TimeitGuard timeit(exec_stats.execution_time);

    mlir::MLIRContext mlir_ctx;
    mlir_ctx.loadDialect<mlir::tensor::TensorDialect,
                         mlir::spu::ring::RingDialect, mlir::func::FuncDialect,
                         mlir::scf::SCFDialect, mlir::linalg::LinalgDialect>();

    auto &engine = mlir_ctx.getDiagEngine();
    engine.registerHandler(
        [&](mlir::Diagnostic &diag) { SPDLOG_ERROR(diag.str()); });

    auto moduleOpRef =
        mlir::parseSourceString<mlir::ModuleOp>(executable.code(), &mlir_ctx);

    SPU_ENFORCE(moduleOpRef, "MLIR parser failure");

    if (!moduleOpRef.get()->hasAttr("ring.version")) {
      // There are tests that has no version attributes.
      // So treats this as a warning
      SPDLOG_WARN("Missing ir version");
    } else {
      auto ir_version = mlir::dyn_cast<mlir::StringAttr>(
                            moduleOpRef.get()->getAttr("ring.version"))
                            .str();
      if (ir_version != getVersionStr()) {
        SPU_THROW(
            "IR was generated by compiler {} and does not match current "
            "runtime "
            "{}",
            ir_version, getVersionStr());
      }
    }

    auto entry_function = mlir::spu::get_entrypoint(moduleOpRef.get());
    SPU_ENFORCE(entry_function, "main module not found");

    ExecutionOptions opts;
    opts.do_type_check = rt_config.enable_type_checker();
    opts.do_log_execution = rt_config.enable_pphlo_trace();
    opts.do_parallel = rt_config.experimental_enable_inter_op_par();

    if (opts.do_parallel) {
      opts.concurrency = rt_config.experimental_inter_op_concurrency();
      mlir_ctx.enableMultithreading();
      mlir_ctx.enterMultiThreadedExecution();
    }
    outputs = runRegion(executor, sctx, nullptr, entry_function.getBody(),
                        inputs, opts);

    if (opts.do_parallel) {
      mlir_ctx.exitMultiThreadedExecution();
    }
  }

  // sync output to environment.
  {
    TimeitGuard timeit(exec_stats.outfeed_time);
    for (int32_t idx = 0; idx < executable.output_names_size(); idx++) {
      const auto &output_dtype = executable.output_dtypes(idx);
      env->setVar(executable.output_names(idx), outputs[idx],
                  PyFormatToPtType(output_dtype));
    }
  }

  comm_stats.diff(sctx->lctx());
  if ((getGlobalTraceFlag(sctx->id()) & TR_REC) != 0) {
    printProfilingData(sctx, executable.name(), exec_stats, comm_stats);
  }
}

void execute(OpExecutor *executor, spu::SPUContext *sctx,
             const spu::ExecutableProto &executable, SymbolTable *env) {
  return executeImpl(executor, sctx, executable, env);
}

void execute(OpExecutor *executor, spu::SPUContext *sctx,
             const std::string &text,
             const std::vector<std::string> &input_names,
             const std::vector<std::string> &output_names, SymbolTable *env) {
  ExecutableProto executable;
  executable.set_name("unnamed");
  *executable.mutable_input_names() = {input_names.begin(), input_names.end()};
  *executable.mutable_output_names() = {output_names.begin(),
                                        output_names.end()};
  executable.set_code(text);

  return executeImpl(executor, sctx, executable, env);
}

void ConcreteExecutor::runKernelImpl(SPUContext *sctx, SymbolScope *sscope,
                                     mlir::Operation &op,
                                     const ExecutionOptions &opts) {
  llvm::TypeSwitch<mlir::Dialect *, void>(op.getDialect())
      .Case([&](mlir::arith::ArithDialect *d) {
        dispatch(d, sctx, sscope, op, this, opts);
      })
      .Case([&](mlir::math::MathDialect *d) {
        dispatch(d, sctx, sscope, op, this, opts);
      })
      .Case([&](mlir::linalg::LinalgDialect *d) {
        dispatch(d, sctx, sscope, op, this, opts);
      })
      .Case([&](mlir::scf::SCFDialect *d) {
        dispatch(d, sctx, sscope, op, this, opts);
      })
      .Case([&](mlir::func::FuncDialect *d) {
        dispatch(d, sctx, sscope, op, this, opts);
      })
      .Case([&](mlir::tensor::TensorDialect *d) {
        dispatch(d, sctx, sscope, op, this, opts);
      })
      .Case([&](mlir::spu::ring::RingDialect *d) {
        dispatch(d, sctx, sscope, op, this, opts);
      })
      .Default([&](mlir::Dialect *d) {
        SPU_THROW("Unknown dialect = {}", d->getNamespace().str());
      });
  if (opts.do_log_execution) {
    if (op.getNumResults() != 0) {
      auto key = op.getResult(0);
      if (sscope->hasValue(key)) {
        auto v = sscope->lookupValue(key);
        auto v_p = kernel::hal::dbg_print<int64_t>(sctx, v);
        if (sctx->lctx()->Rank() == 0) {
          SPDLOG_INFO("op {}, v {}, type {}", mlir::spu::mlirObjectToString(op),
                      v_p, v.eltype());
        }
      }
    }
  }
}

}  // namespace spu::device
