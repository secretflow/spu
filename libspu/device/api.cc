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

#include "llvm/Support/ErrorHandling.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Parser/Parser.h"
#include "spdlog/spdlog.h"

#include "libspu/device/pphlo/pphlo_executor.h"
#include "libspu/dialect/pphlo_dialect.h"

#include "libspu/device/device.pb.h"

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
  size_t send_actions = 0;

  void reset(const std::shared_ptr<yacl::link::Context> &lctx) {
    if (!lctx) {
      return;
    }
    send_actions = lctx->GetStats()->sent_actions;
    send_bytes = lctx->GetStats()->sent_bytes;
  }

  void diff(const std::shared_ptr<yacl::link::Context> &lctx) {
    if (!lctx) {
      return;
    }
    send_bytes = lctx->GetStats()->sent_bytes - send_bytes;
    send_actions = lctx->GetStats()->sent_actions - send_actions;
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

  inline double getTotalTimeInSecond() const {
    return std::chrono::duration_cast<std::chrono::duration<double>>(total_time)
        .count();
  }
};

void takeSnapshot(size_t rank, const RuntimeConfig &rt_config,
                  const ExecutableProto &executable, const SymbolTable &env) {
  const std::string &dump_dir = rt_config.processor_dump_dir();
  // Naming convention for dumped files must align with debug runner.
  std::filesystem::path dump_folder(dump_dir);
  std::filesystem::create_directories(dump_folder);
  auto dump_path = dump_folder / fmt::format("snapshot_{}.spu", rank);

  SnapshotProto snapshot;
  snapshot.set_rank(rank);
  *snapshot.mutable_executable() = executable;
  *snapshot.mutable_runtime_cfg() = rt_config;
  *snapshot.mutable_environ() = env.toProto();

  std::ofstream dump_file(dump_path, std::ios::binary | std::ios::out);
  dump_file << snapshot.SerializeAsString();
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
    }

    static std::map<int64_t, std::string> kModules = {
        {TR_HLO, "HLO"}, {TR_HAL, "HAL"}, {TR_MPC, "MPC"}};

    for (const auto &[mod_flag, mod_name] : kModules) {
      double total_time = 0.0;
      for (const auto &[key, stat] : stats) {
        if ((key.flag & mod_flag) != 0) {
          total_time += stat.getTotalTimeInSecond();
        }
      }
      if ((tracer->getFlag() & mod_flag) != 0) {
        SPDLOG_INFO("{} profiling: total time {}", mod_name, total_time);
        for (const auto &[key, stat] : stats) {
          if ((key.flag & mod_flag) != 0) {
            SPDLOG_INFO("- {}, executed {} times, duration {}s", key.name,
                        stat.count, stat.getTotalTimeInSecond());
          }
        }
      }
    }
  }

  // print link statistics
  SPDLOG_INFO("Link details: total send bytes {}, send actions {}",
              comm_stats.send_bytes, comm_stats.send_actions);
}

void SPUErrorHandler(void *use_data, const char *reason, bool gen_crash_diag) {
  (void)use_data;
  (void)gen_crash_diag;
  SPU_THROW(reason);
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
  std::vector<spu::Value> inputs;
  {
    TimeitGuard timeit(exec_stats.infeed_time);
    inputs.reserve(executable.input_names_size());
    for (int32_t idx = 0; idx < executable.input_names_size(); idx++) {
      inputs.emplace_back(env->getVar(executable.input_names(idx)));
    }
  }

  // TODO: rename this flag, enable_execution_dump?
  const RuntimeConfig rt_config = sctx->config();
  if (rt_config.enable_processor_dump()) {
    const bool isRefHal = sctx->lctx() == nullptr;
    const size_t rank = isRefHal ? 0 : sctx->lctx()->Rank();
    takeSnapshot(rank, rt_config, executable, *env);
  }

  // execution
  std::vector<spu::Value> outputs;
  {
    TimeitGuard timeit(exec_stats.execution_time);

    mlir::MLIRContext mlir_ctx;
    mlir_ctx.loadDialect<mlir::pphlo::PPHloDialect, mlir::func::FuncDialect>();

    auto &engine = mlir_ctx.getDiagEngine();
    engine.registerHandler(
        [&](mlir::Diagnostic &diag) { SPDLOG_ERROR(diag.str()); });

    mlir_ctx.loadDialect<mlir::pphlo::PPHloDialect, mlir::func::FuncDialect>();
    auto moduleOpRef =
        mlir::parseSourceString<mlir::ModuleOp>(executable.code(), &mlir_ctx);

    SPU_ENFORCE(moduleOpRef, "MLIR parser failure");

    auto entry_function = moduleOpRef->lookupSymbol<mlir::func::FuncOp>("main");
    SPU_ENFORCE(entry_function, "main module not found");

    ExecutionOptions opts;
    opts.do_type_check = rt_config.enable_type_checker();
    opts.do_log_execution = rt_config.enable_pphlo_trace();
    opts.do_parallel = rt_config.experimental_enable_inter_op_par();
    if (opts.do_parallel) {
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
      env->setVar(executable.output_names(idx), outputs[idx]);
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

}  // namespace spu::device
