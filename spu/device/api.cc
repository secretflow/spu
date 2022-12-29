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

#include "spu/device/api.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <vector>

#include "llvm/Support/ErrorHandling.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Parser/Parser.h"
#include "spdlog/spdlog.h"

#include "spu/device/pphlo/pphlo_executor.h"
#include "spu/dialect/pphlo_dialect.h"

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

void dumpExecutableToFolder(const ExecutableProto &executable, size_t rank,
                            absl::Span<spu::Value const> inputs,
                            const std::string &dump_dir) {
  // Naming convention for dumped files must align with debug runner.
  std::filesystem::path dump_folder(dump_dir);
  dump_folder /= executable.name();

  std::filesystem::create_directories(dump_folder);

  // dump executable.
  if (rank == 0) {
    auto fname = dump_folder / std::string("executable.txt");
    SPDLOG_INFO("Dump executable to {}", fname);
    std::ofstream ir_file(fname, std::ios::binary | std::ios::out);
    ir_file << executable.SerializeAsString();
  }

  // dump all inputs.
  {
    size_t var_counter = 0;
    for (const auto &val : inputs) {
      auto fname =
          dump_folder / fmt::format("data_{}_{}.txt", rank, var_counter++);
      SPDLOG_INFO("Dump data to {}", fname);
      std::ofstream inputs_file(fname, std::ios::binary | std::ios::out);
      inputs_file << val.toProto().SerializeAsString();
    }
  }
}

void printProfilingData(spu::HalContext *hctx, const std::string &name,
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

    const auto &tracer = GET_TRACER(hctx);
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
      SPDLOG_INFO("{} profiling: total time {}", mod_name, total_time);
      for (const auto &[key, stat] : stats) {
        if ((key.flag & mod_flag) != 0) {
          SPDLOG_INFO("- {}, executed {} times, duration {}s", key.name,
                      stat.count, stat.getTotalTimeInSecond());
        }
      }
    }
  }

  // print link statistics
  SPDLOG_INFO("Link details: total send bytes {}, send actions {}",
              comm_stats.send_bytes, comm_stats.send_actions);
}

void setupTrace(spu::HalContext *hctx, const spu::RuntimeConfig &rt_config) {
  int64_t tr_flag = 0;
  if (rt_config.enable_action_trace()) {
    tr_flag |= TR_LOG;
  }

  if (rt_config.enable_pphlo_profile()) {
    tr_flag |= TR_HLO;
    tr_flag |= TR_REC;
  }

  if (rt_config.enable_hal_profile()) {
    tr_flag |= TR_HAL | TR_MPC;
    tr_flag |= TR_REC;
  }

  initTrace(tr_flag);
  GET_TRACER(hctx)->getProfState()->clearRecords();
}

void SPUErrorHandler(void *use_data, const char *reason, bool gen_crash_diag) {
  (void)use_data;
  (void)gen_crash_diag;
  YACL_THROW(reason);
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

} // namespace

void executeImpl(OpExecutor *executor, spu::HalContext *hctx,
                 const ExecutableProto &executable, SymbolTable *env) {
  setupTrace(hctx, hctx->rt_config());
  installLLVMErrorHandler();

  CommunicationStats comm_stats;
  comm_stats.reset(hctx->lctx());
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

  // TODO: rename this flag, enable_executable_dump?
  const RuntimeConfig rt_config = hctx->rt_config();
  if (rt_config.enable_processor_dump()) {
    const bool isRefHal = hctx->lctx() == nullptr;
    const size_t rank = isRefHal ? 0 : hctx->lctx()->Rank();
    dumpExecutableToFolder(executable, rank, inputs,
                           rt_config.processor_dump_dir());
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

    YACL_ENFORCE(moduleOpRef, "MLIR parser failure");

    auto entry_function = moduleOpRef->lookupSymbol<mlir::func::FuncOp>("main");
    YACL_ENFORCE(entry_function, "main module not found");

    ExecutionOptions opts;
    opts.do_type_check = rt_config.enable_type_checker();
    opts.do_log_execution = rt_config.enable_pphlo_trace();
    opts.do_parallel = false;
    outputs = runRegion(executor, hctx, nullptr, entry_function.getBody(),
                        inputs, opts);
  }

  // sync output to environment.
  {
    TimeitGuard timeit(exec_stats.outfeed_time);
    for (int32_t idx = 0; idx < executable.output_names_size(); idx++) {
      env->setVar(executable.output_names(idx), outputs[idx]);
    }
  }

  comm_stats.diff(hctx->lctx());
  if ((getGlobalTraceFlag() & TR_REC) != 0) {
    printProfilingData(hctx, executable.name(), exec_stats, comm_stats);
  }
}

void execute(OpExecutor *executor, spu::HalContext *hctx,
             const spu::ExecutableProto &executable, SymbolTable *env) {
  return executeImpl(executor, hctx, executable, env);
}

void execute(OpExecutor *executor, spu::HalContext *hctx,
             const std::string &text,
             const std::vector<std::string> &input_names,
             const std::vector<std::string> &output_names, SymbolTable *env) {
  ExecutableProto executable;
  executable.set_name("unnamed");
  *executable.mutable_input_names() = {input_names.begin(), input_names.end()};
  *executable.mutable_output_names() = {output_names.begin(),
                                        output_names.end()};
  executable.set_code(text);

  return executeImpl(executor, hctx, executable, env);
}

} // namespace spu::device
