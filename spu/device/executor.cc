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

#include "spu/device/executor.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <vector>

#include "spdlog/spdlog.h"

#include "spu/kernel/context.h"

namespace spu::device {

void Executor::runWithEnv(const ExecutableProto &exec, SymbolTable *env) {
  // setup global states.
  const RuntimeConfig rt_config = hctx_->rt_config();

  //
  const bool isRefHal = hctx_->lctx() == nullptr;
  const size_t rank = isRefHal ? 0 : hctx_->lctx()->Rank();

  module_name_ = exec.name();

  Timer timer;
  Timer stage_timer;

  // prepare inputs from environment.
  std::vector<spu::Value> inputs;
  inputs.reserve(exec.input_names_size());
  for (int32_t idx = 0; idx < exec.input_names_size(); idx++) {
    const std::string &sym_name = exec.input_names(idx);
    inputs.emplace_back(env->getVar(sym_name));
  }

  const auto input_time = stage_timer.count();

  // TODO: rename this flag, enable_executable_dump?
  if (rt_config.enable_processor_dump()) {
    // Naming convention for dumped files must align with debug runner.
    std::filesystem::path dump_folder(rt_config.processor_dump_dir());
    dump_folder /= exec.name();

    std::filesystem::create_directories(dump_folder);

    // dump executable.
    if (rank == 0) {
      auto fname = dump_folder / std::string("exec.txt");
      SPDLOG_INFO("Dump exec to {}", fname);
      std::ofstream ir_file(fname, std::ios::binary | std::ios::out);
      ir_file << exec.SerializeAsString();
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

  // Profile: before execution stamp
  stage_timer.reset();
  auto outputs = run(exec.code(), inputs);
  const auto exec_time = stage_timer.count();

  // sync output to environment.
  stage_timer.reset();
  for (int32_t idx = 0; idx < exec.output_names_size(); idx++) {
    const std::string &sym_name = exec.output_names(idx);
    env->setVar(sym_name, outputs[idx]);
  }
  const auto output_time = stage_timer.count();

  // Collect time profile data
  auto total_time = timer.count();

  // Only one party prints for multi-threading simulation
  if (hctx_->rt_config().enable_pphlo_profile()) {
    SPDLOG_INFO(
        "[Profiling] SPU execution {} completed, input processing took {}s, "
        "execution took {}s, output processing took {}s, total time {}s.",
        module_name_, input_time.count(), exec_time.count(),
        output_time.count(), total_time.count());
    const auto &records = getProfileRecords();
    double total_time = .0;
    for (const auto &[name, record] : records) {
      total_time += record.time.count();
    }
    SPDLOG_INFO("HLO profiling: total time: {}", total_time);
    for (const auto &[name, record] : records) {
      SPDLOG_INFO("- {}, executed {} times, duration {}s", name, record.count,
                  record.time.count());
    }
  }

  if (hctx_->getProfilingEnabled()) {
    const auto &records = hctx_->getActionStats();
    double total_time = .0;
    for (const auto &[_, record] : records) {
      total_time += record.getTotalTimeInSecond();
    }
    SPDLOG_INFO("HAL profiling: total time {}", total_time);
    for (const auto &[name, record] : records) {
      SPDLOG_INFO("- {}, executed {} times, duration {}s", name, record.count,
                  record.getTotalTimeInSecond());
    }
  }

  if (hctx_->prot()->getProfilingEnabled()) {
    const auto &records = hctx_->prot()->getActionStats();
    double total_time = .0;
    for (const auto &[_, record] : records) {
      total_time += record.getTotalTimeInSecond();
    }
    SPDLOG_INFO("MPC profiling: total time {}", total_time);
    for (const auto &[name, record] : records) {
      SPDLOG_INFO("- {}, executed {} times, duration {}s", name, record.count,
                  record.getTotalTimeInSecond());
    }
  }
}

void Executor::runWithEnv(const std::string &text,
                          const std::vector<std::string> &input_names,
                          const std::vector<std::string> &output_names,
                          SymbolTable *env) {
  ExecutableProto exec;
  exec.set_name("unnamed");
  *exec.mutable_input_names() = {input_names.begin(), input_names.end()};
  *exec.mutable_output_names() = {output_names.begin(), output_names.end()};
  exec.set_code(text);

  runWithEnv(exec, env);
}

} // namespace spu::device
