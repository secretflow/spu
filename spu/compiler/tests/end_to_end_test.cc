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

#include <filesystem>
#include <fstream>
#include <string_view>

#include "fmt/format.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "spdlog/spdlog.h"

#include "spu/compiler/common/compilation_context.h"
#include "spu/compiler/compile.h"

namespace spu::compiler {

namespace {

std::string readFileContent(const std::filesystem::path &in) {
  if (!std::filesystem::exists(in)) {
    spdlog::error("File {} does not exist!", in.c_str());
    assert(false);
    return {};
  }

  std::ifstream in_stream(in);
  std::string contents{std::istreambuf_iterator<char>{in_stream}, {}};
  return contents;
}

std::string runCompile(const std::filesystem::path &in) {
  CompilationContext ctx;
  return spu::compiler::compile(&ctx, in);
}

void RunTest(std::string_view test_name) {
  std::filesystem::path in(
      fmt::format("spu/compiler/test_data/{}.hlo.pb", test_name));
  std::filesystem::path out(
      fmt::format("spu/compiler/test_data/{}.mlir", test_name));

  ASSERT_EQ(runCompile(in), readFileContent(out));
}

} // namespace

TEST(E2E, Basic) { RunTest("jit_comp"); }

TEST(E2E, Empty_Return_Tuple) { RunTest("empty_return_tuple"); }

TEST(E2E, LR) { RunTest("jit_keras"); }

TEST(E2E, Credit_Fraud_train) {
  RunTest("credit_fraud.train_step.before_optimizations");
}

TEST(E2E, Credit_Fraud_test) {
  RunTest("credit_fraud.test_step.before_optimizations");
}

TEST(E2E, Credit_Fraud_predict) {
  RunTest("credit_fraud.predict_step.before_optimizations");
}

TEST(E2E, Credit_Fraud_Partial_Metrics_train) {
  RunTest("credit_fraud_metrics.train_step.before_optimizations");
}

TEST(E2E, Credit_Fraud_Partial_Metrics_test) {
  RunTest("credit_fraud_metrics.test_step.before_optimizations");
}

TEST(E2E, Credit_Fraud_Partial_Metrics_predict) {
  RunTest("credit_fraud_metrics.predict_step.before_optimizations");
}
TEST(E2E, Jax_cnn) { RunTest("jax_cnn"); }

} // namespace spu::compiler
