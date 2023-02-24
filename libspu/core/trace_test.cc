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

#include "libspu/core/trace.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "spdlog/sinks/ostream_sink.h"
#include "spdlog/spdlog.h"

namespace spu {
namespace {

template <typename... Args>
std::shared_ptr<spdlog::logger> makeSStreamLogger(std::ostringstream& oss) {
  auto oss_sink = std::make_shared<spdlog::sinks::ostream_sink_mt>(oss);
  auto oss_logger = std::make_shared<spdlog::logger>("oss_logger", oss_sink);
  // oss_logger->set_level(spdlog::level::info);
  return oss_logger;
}

}  // namespace

TEST(TraceTest, TracerLogWorks) {
  std::ostringstream oss;
  initTrace("id", TR_MODALL | TR_LAR, makeSStreamLogger(oss));

  Tracer tracer(TR_MODALL | TR_LAR);
  tracer.logActionBegin(1, "f", "");
  tracer.logActionBegin(2, "g", "");
  tracer.logActionEnd(2, "g", "");
  tracer.logActionEnd(1, "h", "");
}

TEST(TraceTest, ActionWorks) {
  std::ostringstream oss;
  initTrace("id", TR_MODALL | TR_LAR, makeSStreamLogger(oss));

  auto tracer = std::make_shared<Tracer>(TR_MODALL | TR_LAR);
  {
    TraceAction ta0(tracer, (TR_MOD1 | TR_LOG), ~0, "f");
    TraceAction ta1(tracer, (TR_MOD1 | TR_LAR), ~TR_MOD1, "g", 10);
    TraceAction ta2(tracer, (TR_MOD1 | TR_LAR), ~TR_MOD1, "ignored", 10);
    TraceAction ta3(tracer, (TR_MOD2 | TR_LAR), ~0, "h", 10, 20);
  }

  ASSERT_EQ(tracer->getProfState()->getRecords().size(), 2);
  EXPECT_EQ(tracer->getProfState()->getRecords()[0].name, "h");
  EXPECT_EQ(tracer->getProfState()->getRecords()[1].name, "g");
}

/// macros examples.
struct Context {
  static std::string id() { return "id"; }
  static std::string pid() { return ""; }
};
void g(Context* ctx) { SPU_TRACE_HAL_LEAF(ctx); }

void h(Context* ctx) { SPU_TRACE_HAL_DISP(ctx); }

void f(Context* ctx) {
  SPU_TRACE_HAL_DISP(ctx);
  g(ctx);
  h(ctx);
};

TEST(TraceTest, Example) {
  std::ostringstream oss;
  initTrace("id", TR_MODALL | TR_LAR, makeSStreamLogger(oss));
  Context ctx;
  f(&ctx);

  auto tracer = GET_TRACER(&ctx);

  ASSERT_EQ(tracer->getProfState()->getRecords().size(), 1);
  EXPECT_EQ(tracer->getProfState()->getRecords()[0].name, "g");
  // std::cout << oss.str() << std::endl;
}

}  // namespace spu
