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

#pragma once

#include <future>
#include <vector>

#include "yasl/link/link.h"
#include "yasl/link/test_util.h"

namespace spu::mpc::util {

/// This helper macro simulate a secret function with given number of parties.
//
// the type of been simulated function is:
//   fn :: link -> Args... -> Result
//
// the type of the simulator function is:
//   gn :: size_t -> (link -> Args... -> Result) -> std::vector<Result>
//
// for example:
//   auto npc = 2;
//   auto fn = [](const std::shared_ptr<yasl::link::Context>& lctx) {
//     balabala...
//   }
//   simulate(npc, fn);
//
template <typename Fn, typename... Args,
          typename R = std::invoke_result_t<
              Fn, const std::shared_ptr<yasl::link::Context>&, Args...>,
          std::enable_if_t<!std::is_same_v<R, void>, int> = 0>
std::vector<R> simulate(size_t npc, Fn&& fn, Args&&... args) {
  auto lctxs = yasl::link::test::SetupWorld(fmt::format("sim.{}", npc), npc);

  std::vector<R> results;
  std::vector<std::future<R>> futures;
  for (size_t rank = 0; rank < npc; rank++) {
    futures.push_back(std::async(fn, lctxs[rank], std::forward<Args>(args)...));
  }

  for (size_t rank = 0; rank < npc; rank++) {
    results.push_back(futures[rank].get());
  }

  return results;
}

template <typename Fn, typename... Args,
          typename R = std::invoke_result_t<
              Fn, const std::shared_ptr<yasl::link::Context>&, Args...>,
          std::enable_if_t<std::is_same_v<R, void>, int> = 0>
void simulate(size_t npc, Fn&& fn, Args&&... args) {
  auto lctxs = yasl::link::test::SetupWorld(fmt::format("sim.{}", npc), npc);

  std::vector<std::future<void>> futures;
  for (size_t rank = 0; rank < npc; rank++) {
    futures.push_back(std::async(fn, lctxs[rank], std::forward<Args>(args)...));
  }

  for (size_t rank = 0; rank < npc; rank++) {
    futures[rank].get();
  }
}

}  // namespace spu::mpc::util
