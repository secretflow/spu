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

#include <iostream>
#include <vector>

#include "benchmark/benchmark.h"
#include "cryptoTools/Common/CLP.h"
#include "cryptoTools/Common/Defines.h"
#include "cryptoTools/Crypto/PRNG.h"
#include "cryptoTools/Crypto/RandomOracle.h"
#include "cryptoTools/Network/Channel.h"
#include "cryptoTools/Network/IOService.h"
#include "libOTe/NChooseOne/Kkrt/KkrtNcoOtReceiver.h"
#include "libOTe/NChooseOne/Kkrt/KkrtNcoOtSender.h"
#include "libPSI/PSI/Kkrt/KkrtPsiReceiver.h"
#include "libPSI/PSI/Kkrt/KkrtPsiSender.h"

using namespace osuCrypto;

std::vector<block> CreateRangeItems(size_t begin, size_t size) {
  std::vector<block> ret;
  RandomOracle hash(sizeof(block));

  for (size_t i = 0; i < size; i++) {
    std::string stri = std::to_string(begin + i);
    ret.emplace_back();
    hash.Reset();
    hash.Update(stri.c_str(), stri.size());
    hash.Final(ret.back());
  }
  return ret;
}

static void BM_KkrtPsi(benchmark::State &state) {
  uint64_t statSetParam = 40;
  std::string ip = "localhost:1212";

  for (auto _ : state) {
    state.PauseTiming();
    size_t n = state.range(0);
    auto alice_items = CreateRangeItems(1, n);
    auto bob_items = CreateRangeItems(2, n);

    PRNG prng(_mm_set_epi32(4253465, 746587658, 234435, 23987045));

    std::future<void> kkrt_psi_send = std::async([&] {
      auto mode = SessionMode::Server;
      IOService ios;
      Session ses(ios, ip, mode);
      Channel chl = ses.addChannel();

      if (!chl.waitForConnection(std::chrono::milliseconds(1000))) {
        std::cout << "waiting for connection" << std::flush;
        while (!chl.waitForConnection(std::chrono::milliseconds(1000)))
          std::cout << "." << std::flush;
        std::cout << " done" << std::endl;
      }

      KkrtNcoOtSender ot;
      KkrtPsiSender sender;
      sender.init(alice_items.size(), n, statSetParam, chl, ot,
                  prng.get<block>());
      sender.sendInput(alice_items, chl);
    });

    std::future<uint64_t> kkrt_psi_recv = std::async([&] {
      IOService ios;
      auto mode = SessionMode::Client;
      Session ses(ios, ip, mode);
      Channel chl = ses.addChannel();

      if (!chl.waitForConnection(std::chrono::milliseconds(1000))) {
        std::cout << "waiting for connection" << std::flush;
        while (!chl.waitForConnection(std::chrono::milliseconds(1000)))
          std::cout << "." << std::flush;
        std::cout << " done" << std::endl;
      }
      KkrtNcoOtReceiver ot;
      KkrtPsiReceiver recver;
      recver.init(n, bob_items.size(), statSetParam, chl, ot,
                  prng.get<block>());
      recver.sendInput(bob_items, chl);

      return recver.mIntersection.size();
    });

    state.ResumeTiming();

    kkrt_psi_send.get();
    auto psi_result = kkrt_psi_recv.get();
  }
}

// [256k, 512k, 1m, 2m, 4m, 8m]
BENCHMARK(BM_KkrtPsi)
    ->Unit(benchmark::kMillisecond)
    ->Arg(256 << 10)
    ->Arg(512 << 10)
    ->Arg(1 << 20)
    ->Arg(2 << 20)
    ->Arg(4 << 20)
    ->Arg(8 << 20);

BENCHMARK_MAIN();
