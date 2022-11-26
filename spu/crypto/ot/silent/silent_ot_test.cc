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

#include "spu/crypto/ot/silent/silent_ot.h"

#include <limits.h>
#include <stdio.h>
#include <unistd.h>

#include <future>
#include <thread>

#include "gtest/gtest.h"
#include "yacl/link/test_util.h"

// https://gcc.gnu.org/bugzilla/show_bug.cgi?id=69884
#if defined __GNUC__ && __GNUC__ >= 6
#pragma GCC diagnostic ignored "-Wignored-attributes"
#endif

namespace spu {

const static int threads = 1;

uint64_t total_sent(IO *ios[threads]) {
  uint64_t sent = 0;
  for (int i = 0; i < threads; i++) sent += ios[i]->counter;
  return sent;
}

double test_ot_cm_cc(SilentOT *ot, IO *ios[threads], int party,
                     int64_t length) {
  std::vector<block> b0(length);
  std::vector<block> b1(length);
  std::vector<block> r(length);
  PRG prg(fix_key);
  prg.random_block(b0.data(), length);
  prg.random_block(b1.data(), length);
  std::vector<uint8_t> b(length);
  PRG prg2;
  prg2.random_bool(reinterpret_cast<bool *>(b.data()), length);

  uint64_t sent = total_sent(ios);

  auto start = clock_start();
  if (party == ALICE) {
    ot->send_ot_cm_cc(b0.data(), b1.data(), length);
  } else {
    ot->recv_ot_cm_cc(r.data(), reinterpret_cast<bool *>(b.data()), length);
  }
  ios[0]->flush();
  long long t = time_from(start);

  std::cout << (party == ALICE ? "[ALICE]\t" : "[BOB]\t")
            << "Silent OT (cm, cc)\t" << t << " us"
            << ", " << (total_sent(ios) - sent) << " bytes" << endl;

  if (party == BOB) {
    for (int64_t i = 0; i < length; ++i) {
      if (b[i] != 0) {
        if (!cmpBlock(&r[i], &b1[i], 1)) {
          std::cout << i << "\n";
          error("wrong!\n");
        }
      } else {
        if (!cmpBlock(&r[i], &b0[i], 1)) {
          std::cout << i << "\n";
          error("wrong!\n");
        }
      }
    }
  }
  std::cout << (party == ALICE ? "[ALICE]\t" : "[BOB]\t") << "Tests passed.\t"
            << std::endl;
  return t;
}

double test_ot_rm_cc(SilentOT *ot, IO *ios[threads], int party,
                     int64_t length) {
  std::vector<block> b0(length);
  std::vector<block> r(length);
  std::vector<block> b1(length);
  std::vector<uint8_t> b(length);
  PRG prg;
  prg.random_bool(reinterpret_cast<bool *>(b.data()), length);

  uint64_t sent = total_sent(ios);

  auto start = clock_start();
  if (party == ALICE) {
    ot->send_ot_rm_cc(b0.data(), b1.data(), length);
  } else {
    ot->recv_ot_rm_cc(r.data(), reinterpret_cast<bool *>(b.data()), length);
  }
  ios[0]->flush();
  long long t = time_from(start);

  std::cout << (party == ALICE ? "[ALICE]\t" : "[BOB]\t")
            << "Silent OT (rm, cc)\t" << t << " us"
            << ", " << (total_sent(ios) - sent) << " bytes" << endl;

  if (party == ALICE) {
    ios[0]->send_block(b0.data(), length);
    ios[0]->send_block(b1.data(), length);
  } else if (party == BOB) {
    ios[0]->recv_block(b0.data(), length);
    ios[0]->recv_block(b1.data(), length);
    for (int64_t i = 0; i < length; ++i) {
      if (b[i] != 0) {
        assert(cmpBlock(&r[i], &b1[i], 1));
      } else {
        assert(cmpBlock(&r[i], &b0[i], 1));
      }
    }
  }
  std::cout << (party == ALICE ? "[ALICE]\t" : "[BOB]\t") << "Tests passed.\t"
            << std::endl;
  ios[0]->flush();
  return t;
}

double test_ot_rcm_cc(SilentOT *ot, IO *ios[threads], int party,
                      int64_t length) {
  std::vector<block> b0(length);
  std::vector<block> r(length);
  std::vector<uint8_t> b(length);
  block delta;
  PRG prg;
  prg.random_bool(reinterpret_cast<bool *>(b.data()), length);

  uint64_t sent = total_sent(ios);

  auto start = clock_start();
  if (party == ALICE) {
    ot->send_ot_rcm_cc(b0.data(), length);
    delta = ot->ferret->Delta;
  } else {
    ot->recv_ot_rcm_cc(r.data(), reinterpret_cast<bool *>(b.data()), length);
  }
  ios[0]->flush();
  long long t = time_from(start);

  std::cout << (party == ALICE ? "[ALICE]\t" : "[BOB]\t")
            << "Silent OT (rcm, cc)\t" << t << " us"
            << ", " << (total_sent(ios) - sent) << " bytes" << endl;

  if (party == ALICE) {
    ios[0]->send_block(&delta, 1);
    ios[0]->send_block(b0.data(), length);
  } else if (party == BOB) {
    ios[0]->recv_block(&delta, 1);
    ios[0]->recv_block(b0.data(), length);
    for (int64_t i = 0; i < length; ++i) {
      block b1 = b0[i] ^ delta;
      if (b[i] != 0) {
        if (!cmpBlock(&r[i], &b1, 1)) {
          error("COT failed!");
        }
      } else {
        if (!cmpBlock(&r[i], &b0[i], 1)) {
          error("COT failed!");
        }
      }
    }
  }
  std::cout << (party == ALICE ? "[ALICE]\t" : "[BOB]\t") << "Tests passed.\t"
            << std::endl;
  ios[0]->flush();
  return t;
}

double test_ot_rm_rc(SilentOT *ot, IO *ios[threads], int party,
                     int64_t length) {
  std::vector<block> b0(length);
  std::vector<block> r(length);
  std::vector<block> b1(length);
  std::vector<uint8_t> b(length);
  PRG prg;

  uint64_t sent = total_sent(ios);

  auto start = clock_start();
  if (party == ALICE) {
    ot->send_ot_rm_rc(b0.data(), b1.data(), length);
  } else {
    ot->recv_ot_rm_rc(r.data(), reinterpret_cast<bool *>(b.data()), length);
  }
  ios[0]->flush();
  long long t = time_from(start);

  std::cout << (party == ALICE ? "[ALICE]\t" : "[BOB]\t")
            << (party == ALICE ? "[ALICE]" : "[BOB]")
            << "\tSilent OT (rm, rc)\t" << t << " us"
            << ", " << (total_sent(ios) - sent) << " bytes" << endl;

  if (party == ALICE) {
    ios[0]->send_block(b0.data(), length);
    ios[0]->send_block(b1.data(), length);
  } else if (party == BOB) {
    ios[0]->recv_block(b0.data(), length);
    ios[0]->recv_block(b1.data(), length);
    for (int64_t i = 0; i < length; ++i) {
      if (b[i] != 0) {
        assert(cmpBlock(&r[i], &b1[i], 1));
      } else {
        assert(cmpBlock(&r[i], &b0[i], 1));
      }
    }
  }
  std::cout << (party == ALICE ? "[ALICE]\t" : "[BOB]\t") << "Tests passed.\t"
            << std::endl;
  ios[0]->flush();
  return t;
}

template <int l>
double test_ot_cm_cc(SilentOT *ot, IO *ios[threads], int party,
                     int64_t length) {
  using T = typename std::conditional<(l <= 8), uint8_t, uint64_t>::type;
  std::vector<T> msgs_data(length * 2);
  std::vector<T *> msgs(length);
  std::vector<T> r(length);

  PRG prg(fix_key);
  for (int i = 0; i < length; i++) {
    msgs[i] = &msgs_data[i * 2];
    prg.random_data(msgs[i], 2 * sizeof(T));
  }

  std::vector<uint8_t> b(length);
  PRG prg2;
  prg2.random_bool(reinterpret_cast<bool *>(b.data()), length);

  uint64_t sent = total_sent(ios);

  auto start = clock_start();
  if (party == ALICE) {
    ot->send_ot_cm_cc<T>(msgs.data(), length, l);
  } else {
    ot->recv_ot_cm_cc<T>(r.data(), b.data(), length, l);
  }
  ios[0]->flush();
  long long t = time_from(start);

  std::cout << (party == ALICE ? "[ALICE]\t" : "[BOB]\t")
            << "Silent OT (cm, cc)\t"
            << "l = " << l << ":\t" << t << " us"
            << ", " << (total_sent(ios) - sent) << " bytes" << endl;

  if (party == BOB) {
    T mask = (T)((1ULL << l) - 1ULL);
    for (int64_t i = 0; i < length; ++i) {
      if (b[i]) {
        if (((r[i] ^ msgs[i][1]) & mask) != 0) {
          std::cout << i << ", " << std::hex << (uint64_t)r[i] << ", "
                    << std::hex << (uint64_t)msgs[i][1] << "\n";
          error("wrong!\n");
        }
      } else {
        if (((r[i] ^ msgs[i][0]) & mask) != 0) {
          std::cout << i << ", " << std::hex << (uint64_t)r[i] << ", "
                    << std::hex << (uint64_t)msgs[i][0] << "\n";
          error("wrong!\n");
        }
      }
    }
  }
  std::cout << (party == ALICE ? "[ALICE]\t" : "[BOB]\t") << "Tests passed.\t"
            << std::endl;
  return t;
}

template <int N, int l>
double test_ot_cm_cc(SilentOT *ot, IO *ios[threads], int party,
                     int64_t length) {
  using T = typename std::conditional<(l <= 8), uint8_t, uint64_t>::type;
  std::vector<T> msgs_data(length * N);
  std::vector<T *> msgs(length);
  std::vector<T> r(length);

  PRG prg(fix_key);
  for (int i = 0; i < length; i++) {
    msgs[i] = &msgs_data[i * N];
    prg.random_data(msgs[i], N * sizeof(T));
  }

  std::vector<uint8_t> b(length);
  PRG prg2;
  prg2.random_data(b.data(), length);
  for (int i = 0; i < length; i++) b[i] = b[i] % N;

  uint64_t sent = total_sent(ios);

  auto start = clock_start();
  if (party == ALICE) {
    ot->send_ot_cm_cc<T>(msgs.data(), length, N, l);
  } else {
    ot->recv_ot_cm_cc<T>(r.data(), b.data(), length, N, l);
  }
  ios[0]->flush();
  long long t = time_from(start);

  std::cout << (party == ALICE ? "[ALICE]\t" : "[BOB]\t")
            << "Silent OT (cm, cc)\t"
            << "N = " << N << ",\tl = " << l << ":\t" << t << " us"
            << ", " << (total_sent(ios) - sent) << " bytes" << endl;

  if (party == BOB) {
    T mask = (T)((1ULL << l) - 1ULL);
    for (int64_t i = 0; i < length; ++i) {
      if (((r[i] ^ msgs[i][b[i]]) & mask) != 0) {
        std::cout << i << ", " << std::hex << (uint64_t)r[i] << ", " << std::hex
                  << (uint64_t)msgs[i][b[i]] << "\n";
        error("wrong!\n");
      }
    }
  }
  std::cout << (party == ALICE ? "[ALICE]\t" : "[BOB]\t") << "Tests passed.\t"
            << std::endl;
  return t;
}

template <int l>
double test_ot_rm_rc(SilentOT *ot, IO *ios[threads], int party,
                     int64_t length) {
  using T = typename std::conditional<(l <= 8), uint8_t, uint64_t>::type;
  std::vector<T> msg0(length);
  std::vector<T> msg1(length);
  std::vector<T> r(length);

  std::vector<uint8_t> b(length);

  uint64_t sent = total_sent(ios);

  auto start = clock_start();
  if (party == ALICE) {
    ot->send_ot_rm_rc<T>(msg0.data(), msg1.data(), length, l);
  } else {
    ot->recv_ot_rm_rc<T>(r.data(), reinterpret_cast<bool *>(b.data()), length,
                         l);
  }
  ios[0]->flush();
  long long t = time_from(start);

  std::cout << (party == ALICE ? "[ALICE]\t" : "[BOB]\t")
            << "Silent OT (rm, rc)\t"
            << "l = " << l << ":\t" << t << " us"
            << ", " << (total_sent(ios) - sent) << " bytes" << endl;

  if (party == ALICE) {
    ios[0]->send_data(msg0.data(), length * sizeof(T));
    ios[0]->send_data(msg1.data(), length * sizeof(T));
  } else if (party == BOB) {
    ios[0]->recv_data(msg0.data(), length * sizeof(T));
    ios[0]->recv_data(msg1.data(), length * sizeof(T));
    for (int64_t i = 0; i < length; ++i) {
      // std::cout << "msg0: " << (int)msg0[i] << ", msg1: " << (int)msg1[i] <<
      // ", b: " << (int)b[i] << ", r: " << (int)r[i] << std::endl;
      if (b[i] != 0) {
        assert(r[i] == msg1[i]);
      } else {
        assert(r[i] == msg0[i]);
      }
    }
  }

  std::cout << (party == ALICE ? "[ALICE]\t" : "[BOB]\t") << "Tests passed.\t"
            << std::endl;
  return t;
}

void test_silent(int party, IO *ios[threads], int64_t num_ot) {
  uint64_t sent = ios[0]->counter;
  auto start = clock_start();
  auto silent_ot = std::make_unique<SilentOT>(
      party, threads, ios, false, true,
      (party == ALICE ? "tmp_silent_ot_pre_alice" : "tmp_silent_ot_pre_bob"));
  double timeused = time_from(start);
  std::cout << party << "\tsetup\t" << timeused / 1000 << "ms"
            << ", " << (ios[0]->counter - sent) << " bytes" << std::endl;

  int64_t num = 1LL << num_ot;
  double t = 0;

  // This line is used to warm up ferret OT because it is generated batch by
  // batch
  t = test_ot_cm_cc(silent_ot.get(), ios, party, 1);

  t = test_ot_cm_cc(silent_ot.get(), ios, party, num);
  std::cout << (party == ALICE ? "[ALICE]\t" : "[BOB]\t")
            << "Silent OT (cm, cc):\t" << double(num) / t * 1e6 << " OTps"
            << std::endl;

  t = test_ot_rm_cc(silent_ot.get(), ios, party, num);
  std::cout << (party == ALICE ? "[ALICE]\t" : "[BOB]\t")
            << "Silent OT (rm, cc):\t" << double(num) / t * 1e6 << " OTps"
            << std::endl;

  t = test_ot_rcm_cc(silent_ot.get(), ios, party, num);
  std::cout << (party == ALICE ? "[ALICE]\t" : "[BOB]\t")
            << "Silent OT (rcm, cc):\t" << double(num) / t * 1e6 << " OTps"
            << std::endl;

  t = test_ot_rm_rc(silent_ot.get(), ios, party, num);
  std::cout << (party == ALICE ? "[ALICE]\t" : "[BOB]\t")
            << "Silent OT (rm, rc):\t" << double(num) / t * 1e6 << " OTps"
            << std::endl;

  t = test_ot_cm_cc<2>(silent_ot.get(), ios, party, num);
  std::cout << (party == ALICE ? "[ALICE]\t" : "[BOB]\t")
            << "Silent OT (cm, cc, l=2):\t" << double(num) / t * 1e6 << " OTps"
            << std::endl;

  t = test_ot_cm_cc<16, 2>(silent_ot.get(), ios, party, num);
  std::cout << (party == ALICE ? "[ALICE]\t" : "[BOB]\t")
            << "Silent OT (cm, cc, N=16, l=2):\t" << double(num) / t * 1e6
            << " OTps" << std::endl;

  t = test_ot_cm_cc<16, 1>(silent_ot.get(), ios, party, num);
  std::cout << (party == ALICE ? "[ALICE]\t" : "[BOB]\t")
            << "Silent OT (cm, cc, N=16, l=1):\t" << double(num) / t * 1e6
            << " OTps" << std::endl;

  t = test_ot_rm_rc<1>(silent_ot.get(), ios, party, num);
  std::cout << (party == ALICE ? "[ALICE]\t" : "[BOB]\t")
            << "Silent OT (rm, rc, l=1):\t" << double(num) / t * 1e6 << " OTps"
            << std::endl;

  char cwd[PATH_MAX];
  if (getcwd(cwd, sizeof(cwd)) != nullptr) {
    printf("Current working dir: %s\n", cwd);
  } else {
    perror("getcwd() error");
  }
}

TEST(SilentOTTest, Test) {
  const int kWorldSize = 2;
  auto contexts = yacl::link::test::SetupWorld(kWorldSize);

  int length = 20;
  std::future<void> alice = std::async([&] {
    CheetahIo *ios[threads];
    for (auto &io : ios) {
      io = new CheetahIo(contexts[0]);
    }
    test_silent(emp::ALICE, ios, length);
    for (auto &io : ios) {
      delete io;
    }
  });

  std::future<void> bob = std::async([&] {
    CheetahIo *ios[threads];
    for (auto &io : ios) {
      io = new CheetahIo(contexts[1]);
    }
    test_silent(emp::BOB, ios, length);
    for (auto &io : ios) {
      delete io;
    }
  });

  alice.get();
  bob.get();
}

}  // namespace spu
