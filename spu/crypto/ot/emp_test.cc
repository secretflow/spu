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

#include <iostream>
#include <thread>

#include "emp-ot/emp-ot.h"
#include "emp-tool/emp-tool.h"
#include "gtest/gtest.h"

namespace spu {
template <typename T>
double test_ot(T *ot, NetIO *io, int party, int64_t length) {
  block *b0 = new block[length], *b1 = new block[length],
        *r = new block[length];
  PRG prg(fix_key);
  prg.random_block(b0, length);
  prg.random_block(b1, length);
  bool *b = new bool[length];
  PRG prg2;
  prg2.random_bool(b, length);

  auto start = clock_start();
  if (party == ALICE) {
    ot->send(b0, b1, length);
  } else {
    ot->recv(r, b, length);
  }
  io->flush();
  long long t = time_from(start);
  if (party == BOB) {
    for (int64_t i = 0; i < length; ++i) {
      if (b[i]) {
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
  std::cout << "Tests passed.\t";
  delete[] b0;
  delete[] b1;
  delete[] r;
  delete[] b;
  return t;
}

template <typename T>
double test_cot(T *ot, NetIO *io, int party, int64_t length) {
  block *b0 = new block[length], *r = new block[length];
  bool *b = new bool[length];
  block delta;
  PRG prg;
  prg.random_block(&delta, 1);
  prg.random_bool(b, length);

  io->sync();
  auto start = clock_start();
  if (party == ALICE) {
    ot->send_cot(b0, length);
    delta = ot->Delta;
  } else {
    ot->recv_cot(r, b, length);
  }
  io->flush();
  long long t = time_from(start);
  if (party == ALICE) {
    io->send_block(&delta, 1);
    io->send_block(b0, length);
  } else if (party == BOB) {
    io->recv_block(&delta, 1);
    io->recv_block(b0, length);
    for (int64_t i = 0; i < length; ++i) {
      block b1 = b0[i] ^ delta;
      if (b[i]) {
        if (!cmpBlock(&r[i], &b1, 1)) error("COT failed!");
      } else {
        if (!cmpBlock(&r[i], &b0[i], 1)) error("COT failed!");
      }
    }
  }
  std::cout << "Tests passed.\t";
  io->flush();
  delete[] b0;
  delete[] r;
  delete[] b;
  return t;
}

template <typename T>
double test_rot(T *ot, NetIO *io, int party, int64_t length) {
  block *b0 = new block[length], *r = new block[length];
  block *b1 = new block[length];
  bool *b = new bool[length];
  PRG prg;
  prg.random_bool(b, length);

  io->sync();
  auto start = clock_start();
  if (party == ALICE) {
    ot->send_rot(b0, b1, length);
  } else {
    ot->recv_rot(r, b, length);
  }
  io->flush();
  long long t = time_from(start);
  if (party == ALICE) {
    io->send_block(b0, length);
    io->send_block(b1, length);
  } else if (party == BOB) {
    io->recv_block(b0, length);
    io->recv_block(b1, length);
    for (int64_t i = 0; i < length; ++i) {
      if (b[i])
        assert(cmpBlock(&r[i], &b1[i], 1));
      else
        assert(cmpBlock(&r[i], &b0[i], 1));
    }
  }
  std::cout << "Tests passed.\t";
  io->flush();
  delete[] b0;
  delete[] b1;
  delete[] r;
  delete[] b;
  return t;
}

template <typename T>
double test_rcot(T *ot, NetIO *io, int party, int64_t length, bool inplace) {
  block *b = nullptr;
  PRG prg;

  io->sync();
  auto start = clock_start();
  int64_t mem_size;
  if (!inplace) {
    mem_size = length;
    b = new block[length];

    // The RCOTs will be generated in the internal buffer
    // then be copied to the user buffer
    ot->rcot(b, length);
  } else {
    // Call byte_memory_need_inplace() to get the buffer size needed
    mem_size = ot->byte_memory_need_inplace((uint64_t)length);
    b = new block[mem_size];

    // The RCOTs will be generated directly to this buffer
    ot->rcot_inplace(b, mem_size);
  }
  long long t = time_from(start);
  io->sync();
  if (party == ALICE) {
    io->send_block(&ot->Delta, 1);
    io->send_block(b, mem_size);
  } else if (party == BOB) {
    block ch[2];
    ch[0] = zero_block;
    block *b0 = new block[mem_size];
    io->recv_block(ch + 1, 1);
    io->recv_block(b0, mem_size);
    for (int64_t i = 0; i < mem_size; ++i) {
      b[i] = b[i] ^ ch[getLSB(b[i])];
    }
    if (!cmpBlock(b, b0, mem_size)) error("RCOT failed");
    delete[] b0;
  }
  std::cout << "Tests passed.\t";
  delete[] b;
  return t;
}

const int threads = 1;

void test_ferret(int party, NetIO *ios[threads], int64_t num_ot) {
  auto start = clock_start();
  std::cout << party << ": before ferret constructor" << std::endl;
  FerretCOT<NetIO> *ferretcot =
      new FerretCOT<NetIO>(party, threads, ios, false);
  std::cout << party << ": after ferret constructor" << std::endl;
  double timeused = time_from(start);
  std::cout << party << "\tsetup\t" << timeused / 1000 << "ms" << std::endl;

  // RCOT
  // The RCOTs will be generated at internal memory, and copied to user buffer
  int64_t num = 1 << num_ot;
  std::cout << "Active FERRET RCOT\t"
            << double(num) /
                   test_rcot<FerretCOT<NetIO>>(ferretcot, ios[0], party, num,
                                               false) *
                   1e6
            << " OTps" << std::endl;

  // RCOT inplace
  // The RCOTs will be generated at user buffer
  // Get the buffer size needed by calling byte_memory_need_inplace()
  uint64_t batch_size = ferretcot->ot_limit;
  std::cout << "Active FERRET RCOT inplace\t"
            << double(batch_size) /
                   test_rcot<FerretCOT<NetIO>>(ferretcot, ios[0], party,
                                               batch_size, true) *
                   1e6
            << " OTps" << std::endl;
  delete ferretcot;
}

void start(int party, int64_t num_ot) {
  int port = 12345;
  NetIO *ios[threads];
  for (int i = 0; i < threads; ++i)
    ios[i] = new NetIO(party == ALICE ? nullptr : "127.0.0.1", port + i);

  std::cout << party << ": after NetIO" << std::endl;

  test_ferret(party, ios, num_ot);

  std::cout << party << ": after ferret" << std::endl;

  for (int i = 0; i < threads; ++i) delete ios[i];
}

TEST(EmpTest, Test) {
  std::cout << "Enter EmpTest" << std::endl;
  std::thread alice(start, ALICE, 10);
  std::thread bob(start, BOB, 10);

  alice.join();
  bob.join();

  std::cout << "Done" << std::endl;
}
}  // namespace spu
