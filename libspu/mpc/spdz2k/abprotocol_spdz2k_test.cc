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

#include "libspu/mpc/ab_api.h"
#include "libspu/mpc/ab_api_test.h"
#include "libspu/mpc/api.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/kernel.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::mpc::test {
namespace {

Shape kShape = {30, 40};
const std::vector<size_t> kShiftBits = {0, 1, 2, 31, 32, 33, 64, 1000};

#define EXPECT_VALUE_EQ(X, Y)                            \
  {                                                      \
    EXPECT_EQ((X).dtype(), (Y).dtype());                 \
    EXPECT_EQ((X).shape(), (Y).shape());                 \
    EXPECT_TRUE(ring_all_equal((X).data(), (Y).data())); \
  }

#define EXPECT_VALUE_ALMOST_EQ(X, Y, ERR)                \
  {                                                      \
    EXPECT_EQ((X).dtype(), (Y).dtype());                 \
    EXPECT_EQ((X).shape(), (Y).shape());                 \
    EXPECT_TRUE(ring_all_equal((X).data(), (Y).data())); \
  }

bool verifyCost(Kernel* kernel, std::string_view name, FieldType field,
                const Shape& shape, size_t npc,
                const Communicator::Stats& cost) {
  if (kernel->kind() == Kernel::Kind::Dynamic) {
    return true;
  }

  auto comm = kernel->comm();
  auto latency = kernel->latency();

  size_t numel = shape.numel();

  bool succeed = true;
  constexpr size_t kBitsPerBytes = 8;
  ce::Params params = {{"K", SizeOf(field) * 8}, {"N", npc}};
  if (comm->eval(params) * numel != cost.comm * kBitsPerBytes) {
    fmt::print("Failed: {} comm mismatch, expected={}, got={}\n", name,
               comm->eval(params) * numel, cost.comm * kBitsPerBytes);
    succeed = false;
  }
  if (latency->eval(params) != cost.latency) {
    fmt::print("Failed: {} latency mismatch, expected={}, got={}\n", name,
               latency->eval(params), cost.latency);
    succeed = false;
  }

  return succeed;
}

}  // namespace

TEST_P(BooleanTest, NotB) {
  const auto factory = std::get<0>(GetParam());
  const RuntimeConfig& conf = std::get<1>(GetParam());
  const size_t npc = std::get<2>(GetParam());

  utils::simulate(npc, [&](std::shared_ptr<yacl::link::Context> lctx) {
    auto obj = factory(conf, lctx);

    /* GIVEN */
    auto p0 = rand_p(obj.get(), kShape);
    auto b0 = p2b(obj.get(), p0);

    /* WHEN */
    auto prev = obj->prot()->getState<Communicator>()->getStats();
    auto r_b = dynDispatch(obj.get(), "not_b", b0);
    auto cost = obj->prot()->getState<Communicator>()->getStats() - prev;
    auto r_p = b2p(obj.get(), r_b);
    auto r_pp = not_p(obj.get(), p0);

    /* THEN */
    EXPECT_VALUE_EQ(r_p, r_pp);
    EXPECT_TRUE(verifyCost(obj->prot()->getKernel("not_b"), "not_b",
                           conf.field(), kShape, npc, cost));
  });
}

TEST_P(ConversionTest, AddBB) {
  const auto factory = std::get<0>(GetParam());
  const RuntimeConfig& conf = std::get<1>(GetParam());
  const size_t npc = std::get<2>(GetParam());

  utils::simulate(npc, [&](std::shared_ptr<yacl::link::Context> lctx) {
    auto obj = factory(conf, lctx);

    if (!obj->hasKernel("and_bb")) {
      return;
    }
    /* GIVEN */
    auto p0 = rand_p(obj.get(), kShape);
    auto p1 = rand_p(obj.get(), kShape);

    /* WHEN */
    auto b0 = p2b(obj.get(), p0);
    auto b1 = p2b(obj.get(), p1);
    auto prev = obj->prot()->getState<Communicator>()->getStats();
    auto tmp = add_bb(obj.get(), b0, b1);
    auto cost = obj->prot()->getState<Communicator>()->getStats() - prev;
    auto re = b2p(obj.get(), tmp);
    auto rp = add_pp(obj.get(), p0, p1);

    /* THEN */
    EXPECT_VALUE_EQ(re, rp);
    EXPECT_TRUE(verifyCost(obj->getKernel("add_bb"), "add_bb", conf.field(),
                           kShape, npc, cost));
  });
}

TEST_P(ConversionTest, AddBP) {
  const auto factory = std::get<0>(GetParam());
  const RuntimeConfig& conf = std::get<1>(GetParam());
  const size_t npc = std::get<2>(GetParam());

  utils::simulate(npc, [&](std::shared_ptr<yacl::link::Context> lctx) {
    auto obj = factory(conf, lctx);

    if (!obj->hasKernel("and_bp")) {
      return;
    }
    /* GIVEN */
    auto p0 = rand_p(obj.get(), kShape);
    auto p1 = rand_p(obj.get(), kShape);

    /* WHEN */
    auto b0 = p2b(obj.get(), p0);
    auto prev = obj->prot()->getState<Communicator>()->getStats();
    // Not a common test!!!!
    auto tmp = dynDispatch(obj.get(), "add_bp", b0, p1);
    auto cost = obj->prot()->getState<Communicator>()->getStats() - prev;
    auto re = b2p(obj.get(), tmp);
    auto rp = add_pp(obj.get(), p0, p1);

    /* THEN */
    EXPECT_VALUE_EQ(re, rp);
    EXPECT_TRUE(verifyCost(obj->getKernel("add_bp"), "add_bp", conf.field(),
                           kShape, npc, cost));
  });
}

TEST_P(ConversionTest, Bit2A) {
  const auto factory = std::get<0>(GetParam());
  const RuntimeConfig& conf = std::get<1>(GetParam());
  const size_t npc = std::get<2>(GetParam());

  utils::simulate(npc, [&](std::shared_ptr<yacl::link::Context> lctx) {
    auto obj = factory(conf, lctx);

    if (!obj->hasKernel("bit2a")) {
      return;
    }
    /* GIVEN */
    auto p0 = rand_p(obj.get(), kShape);
    p0 = msb_p(obj.get(), p0);

    /* WHEN */
    auto b = p2b(obj.get(), p0);
    auto prev = obj->prot()->getState<Communicator>()->getStats();
    auto a = dynDispatch(obj.get(), "bit2a", b);
    auto cost = obj->prot()->getState<Communicator>()->getStats() - prev;

    auto p1 = a2p(obj.get(), a);
    /* THEN */
    EXPECT_VALUE_EQ(p0, p1);
    EXPECT_TRUE(verifyCost(obj->getKernel("bit2a"), "bit2a", conf.field(),
                           kShape, npc, cost));
  });
}

TEST_P(ConversionTest, A2Bit) {
  const auto factory = std::get<0>(GetParam());
  const RuntimeConfig& conf = std::get<1>(GetParam());
  const size_t npc = std::get<2>(GetParam());

  utils::simulate(npc, [&](std::shared_ptr<yacl::link::Context> lctx) {
    auto obj = factory(conf, lctx);

    if (!obj->hasKernel("a2bit")) {
      return;
    }
    /* GIVEN */
    auto p0 = rand_p(obj.get(), kShape);
    p0 = msb_p(obj.get(), p0);

    /* WHEN */
    auto a = p2a(obj.get(), p0);
    auto prev = obj->prot()->getState<Communicator>()->getStats();
    auto b = dynDispatch(obj.get(), "a2bit", a);
    auto cost = obj->prot()->getState<Communicator>()->getStats() - prev;
    // reserve the least significant bit only
    auto p1 = b2p(obj.get(), b);
    /* THEN */
    EXPECT_VALUE_EQ(p0, p1);
    EXPECT_TRUE(verifyCost(obj->getKernel("a2bit"), "a2bit", conf.field(),
                           kShape, npc, cost));
  });
}

TEST_P(ConversionTest, BitLT) {
  const auto factory = std::get<0>(GetParam());
  const RuntimeConfig& conf = std::get<1>(GetParam());
  const size_t npc = std::get<2>(GetParam());

  utils::simulate(npc, [&](std::shared_ptr<yacl::link::Context> lctx) {
    auto obj = factory(conf, lctx);

    if (!obj->hasKernel("bitlt_bb")) {
      return;
    }
    /* GIVEN */
    auto p0 = rand_p(obj.get(), kShape);
    auto p1 = rand_p(obj.get(), kShape);

    /* WHEN */
    auto b0 = p2b(obj.get(), p0);
    auto b1 = p2b(obj.get(), p1);
    auto prev = obj->prot()->getState<Communicator>()->getStats();
    // Not a common test!!!!
    auto tmp = dynDispatch(obj.get(), "bitlt_bb", b0, b1);
    auto cost = obj->prot()->getState<Communicator>()->getStats() - prev;
    auto re = b2p(obj.get(), tmp);

    const auto field = p0.storage_type().as<Ring2k>()->field();
    DISPATCH_ALL_FIELDS(field, "_", [&]() {
      using U = std::make_unsigned<ring2k_t>::type;
      size_t numel = kShape.numel();

      auto p0_data = p0.data().data<U>();
      auto p1_data = p1.data().data<U>();
      auto re_data = re.data().data<U>();
      for (size_t i = 0; i < numel; ++i) {
        if ((p0_data[i] < p1_data[i])) {
          SPU_ENFORCE((re_data[i] == 1), "i {}, p0 {}, p1 {}", i, p0_data[i],
                      p1_data[i]);
        } else {
          SPU_ENFORCE((re_data[i] == 0), "i {}, p0 {}, p1 {}", i, p0_data[i],
                      p1_data[i]);
        }
      }
    });

    /* THEN */
    EXPECT_TRUE(verifyCost(obj->getKernel("bitlt_bb"), "bitlt_bb", conf.field(),
                           kShape, npc, cost));
  });
}

TEST_P(ConversionTest, BitLE) {
  const auto factory = std::get<0>(GetParam());
  const RuntimeConfig& conf = std::get<1>(GetParam());
  const size_t npc = std::get<2>(GetParam());

  utils::simulate(npc, [&](std::shared_ptr<yacl::link::Context> lctx) {
    auto obj = factory(conf, lctx);

    if (!obj->hasKernel("bitle_bb")) {
      return;
    }
    /* GIVEN */
    auto p0 = rand_p(obj.get(), kShape);
    auto p1 = rand_p(obj.get(), kShape);

    /* WHEN */
    auto b0 = p2b(obj.get(), p0);
    auto b1 = p2b(obj.get(), p1);
    auto prev = obj->prot()->getState<Communicator>()->getStats();
    // Not a common test!!!!
    auto tmp = dynDispatch(obj.get(), "bitle_bb", b0, b1);
    auto cost = obj->prot()->getState<Communicator>()->getStats() - prev;
    auto re = b2p(obj.get(), tmp);

    const auto field = p0.storage_type().as<Ring2k>()->field();
    DISPATCH_ALL_FIELDS(field, "_", [&]() {
      using U = std::make_unsigned<ring2k_t>::type;
      size_t numel = kShape.numel();
      auto p0_data = p0.data().data<U>();
      auto p1_data = p1.data().data<U>();
      auto re_data = re.data().data<U>();

      for (size_t i = 0; i < numel; ++i) {
        if ((p0_data[i] <= p1_data[i])) {
          SPU_ENFORCE((re_data[i] == 1), "i {}, p0 {}, p1 {}", i, p0_data[i],
                      p1_data[i]);
        } else {
          SPU_ENFORCE((re_data[i] == 0), "i {}, p0 {}, p1 {}", i, p0_data[i],
                      p1_data[i]);
        }
      }
    });

    /* THEN */
    EXPECT_TRUE(verifyCost(obj->getKernel("bitle_bb"), "bitle_bb", conf.field(),
                           kShape, npc, cost));
  });
}

TEST_P(BooleanTest, BitIntl) {
  const auto factory = std::get<0>(GetParam());
  const RuntimeConfig& conf = std::get<1>(GetParam());
  const size_t npc = std::get<2>(GetParam());
  size_t stride = 0;

  utils::simulate(npc, [&](std::shared_ptr<yacl::link::Context> lctx) {
    auto obj = factory(conf, lctx);
    /* GIVEN */
    auto p0 = rand_p(obj.get(), kShape);

    /* WHEN */
    auto b = p2b(obj.get(), p0);
    auto prev = obj->prot()->getState<Communicator>()->getStats();
    auto tmp = bitintl_b(obj.get(), b, stride);
    auto cost = obj->prot()->getState<Communicator>()->getStats() - prev;

    auto p1 = b2p(obj.get(), tmp);
    auto pp1 = bitintl_b(obj.get(), p0, stride);
    /* THEN */
    EXPECT_VALUE_EQ(p1, pp1);
    EXPECT_TRUE(verifyCost(obj->getKernel("bitintl_b"), "bitintl_b",
                           conf.field(), kShape, npc, cost));
  });
}

TEST_P(BooleanTest, BitDeintl) {
  const auto factory = std::get<0>(GetParam());
  const RuntimeConfig& conf = std::get<1>(GetParam());
  const size_t npc = std::get<2>(GetParam());
  size_t stride = 0;

  utils::simulate(npc, [&](std::shared_ptr<yacl::link::Context> lctx) {
    auto obj = factory(conf, lctx);
    /* GIVEN */
    auto p0 = rand_p(obj.get(), kShape);

    /* WHEN */
    auto b = p2b(obj.get(), p0);
    auto prev = obj->prot()->getState<Communicator>()->getStats();
    auto tmp = bitdeintl_b(obj.get(), b, stride);
    auto cost = obj->prot()->getState<Communicator>()->getStats() - prev;

    auto p1 = b2p(obj.get(), tmp);
    auto pp1 = bitdeintl_b(obj.get(), p0, stride);
    /* THEN */
    EXPECT_VALUE_EQ(p1, pp1);
    EXPECT_TRUE(verifyCost(obj->getKernel("bitintl_b"), "bitintl_b",
                           conf.field(), kShape, npc, cost));
  });
}

TEST_P(BooleanTest, BitIntlAndDeintl) {
  const auto factory = std::get<0>(GetParam());
  const RuntimeConfig& conf = std::get<1>(GetParam());
  const size_t npc = std::get<2>(GetParam());
  size_t stride = 0;

  utils::simulate(npc, [&](std::shared_ptr<yacl::link::Context> lctx) {
    auto obj = factory(conf, lctx);
    /* GIVEN */
    auto p0 = rand_p(obj.get(), kShape);

    /* WHEN */
    auto b = p2b(obj.get(), p0);
    auto prev = obj->prot()->getState<Communicator>()->getStats();
    auto b0 = bitintl_b(obj.get(), b, stride);
    auto cost = obj->prot()->getState<Communicator>()->getStats() - prev;

    auto b1 = bitdeintl_b(obj.get(), b0, stride);
    auto p1 = b2p(obj.get(), b1);
    /* THEN */
    EXPECT_VALUE_EQ(p0, p1);
    EXPECT_TRUE(verifyCost(obj->getKernel("bitintl_b"), "bitintl_b",
                           conf.field(), kShape, npc, cost));
  });
}

}  // namespace spu::mpc::test
