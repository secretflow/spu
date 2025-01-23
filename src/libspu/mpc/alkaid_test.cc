// This test file is only used to test accuracy. For benchmark, please use
// examples/alkaid/benchmark/microbenchmark.cc.

// #define ALKAID_DO_NOT_USE_OFFLINE

#include <chrono>
#include <iostream>
#include <string>

#include "googletest"

#include "libspu/core/ndarray_ref.h"
#include "libspu/mpc/ab_api.h"
#include "libspu/mpc/api.h"
#include "libspu/mpc/offline_recorder.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/utils/simulate.h"
#include "libspu/spu.pb.h"

// hack wrap value for handle namespace error.
#define Ref2Value(x) spu::Value(x, spu::DT_INVALID)
#define Value2Ref(x) x.data()
#define OpenRef(ndref, tag) b2p(sctx_##tag.get(), Ref2Value(ndref))
#define GetComm(tag) sctx_##tag.get()->getState<Communicator>()->getStats()

// #define ALKAID_USE_BITWIDTH_16
#define N 100ull
#define M 100ull
#define RANDOM_INPUT 1

// protocols.
#include "libspu/mpc/aby3/conversion.h"
#include "libspu/mpc/aby3/protocol.h"
#include "libspu/mpc/aby3/type.h"
#include "libspu/mpc/aby3/value.h"
namespace aby3np = spu::mpc::aby3;  // aby3 namespace

#include "libspu/mpc/alkaid/conversion.h"
#include "libspu/mpc/alkaid/mss_utils.h"
#include "libspu/mpc/alkaid/protocol.h"
#include "libspu/mpc/alkaid/type.h"
#include "libspu/mpc/alkaid/value.h"
namespace oursnp = spu::mpc::alkaid;  // ours namespace

#ifdef ALKAID_USE_BITWIDTH_16
using ring2k_t = uint16_t;
#else
using ring2k_t = uint64_t;
#endif
using pub_t = std::array<ring2k_t, 1>;
using namespace spu::mpc;

spu::RuntimeConfig makeConfig(spu::FieldType field, bool use_al = true) {
  spu::RuntimeConfig conf;
  if (use_al) conf.set_protocol(spu::ProtocolKind::ALKAID);
  else
    conf.set_protocol(spu::ProtocolKind::ABY3);
  conf.set_field(field);
  return conf;
}

void printResult(spu::NdArrayRef& result, const std::string& name, size_t num_outputs=10) {
  spu::NdArrayView<pub_t> _r(result);
  std::cout << name << ": ";
  for (int i = 0; i < num_outputs; i++) {
    std::cout << _r[i][0] << " ";
  }
  std::cout << std::endl;
}

void checkOutput(spu::NdArrayRef& spu_result, spu::NdArrayRef& ours_result,
                 const std::string& name) {
  spu::NdArrayView<pub_t> _spu(spu_result);
  spu::NdArrayView<pub_t> _ours(ours_result);

  size_t match_count = 0;

  for (size_t i = 0; i < N * M; i++) {
    if (_spu[i][0] == _ours[i][0]) match_count++;
    // else std::cout << "mismatch at " << i << ": " << _spu[i][0] << " " <<
    // _ours[i][0] << std::endl;
  }

  std::cout << name << " match count rate: "
            << static_cast<double>(match_count) / (N * M) << std::endl;
}

#define EXPECT_VALUE_EQ(X, Y)                            \
  {                                                      \
    EXPECT_EQ((X).shape(), (Y).shape());                 \
    EXPECT_TRUE(ring_all_equal((X).data(), (Y).data())); \
  }

#define EXPECT_VALUE_ALMOST_EQ(X, Y, ERR)                     \
  {                                                           \
    EXPECT_EQ((X).shape(), (Y).shape());                      \
    EXPECT_TRUE(ring_all_equal((X).data(), (Y).data(), ERR)); \
  }

auto field = spu::FieldType::FM64;
spu::RuntimeConfig config_ours = makeConfig(field, true);
spu::RuntimeConfig config_aby3 = makeConfig(field, false);

TEST(AlkaidTest, Resharing) {
  spu::Shape kShape = {N, M};

  utils::simulate(3, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto sctx_ours = makeAlkaidProtocol(config_ours, lctx);
    auto kectx_ours = spu::KernelEvalContext(sctx_ours.get());
    auto sctx_aby3 = makeAby3Protocol(config_aby3, lctx);
    auto kectx_aby3 = spu::KernelEvalContext(sctx_aby3.get());
    if (lctx.get()->Rank() == 0) OfflineRecorder::StopRecorder();
    
  });
}
int main(){
  spu::Shape kShape = {N, M};

  utils::simulate(3, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
/**
 * ----------------------------------------------
 *                Input random x, y.
 * ----------------------------------------------
 */
#ifdef RANDOM_INPUT
    auto x_p_aby3 = rand_p(sctx_aby3.get(), kShape);
    // Value2Ref(x_p_ours).eltype() = spu::makeType<oursnp::Pub2kTy>(field);
    auto y_p_aby3 = rand_p(sctx_aby3.get(), kShape);
// Value2Ref(y_p_ours).eltype() = spu::makeType<oursnp::Pub2kTy>(field);
#else
// auto x_p = make_p(sctx_ours.get(),
// static_cast<uint128_t>(16813041318660192066ull), kShape);
#ifdef ALKAID_USE_BITWIDTH_16
            auto x_p_aby3 = make_p(sctx_aby3.get(), static_cast<uint128_t>(1ull << 15), kShape);
            auto y_p_aby3 = make_p(sctx_aby3.get(), static_cast<uint128_t>(1ull << 15), kShape);
#else
            auto x_p_aby3 = make_p(sctx_aby3.get(), static_cast<uint128_t>(1ull << 63), kShape);
            auto y_p_aby3 = make_p(sctx_aby3.get(), static_cast<uint128_t>(1ull << 63), kShape);
#endif
#endif
    auto x_p_ours = x_p_aby3;
    auto y_p_ours = y_p_aby3;
    if (lctx.get()->Rank() == 0)
      std::cout << "Public input initialize." << std::endl;
    auto x_sb_aby3 = p2b(sctx_aby3.get(), x_p_aby3);
    auto y_sb_aby3 = p2b(sctx_aby3.get(), y_p_aby3);
    if (lctx.get()->Rank() == 0)
      std::cout << "ABY3 sharing is converted." << std::endl;
    auto x_sb_ours = p2b(sctx_ours.get(), x_p_ours);
    auto y_sb_ours = p2b(sctx_ours.get(), y_p_ours);
    if (lctx.get()->Rank() == 0)
      std::cout << "Ours sharing is converted." << std::endl;
#ifdef ALKAID_USE_BITWIDTH_16
    Value2Ref(x_sb_aby3).eltype() =
        spu::makeType<aby3np::BShrTy>(aby3np::calcBShareBacktype(16), 16);
    Value2Ref(y_sb_aby3).eltype() =
        spu::makeType<aby3np::BShrTy>(aby3np::calcBShareBacktype(16), 16);
    Value2Ref(x_sb_ours).eltype() =
        spu::makeType<oursnp::BShrTy>(oursnp::calcBShareBacktype(16), 16);
    Value2Ref(y_sb_ours).eltype() =
        spu::makeType<oursnp::BShrTy>(oursnp::calcBShareBacktype(16), 16);
#endif
    if (lctx.get()->Rank() == 0) printResult(x_p_aby3.data(), "Input x: ");
    if (lctx.get()->Rank() == 0) printResult(y_p_aby3.data(), "Input y: ");
    if (lctx.get()->Rank() == 0)
      std::cout << "Number of threads: " << yacl::get_num_threads()
                << std::endl;

    // /**
    //  * ----------------------------------------------
    //  *                Test: Offline.
    //  * ----------------------------------------------
    //  */
    // {
    //   if (lctx.get()->Rank() == 0) std::cout <<
    //   "##############################################\nOffline\n##############################################"
    //   << std::endl;

    //   spu::Shape newkShape = {N, M};
    //   if (lctx.get()->Rank() == 0) std::cout << "Mult size: " << N * M <<
    //   std::endl; size_t comm = GetComm(ours).comm; size_t latency =
    //   GetComm(ours).latency; auto offline_start =
    //   std::chrono::high_resolution_clock::now(); if (lctx.get()->Rank() == 0)
    //   OfflineRecorder::StartRecorder();

    //   auto x_test_mss = make_p(sctx_aby3.get(), static_cast<uint128_t>(1ull
    //   << 7), newkShape);
    //   // auto xb_test_mss = p2b(sctx_ours.get(), x_test_mss);
    //   // auto xb_test_rss = oursnp::ResharingMrss2Rss(&kectx_ours,
    //   Value2Ref(xb_test_mss)); auto xb_test_rss =
    //   Value2Ref(p2b(sctx_ours.get(), x_test_mss));

    //   spu::NdArrayRef x_b_ours;
    //   for (int i = 0; i < EXP_TIMES; i++) {
    //     auto xb_res_ass = oursnp::RssAnd2NoComm(&kectx_ours, xb_test_rss,
    //     xb_test_rss); auto xb_res_rss = oursnp::ResharingAss2Rss(&kectx_ours,
    //     xb_res_ass);
    //   }

    //   auto offline_ours = std::chrono::high_resolution_clock::now();
    //   auto duration_ours =
    //   std::chrono::duration_cast<std::chrono::microseconds>(offline_ours -
    //   offline_start); if (lctx.get()->Rank() == 0) std::cout <<
    //   "------------------------ offline, ours" << std::endl; if
    //   (lctx.get()->Rank() == 0) std::cout << "offline micro seconds: " <<
    //   duration_ours.count() << std::endl; if (lctx.get()->Rank() == 0)
    //   std::cout << "offline comm: " << GetComm(ours).comm - comm <<
    //   std::endl; if (lctx.get()->Rank() == 0) std::cout << "offline latency:
    //   " << GetComm(ours).latency - latency << std::endl; if
    //   (lctx.get()->Rank() == 0) OfflineRecorder::StopRecorder();
    // }

    /**
     * ----------------------------------------------
     *                Test: ExtMsb.
     * ----------------------------------------------
     */
    {
      if (lctx.get()->Rank() == 0)
        std::cout << "##############################################\nExtMsb\n#"
                     "#############################################"
                  << std::endl;
      auto x_ari_aby3 = p2a(sctx_aby3.get(), x_p_aby3);
      auto x_ari_ours = p2a(sctx_ours.get(), x_p_ours);
      if (lctx.get()->Rank() == 0) std::cout << "Prepare input." << std::endl;

      size_t comm = GetComm(aby3).comm;
      size_t latency = GetComm(aby3).latency;
      auto msb_start = std::chrono::high_resolution_clock::now();

      spu::Value msbx_s;
      for (int i = 0; i < EXP_TIMES; i++) {
#ifdef ALKAID_USE_BITWIDTH_16
        auto msbx_s_ref =
            aby3np::MsbA2BForBitwidth16(&kectx_aby3, Value2Ref(x_ari_aby3));
        msbx_s = Ref2Value(msbx_s_ref);
#else
                msbx_s = msb_a2b(sctx_aby3.get(), x_ari_aby3);
#endif
      }

      auto msb_spu = std::chrono::high_resolution_clock::now();
      auto duration_spu = std::chrono::duration_cast<std::chrono::microseconds>(
          msb_spu - msb_start);
      if (lctx.get()->Rank() == 0)
        std::cout << "------------------------ MSB, spu" << std::endl;
      if (lctx.get()->Rank() == 0)
        std::cout << "msb micro seconds: " << duration_spu.count() << std::endl;
      if (lctx.get()->Rank() == 0)
        std::cout << "msb comm: " << GetComm(aby3).comm - comm << std::endl;
      if (lctx.get()->Rank() == 0)
        std::cout << "msb latency: " << GetComm(aby3).latency - latency
                  << std::endl;

      // ---------------------------------------------------------------

      comm = GetComm(ours).comm;
      latency = GetComm(ours).latency;
      if (lctx.get()->Rank() == 0) OfflineRecorder::StartRecorder();

      // auto x_ari_test = x_ari_ours;
      // Value2Ref(x_ari_test).eltype() =
      // spu::makeType<aby3::AShrTy>(field); auto msbx_ours_s =
      // Value2Ref(msb_a2b(sctx_aby3.get(), x_ari));

      spu::NdArrayRef msbx_ours_s;
      for (int i = 0; i < EXP_TIMES; i++) {
#ifdef ALKAID_USE_BITWIDTH_16
        msbx_ours_s = oursnp::MsbA2BMultiFanInForBitwidth16(
            &kectx_ours, Value2Ref(x_ari_ours));
#else
                msbx_ours_s = oursnp::MsbA2BMultiFanIn(&kectx_ours, Value2Ref(x_ari_ours));
#endif
      }

      auto msb_ours = std::chrono::high_resolution_clock::now();
      auto duration_ours =
          std::chrono::duration_cast<std::chrono::microseconds>(msb_ours -
                                                                msb_spu);
      if (lctx.get()->Rank() == 0) OfflineRecorder::StopRecorder();
      if (lctx.get()->Rank() == 0)
        std::cout << "------------------------ MSB, ours" << std::endl;
      if (lctx.get()->Rank() == 0)
        std::cout << "msb micro seconds: " << duration_ours.count()
                  << std::endl;
      if (lctx.get()->Rank() == 0)
        std::cout << "msb comm: " << GetComm(ours).comm - comm << std::endl;
      if (lctx.get()->Rank() == 0)
        std::cout << "msb latency: " << GetComm(ours).latency - latency
                  << std::endl;

      // Check output.
      auto msb_p_spu = OpenRef(Value2Ref(msbx_s), aby3);
      auto msb_p_ours = OpenRef(msbx_ours_s, ours);
      if (lctx.get()->Rank() == 0) printResult(msb_p_spu.data(), "msb, spu");
      if (lctx.get()->Rank() == 0) printResult(msb_p_ours.data(), "msb, ours");
      if (lctx.get()->Rank() == 0)
        checkOutput(msb_p_spu.data(), msb_p_ours.data(), "msb");
    }

    /**
     * ----------------------------------------------
     *                Test: B2A.
     * ----------------------------------------------
     */
    {
      if (lctx.get()->Rank() == 0)
        std::cout << "##############################################\nB2A\n####"
                     "##########################################"
                  << std::endl;

      size_t comm = GetComm(aby3).comm;
      size_t latency = GetComm(aby3).latency;
      auto b2a_start = std::chrono::high_resolution_clock::now();

      spu::Value x_a;
      for (int i = 0; i < EXP_TIMES; i++) {
#ifdef ALKAID_USE_BITWIDTH_16
        auto x_a_ref =
            aby3np::B2AForBitwidth16(&kectx_aby3, Value2Ref(x_sb_aby3));
        x_a = Ref2Value(x_a_ref);
#else
                x_a = b2a(sctx_aby3.get(), x_sb_aby3);
#endif
      }

      auto b2a_spu = std::chrono::high_resolution_clock::now();
      auto duration_spu = std::chrono::duration_cast<std::chrono::microseconds>(
          b2a_spu - b2a_start);
      if (lctx.get()->Rank() == 0)
        std::cout << "------------------------ B2A, spu" << std::endl;
      if (lctx.get()->Rank() == 0)
        std::cout << "b2a micro seconds: " << duration_spu.count() << std::endl;
      if (lctx.get()->Rank() == 0)
        std::cout << "b2a comm: " << GetComm(aby3).comm - comm << std::endl;
      if (lctx.get()->Rank() == 0)
        std::cout << "b2a latency: " << GetComm(aby3).latency - latency
                  << std::endl;

      // ---------------------------------------------------------------

      comm = GetComm(ours).comm;
      latency = GetComm(ours).latency;
      if (lctx.get()->Rank() == 0) OfflineRecorder::StartRecorder();

      spu::NdArrayRef x_a_ours;
      for (int i = 0; i < EXP_TIMES; i++) {
#ifdef ALKAID_USE_BITWIDTH_16
        x_a_ours = oursnp::B2AMultiFanInForBitwidth16(&kectx_ours,
                                                      Value2Ref(x_sb_ours));
#else
                x_a_ours = oursnp::B2AMultiFanIn(&kectx_ours, Value2Ref(x_sb_ours));
#endif
      }

      auto b2a_ours = std::chrono::high_resolution_clock::now();
      auto duration_ours =
          std::chrono::duration_cast<std::chrono::microseconds>(b2a_ours -
                                                                b2a_spu);
      if (lctx.get()->Rank() == 0) OfflineRecorder::StopRecorder();
      if (lctx.get()->Rank() == 0)
        std::cout << "------------------------ B2A, ours" << std::endl;
      if (lctx.get()->Rank() == 0)
        std::cout << "b2a micro seconds: " << duration_ours.count()
                  << std::endl;
      if (lctx.get()->Rank() == 0)
        std::cout << "b2a comm: " << GetComm(ours).comm - comm << std::endl;
      if (lctx.get()->Rank() == 0)
        std::cout << "b2a latency: " << GetComm(ours).latency - latency
                  << std::endl;

      // Check output.
      auto x_p_spu = OpenRef(Value2Ref(a2b(sctx_aby3.get(), x_a)), aby3);
      auto x_p_ours =
          OpenRef(oursnp::A2BMultiFanIn(&kectx_ours, x_a_ours), ours);
      if (lctx.get()->Rank() == 0) printResult(x_p_spu.data(), "b2a, spu");
      if (lctx.get()->Rank() == 0) printResult(x_p_ours.data(), "b2a, ours");
      if (lctx.get()->Rank() == 0)
        checkOutput(x_p_spu.data(), x_p_ours.data(), "b2a");
    }

    /**
     * ----------------------------------------------
     *                Test: A2B.
     * ----------------------------------------------
     */
    {
      if (lctx.get()->Rank() == 0)
        std::cout << "##############################################\nA2B\n####"
                     "##########################################"
                  << std::endl;

      auto x_ari_aby3 = p2a(sctx_aby3.get(), x_p_aby3);
      auto x_ari_ours = p2a(sctx_ours.get(), x_p_ours);

      size_t comm = GetComm(aby3).comm;
      size_t latency = GetComm(aby3).latency;
      auto a2b_start = std::chrono::high_resolution_clock::now();

      spu::Value x_b;
      for (int i = 0; i < EXP_TIMES; i++) {
#ifdef ALKAID_USE_BITWIDTH_16
        auto x_b_ref =
            aby3np::A2BForBitwidth16(&kectx_aby3, Value2Ref(x_ari_aby3));
        x_b = Ref2Value(x_b_ref);
#else
                x_b = a2b(sctx_aby3.get(), x_ari_aby3);
#endif
      }

      auto a2b_spu = std::chrono::high_resolution_clock::now();
      auto duration_spu = std::chrono::duration_cast<std::chrono::microseconds>(
          a2b_spu - a2b_start);
      if (lctx.get()->Rank() == 0)
        std::cout << "------------------------ A2B, spu" << std::endl;
      if (lctx.get()->Rank() == 0)
        std::cout << "a2b micro seconds: " << duration_spu.count() << std::endl;
      if (lctx.get()->Rank() == 0)
        std::cout << "a2b comm: " << GetComm(aby3).comm - comm << std::endl;
      if (lctx.get()->Rank() == 0)
        std::cout << "a2b latency: " << GetComm(aby3).latency - latency
                  << std::endl;

      // ---------------------------------------------------------------

      comm = GetComm(ours).comm;
      latency = GetComm(ours).latency;
      if (lctx.get()->Rank() == 0) OfflineRecorder::StartRecorder();

      spu::NdArrayRef x_b_ours;
      for (int i = 0; i < EXP_TIMES; i++) {
#ifdef ALKAID_USE_BITWIDTH_16
        x_b_ours = oursnp::A2BMultiFanInForBitwidth16(
            &kectx_ours, Value2Ref(x_ari_ours));
#else
                x_b_ours = oursnp::A2BMultiFanIn(&kectx_ours, Value2Ref(x_ari_ours));
#endif
      }

      auto a2b_ours = std::chrono::high_resolution_clock::now();
      auto duration_ours =
          std::chrono::duration_cast<std::chrono::microseconds>(a2b_ours -
                                                                a2b_spu);
      if (lctx.get()->Rank() == 0) OfflineRecorder::StopRecorder();
      if (lctx.get()->Rank() == 0)
        std::cout << "------------------------ A2B, ours" << std::endl;
      if (lctx.get()->Rank() == 0)
        std::cout << "a2b micro seconds: " << duration_ours.count()
                  << std::endl;
      if (lctx.get()->Rank() == 0)
        std::cout << "a2b comm: " << GetComm(ours).comm - comm << std::endl;
      if (lctx.get()->Rank() == 0)
        std::cout << "a2b latency: " << GetComm(ours).latency - latency
                  << std::endl;

      // Check output.
      auto x_p_spu = OpenRef(Value2Ref(x_b), aby3);
      auto x_p_ours = OpenRef(x_b_ours, ours);
      if (lctx.get()->Rank() == 0) printResult(x_p_spu.data(), "a2b, spu");
      if (lctx.get()->Rank() == 0) printResult(x_p_ours.data(), "a2b, ours");
      if (lctx.get()->Rank() == 0)
        checkOutput(x_p_spu.data(), x_p_ours.data(), "a2b");
    }

    // /**
    //  * ----------------------------------------------
    //  *                Test: PPA.
    //  * ----------------------------------------------
    //  */
    // {
    //   if (lctx.get()->Rank() == 0) std::cout <<
    //   "##############################################\nPPA\n##############################################"
    //   << std::endl;

    //   size_t comm = GetComm(aby3).comm;
    //   size_t latency = GetComm(aby3).latency;
    //   auto b2a_start = std::chrono::high_resolution_clock::now();

    //   spu::Value x_a;
    //   for (int i = 0; i < EXP_TIMES; i++) {
    //     #ifdef ALKAID_USE_BITWIDTH_16
    //     x_a = add_bb(sctx_aby3.get(), x_sb_aby3, x_sb_aby3);
    //     #else
    //     x_a = add_bb(sctx_aby3.get(), x_sb_aby3, x_sb_aby3);
    //     #endif
    //   }

    //   auto b2a_spu = std::chrono::high_resolution_clock::now();
    //   auto duration_spu =
    //   std::chrono::duration_cast<std::chrono::microseconds>(b2a_spu -
    //   b2a_start); if (lctx.get()->Rank() == 0) std::cout <<
    //   "------------------------ PPA, spu" << std::endl; if
    //   (lctx.get()->Rank() == 0) std::cout << "ppa micro seconds: " <<
    //   duration_spu.count() << std::endl; if (lctx.get()->Rank() == 0)
    //   std::cout << "ppa comm: " << GetComm(aby3).comm - comm << std::endl; if
    //   (lctx.get()->Rank() == 0) std::cout << "ppa latency: " <<
    //   GetComm(aby3).latency - latency << std::endl;

    //   // ---------------------------------------------------------------

    //   comm = GetComm(ours).comm;
    //   latency = GetComm(ours).latency;
    //   if (lctx.get()->Rank() == 0) OfflineRecorder::StartRecorder();

    //   spu::NdArrayRef x_a_ours;
    //   for (int i = 0; i < EXP_TIMES; i++) {
    //     #ifdef ALKAID_USE_BITWIDTH_16
    //     x_a_ours = oursnp::PPATestForBitwidth16(&kectx_ours,
    //     Value2Ref(x_sb_ours)); #else x_a_ours =
    //     oursnp::PPATest(&kectx_ours, Value2Ref(x_sb_ours)); #endif
    //   }

    //   auto b2a_ours = std::chrono::high_resolution_clock::now();
    //   auto duration_ours =
    //   std::chrono::duration_cast<std::chrono::microseconds>(b2a_ours -
    //   b2a_spu); if (lctx.get()->Rank() == 0) OfflineRecorder::StopRecorder();
    //   if (lctx.get()->Rank() == 0) std::cout << "------------------------
    //   PPA, ours" << std::endl; if (lctx.get()->Rank() == 0) std::cout << "ppa
    //   micro seconds: " << duration_ours.count() << std::endl; if
    //   (lctx.get()->Rank() == 0) std::cout << "ppa comm: " <<
    //   GetComm(ours).comm - comm << std::endl; if (lctx.get()->Rank() == 0)
    //   std::cout << "ppa latency: " << GetComm(ours).latency - latency <<
    //   std::endl;

    //   // Check output.
    //   // auto x_p_spu = OpenRef(Value2Ref(a2b(sctx_aby3.get(), x_a)),
    //   aby3);
    //   // auto x_p_ours = OpenRef(oursnp::A2BMultiFanIn(&kectx_ours,
    //   x_a_ours), ours);
    //   // // if (lctx.get()->Rank() == 0) printResult(msbx_p.data(), "msb,
    //   spu");
    //   // // if (lctx.get()->Rank() == 0) printResult(msbx_ours_p.data(),
    //   "msb, ours");
    //   // if (lctx.get()->Rank() == 0) checkOutput(x_p_spu.data(),
    //   x_p_ours.data(), "b2a");
    // }

    // // 64-fan-in
    // {
    //   auto input00 = p2b(sctx.get(), rand_p(sctx.get(), {M * 32, N})); // 64
    //   -> 32 auto input01 = p2b(sctx.get(), rand_p(sctx.get(), {M * 32, N}));
    //   // 64 -> 32 auto input10 = p2b(sctx.get(), rand_p(sctx.get(), {M * 16,
    //   N})); // 32 -> 16 auto input11 = p2b(sctx.get(), rand_p(sctx.get(), {M
    //   * 16, N})); // 32 -> 16 auto input12 = p2b(sctx.get(),
    //   rand_p(sctx.get(), {M * 16, N})); // 32 -> 16 auto input13 =
    //   p2b(sctx.get(), rand_p(sctx.get(), {M * 16, N})); // 32 -> 16 auto
    //   input20 = p2b(sctx.get(), rand_p(sctx.get(), {M * 8, N} )); // 16 -> 8
    //   auto input21 = p2b(sctx.get(), rand_p(sctx.get(), {M * 8, N} )); // 16
    //   -> 8 auto input30 = p2b(sctx.get(), rand_p(sctx.get(), {M * 4, N} ));
    //   // 8 -> 4 auto input31 = p2b(sctx.get(), rand_p(sctx.get(), {M * 4, N}
    //   )); // 8 -> 4 auto input32 = p2b(sctx.get(), rand_p(sctx.get(), {M * 4,
    //   N} )); // 8 -> 4 auto input33 = p2b(sctx.get(), rand_p(sctx.get(), {M *
    //   4, N} )); // 8 -> 4 auto input40 = p2b(sctx.get(), rand_p(sctx.get(),
    //   {M * 2, N} )); // 4 -> 2 auto input41 = p2b(sctx.get(),
    //   rand_p(sctx.get(), {M * 2, N} )); // 4 -> 2 auto input50 =
    //   p2b(sctx.get(), rand_p(sctx.get(), {M, N}     )); // 2 -> 1 auto
    //   input51 = p2b(sctx.get(), rand_p(sctx.get(), {M, N}     )); // 2 -> 1
    //   auto input52 = p2b(sctx.get(), rand_p(sctx.get(), {M, N}     )); // 2
    //   -> 1 auto input53 = p2b(sctx.get(), rand_p(sctx.get(), {M, N}     ));
    //   // 2 -> 1

    //   // spu naive and for 64 input
    //   auto start = std::chrono::high_resolution_clock::now();
    //   size_t comm = GetComm.comm;
    //   size_t latency = GetComm.latency;
    //   auto result0 = and_bb(sctx.get(), input00, input01);
    //   auto result1 = and_bb(sctx.get(), input10, input11);
    //   auto result2 = and_bb(sctx.get(), input20, input21);
    //   auto result3 = and_bb(sctx.get(), input30, input31);
    //   auto result4 = and_bb(sctx.get(), input40, input41);
    //   auto result5 = and_bb(sctx.get(), input50, input51);
    //   auto end = std::chrono::high_resolution_clock::now();
    //   auto duration =
    //   std::chrono::duration_cast<std::chrono::microseconds>(end - start); if
    //   (lctx.get()->Rank() == 0) std::cout << "------------------------
    //   64-fan-in, spu" << std::endl; if (lctx.get()->Rank() == 0) std::cout <<
    //   "64-fan-in micro seconds: " << duration.count() << std::endl; if
    //   (lctx.get()->Rank() == 0) std::cout << "64-fan-in sent: " <<
    //   GetComm.comm - comm << std::endl; if (lctx.get()->Rank() == 0)
    //   std::cout << "64-fan-in latency: " << GetComm.latency - latency <<
    //   std::endl;

    //   // ours naive and for 64 input
    //   auto input10_mss = oursnp::ResharingRss2Mrss(&kectx,
    //   Value2Ref(input10)); auto input11_mss =
    //   oursnp::ResharingRss2Mrss(&kectx, Value2Ref(input11)); auto
    //   input12_mss = oursnp::ResharingRss2Mrss(&kectx,
    //   Value2Ref(input12)); auto input13_mss =
    //   oursnp::ResharingRss2Mrss(&kectx, Value2Ref(input13)); auto
    //   input30_mss = oursnp::ResharingRss2Mrss(&kectx,
    //   Value2Ref(input30)); auto input31_mss =
    //   oursnp::ResharingRss2Mrss(&kectx, Value2Ref(input31)); auto
    //   input32_mss = oursnp::ResharingRss2Mrss(&kectx,
    //   Value2Ref(input32)); auto input33_mss =
    //   oursnp::ResharingRss2Mrss(&kectx, Value2Ref(input33)); auto
    //   input50_mss = oursnp::ResharingRss2Mrss(&kectx,
    //   Value2Ref(input50)); auto input51_mss =
    //   oursnp::ResharingRss2Mrss(&kectx, Value2Ref(input51)); auto
    //   input52_mss = oursnp::ResharingRss2Mrss(&kectx,
    //   Value2Ref(input52)); auto input53_mss =
    //   oursnp::ResharingRss2Mrss(&kectx, Value2Ref(input53));

    //   start = std::chrono::high_resolution_clock::now();
    //   comm = GetComm.comm;
    //   latency = GetComm.latency;
    //   auto result1_ours = oursnp::MrssAnd4NoComm(&kectx, input10_mss,
    //   input11_mss, input12_mss, input13_mss); auto result1_mss =
    //   oursnp::ResharingAss2Mrss(&kectx, result1_ours); auto result3_ours =
    //   oursnp::MrssAnd4NoComm(&kectx, input30_mss, input31_mss, input32_mss,
    //   input33_mss); auto result3_mss = oursnp::ResharingAss2Mrss(&kectx,
    //   result3_ours); auto result5_ours = oursnp::MrssAnd4NoComm(&kectx,
    //   input50_mss, input51_mss, input52_mss, input53_mss); auto result5_mss =
    //   oursnp::ResharingAss2Mrss(&kectx, result5_ours); auto end_ours =
    //   std::chrono::high_resolution_clock::now(); auto duration_ours =
    //   std::chrono::duration_cast<std::chrono::microseconds>(end_ours -
    //   start); if (lctx.get()->Rank() == 0) std::cout <<
    //   "------------------------ 64-fan-in, ours" << std::endl; if
    //   (lctx.get()->Rank() == 0) std::cout << "64-fan-in micro seconds: " <<
    //   duration_ours.count() << std::endl; if (lctx.get()->Rank() == 0)
    //   std::cout << "64-fan-in sent: " << GetComm.comm - comm << std::endl; if
    //   (lctx.get()->Rank() == 0) std::cout << "64-fan-in latency: " <<
    //   GetComm.latency - latency << std::endl;
    //   // oursnp::MrssAnd4NoComm(&kectx, x_mss, y_mss, a_mss, b_mss);

    // }
  });
}