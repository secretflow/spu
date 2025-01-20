// #define EQ_DO_NOT_USE_OFFLINE

#include <iostream>
#include <string>
#include <chrono>

#include "libspu/spu.pb.h"
#include "libspu/core/ndarray_ref.h"

#include "libspu/mpc/api.h"
#include "libspu/mpc/ab_api.h"
#include "libspu/mpc/offline_recorder.h"
#include "libspu/mpc/utils/simulate.h"
#include "libspu/mpc/common/communicator.h"

// hack wrap value for handle namespace error.
#define MyWrapValue(x) spu::Value(x, spu::DT_INVALID)
#define MyUnwrapValue(x) x.data()
#define OpenRef(ndref, tag) b2p(sctx_##tag.get(), MyWrapValue(ndref))
#define GetComm(tag) sctx_##tag.get()->getState<Communicator>()->getStats()

#define USE_ALBO
// #define EQ_USE_BITWIDTH_16
#define NumOutputs 3
#define N 1
#define M 4200
#define EXP_TIMES 100
#define THREADS 48
// #define RANDOM_INPUT 1

// protocols.
#include "libspu/mpc/aby3/protocol.h"
#include "libspu/mpc/aby3/conversion.h"
#include "libspu/mpc/aby3/value.h"
#include "libspu/mpc/aby3/type.h"
namespace aby3np = spu::mpc::aby3;        // aby3 namespace
#ifdef USE_ALBO
#include "libspu/mpc/albo/protocol.h"
#include "libspu/mpc/albo/conversion.h"
#include "libspu/mpc/albo/value.h"
#include "libspu/mpc/albo/type.h"
#include "libspu/mpc/albo/mss_utils.h"
namespace oursnp = spu::mpc::albo;        // ours namespace
#else
#include "libspu/mpc/alkaid/protocol.h"
#include "libspu/mpc/alkaid/conversion.h"
#include "libspu/mpc/alkaid/value.h"
#include "libspu/mpc/alkaid/type.h"
#include "libspu/mpc/alkaid/mss_utils.h"
namespace oursnp = spu::mpc::alkaid;        // ours namespace
#endif

#ifdef EQ_USE_BITWIDTH_16
using ring2k_t = uint16_t;
#else
using ring2k_t = uint64_t;
#endif
using pub_t = std::array<ring2k_t, 1>; 
using namespace spu::mpc;

spu::RuntimeConfig makeConfig(spu::FieldType field, bool use_al=true) {
  spu::RuntimeConfig conf;
  #ifdef USE_ALBO
  if (use_al) conf.set_protocol(spu::ProtocolKind::ALBO);
  #else
  if (use_al) conf.set_protocol(spu::ProtocolKind::ALKAID);
  #endif
  else conf.set_protocol(spu::ProtocolKind::ABY3);
  conf.set_field(field);
  return conf;
}

void printResult(spu::NdArrayRef& result, const std::string& name) {
  spu::NdArrayView<pub_t> _r(result);
  std::cout << name << ": ";
  for (int i = 0; i < NumOutputs; i++) {
    std::cout << _r[i][0] << " ";
  }
  std::cout << std::endl;
}

void checkOutput(spu::NdArrayRef& spu_result, spu::NdArrayRef& ours_result, const std::string& name) {
  spu::NdArrayView<pub_t> _spu(spu_result);
  spu::NdArrayView<pub_t> _ours(ours_result);

  size_t match_count = 0;

  for (int i = 0; i < N * M; i++) 
  {
    if (_spu[i][0] == _ours[i][0]) match_count++;
    // else std::cout << "mismatch at " << i << ": " << _spu[i][0] << " " << _ours[i][0] << std::endl;
  }

  std::cout << name << " match count rate: " << static_cast<double>(match_count) / (N * M) << std::endl;
}

int main()
{
    spu::Shape kShape = {N, M};

    utils::simulate(3,                                                    
        [&](const std::shared_ptr<yacl::link::Context>& lctx) 
        { 
            /**
             * ----------------------------------------------a
             *                Syetem setup.
             * ----------------------------------------------
             */
            auto field = spu::FieldType::FM64;        
            spu::RuntimeConfig config_ours = makeConfig(field, true);
            #ifdef USE_ALBO
            auto sctx_ours = makeAlboProtocol(config_ours, lctx);
            #else
            auto sctx_ours = makeAlkaidProtocol(config_ours, lctx);
            #endif
            auto kectx_ours = spu::KernelEvalContext(sctx_ours.get());    
            spu::RuntimeConfig config_aby3 = makeConfig(field, false);
            auto sctx_aby3 = makeAby3Protocol(config_aby3, lctx); 
            auto kectx_aby3 = spu::KernelEvalContext(sctx_aby3.get());   
            yacl::set_num_threads(THREADS);

            /**
             * ----------------------------------------------
             *                Input random x, y.
             * ----------------------------------------------
             */
            #ifdef RANDOM_INPUT
            auto x_p_aby3 = rand_p(sctx_aby3.get(), kShape);
            // MyUnwrapValue(x_p_ours).eltype() = spu::makeType<oursnp::Pub2kTy>(field);
            auto y_p_aby3 = rand_p(sctx_aby3.get(), kShape);
            // MyUnwrapValue(y_p_ours).eltype() = spu::makeType<oursnp::Pub2kTy>(field);
            #else
            // auto x_p = make_p(sctx_ours.get(), static_cast<uint128_t>(16813041318660192066ull), kShape);
            #ifdef EQ_USE_BITWIDTH_16
            auto x_p_aby3 = make_p(sctx_aby3.get(), static_cast<uint128_t>(1ull << 12), kShape);
            auto y_p_aby3 = make_p(sctx_aby3.get(), static_cast<uint128_t>(1ull << 12), kShape);
            #else
            auto x_p_aby3 = make_p(sctx_aby3.get(), static_cast<uint128_t>(1ull << 63), kShape);
            auto y_p_aby3 = make_p(sctx_aby3.get(), static_cast<uint128_t>(1ull << 63), kShape);
            #endif
            #endif
            auto x_p_ours = x_p_aby3;
            auto y_p_ours = y_p_aby3;
            if (lctx.get()->Rank() == 0) std::cout << "Public input initialize." << std::endl;
            auto x_sb_aby3 = p2b(sctx_aby3.get(), x_p_aby3);
            auto y_sb_aby3 = p2b(sctx_aby3.get(), y_p_aby3);
            if (lctx.get()->Rank() == 0) std::cout << "ABY3 sharing is converted." << std::endl;
            auto x_sb_ours = p2b(sctx_ours.get(), x_p_ours);
            auto y_sb_ours = p2b(sctx_ours.get(), y_p_ours);
            if (lctx.get()->Rank() == 0) std::cout << "Ours sharing is converted." << std::endl;
            #ifdef EQ_USE_BITWIDTH_16
            MyUnwrapValue(x_sb_aby3).eltype() = spu::makeType<aby3np::BShrTy>(
              aby3np::calcBShareBacktype(16), 16);
            MyUnwrapValue(y_sb_aby3).eltype() = spu::makeType<aby3np::BShrTy>(
              aby3np::calcBShareBacktype(16), 16);
            MyUnwrapValue(x_sb_ours).eltype() = spu::makeType<oursnp::BShrTy>(
              oursnp::calcBShareBacktype(16), 16);
            MyUnwrapValue(y_sb_ours).eltype() = spu::makeType<oursnp::BShrTy>(
              oursnp::calcBShareBacktype(16), 16);
            #endif
            if (lctx.get()->Rank() == 0) printResult(x_p_aby3.data(), "Input x: ");
            if (lctx.get()->Rank() == 0) printResult(y_p_aby3.data(), "Input y: ");
            if (lctx.get()->Rank() == 0) std::cout << "Number of threads: " << yacl::get_num_threads() << std::endl;

            /**
             * ----------------------------------------------
             *                Test: ExtMsb.
             * ----------------------------------------------
             */
            {
              if (lctx.get()->Rank() == 0) std::cout << \
              "##############################################\nExtMsb\n##############################################" \
              << std::endl;
              auto x_ari_aby3 = p2a(sctx_aby3.get(), x_p_aby3);
              auto x_ari_ours = p2a(sctx_ours.get(), x_p_ours);
              if (lctx.get()->Rank() == 0) std::cout << "Prepare input." << std::endl;

              size_t comm = GetComm(aby3).comm;
              size_t latency = GetComm(aby3).latency;
              auto msb_start = std::chrono::high_resolution_clock::now();

              spu::Value msbx_s;
              for (int i = 0; i < EXP_TIMES; i++) {
                #ifdef EQ_USE_BITWIDTH_16
                auto msbx_s_ref = aby3np::MsbA2BForBitwidth16(&kectx_aby3, MyUnwrapValue(x_ari_aby3));
                msbx_s = MyWrapValue(msbx_s_ref);
                #else
                msbx_s = msb_a2b(sctx_aby3.get(), x_ari_aby3);
                #endif
              }

              auto msb_spu = std::chrono::high_resolution_clock::now();
              auto duration_spu = std::chrono::duration_cast<std::chrono::microseconds>(msb_spu - msb_start);
              if (lctx.get()->Rank() == 0) std::cout << "------------------------ MSB, spu" << std::endl;
              if (lctx.get()->Rank() == 0) std::cout << "msb micro seconds: " << duration_spu.count() << std::endl;
              if (lctx.get()->Rank() == 0) std::cout << "msb comm: " << GetComm(aby3).comm - comm << std::endl;
              if (lctx.get()->Rank() == 0) std::cout << "msb latency: " << GetComm(aby3).latency - latency << std::endl;

              // ---------------------------------------------------------------              
              
              comm = GetComm(ours).comm;
              latency = GetComm(ours).latency;
              if (lctx.get()->Rank() == 0) OfflineRecorder::StartRecorder();

              // auto x_ari_test = x_ari_ours;
              // MyUnwrapValue(x_ari_test).eltype() = spu::makeType<aby3::AShrTy>(field);
              // auto msbx_ours_s = MyUnwrapValue(msb_a2b(sctx_aby3.get(), x_ari));

              spu::NdArrayRef msbx_ours_s;
              for (int i = 0; i < EXP_TIMES; i++) {
                #ifdef EQ_USE_BITWIDTH_16
                msbx_ours_s = oursnp::MsbA2BMultiFanInForBitwidth16(&kectx_ours, MyUnwrapValue(x_ari_ours));
                #else
                msbx_ours_s = oursnp::MsbA2BMultiFanIn(&kectx_ours, MyUnwrapValue(x_ari_ours));
                #endif
              }

              auto msb_ours = std::chrono::high_resolution_clock::now();
              auto duration_ours = std::chrono::duration_cast<std::chrono::microseconds>(msb_ours - msb_spu);
              if (lctx.get()->Rank() == 0) OfflineRecorder::StopRecorder();
              if (lctx.get()->Rank() == 0) std::cout << "------------------------ MSB, ours" << std::endl;
              if (lctx.get()->Rank() == 0) std::cout << "msb micro seconds: " << duration_ours.count() << std::endl;
              if (lctx.get()->Rank() == 0) std::cout << "msb comm: " << GetComm(ours).comm - comm << std::endl;
              if (lctx.get()->Rank() == 0) std::cout << "msb latency: " << GetComm(ours).latency - latency << std::endl;

              // Check output.
              auto msb_p_spu = OpenRef(MyUnwrapValue(msbx_s), aby3);
              auto msb_p_ours = OpenRef(msbx_ours_s, ours);
              if (lctx.get()->Rank() == 0) printResult(msb_p_spu.data(), "msb, spu");
              if (lctx.get()->Rank() == 0) printResult(msb_p_ours.data(), "msb, ours");
              if (lctx.get()->Rank() == 0) checkOutput(msb_p_spu.data(), msb_p_ours.data(), "msb");
            }

            /**
             * ----------------------------------------------
             *                Test: B2A.
             * ----------------------------------------------
             */
            {
              if (lctx.get()->Rank() == 0) std::cout << \
              "##############################################\nB2A\n##############################################" \
              << std::endl;

              size_t comm = GetComm(aby3).comm;
              size_t latency = GetComm(aby3).latency;
              auto b2a_start = std::chrono::high_resolution_clock::now();

              spu::Value x_a;
              for (int i = 0; i < EXP_TIMES; i++) {
                #ifdef EQ_USE_BITWIDTH_16
                auto x_a_ref = aby3np::B2AForBitwidth16(&kectx_aby3, MyUnwrapValue(x_sb_aby3));
                x_a = MyWrapValue(x_a_ref);
                #else
                x_a = b2a(sctx_aby3.get(), x_sb_aby3);
                #endif
              }

              auto b2a_spu = std::chrono::high_resolution_clock::now();
              auto duration_spu = std::chrono::duration_cast<std::chrono::microseconds>(b2a_spu - b2a_start);
              if (lctx.get()->Rank() == 0) std::cout << "------------------------ B2A, spu" << std::endl;
              if (lctx.get()->Rank() == 0) std::cout << "b2a micro seconds: " << duration_spu.count() << std::endl;
              if (lctx.get()->Rank() == 0) std::cout << "b2a comm: " << GetComm(aby3).comm - comm << std::endl;
              if (lctx.get()->Rank() == 0) std::cout << "b2a latency: " << GetComm(aby3).latency - latency << std::endl;

              // ---------------------------------------------------------------          

              comm = GetComm(ours).comm;
              latency = GetComm(ours).latency;
              if (lctx.get()->Rank() == 0) OfflineRecorder::StartRecorder();

              spu::NdArrayRef x_a_ours;
              for (int i = 0; i < EXP_TIMES; i++) {
                #ifdef EQ_USE_BITWIDTH_16
                x_a_ours = oursnp::B2AMultiFanInForBitwidth16(&kectx_ours, MyUnwrapValue(x_sb_ours));
                #else
                x_a_ours = oursnp::B2AMultiFanIn(&kectx_ours, MyUnwrapValue(x_sb_ours));
                #endif
              }

              auto b2a_ours = std::chrono::high_resolution_clock::now();
              auto duration_ours = std::chrono::duration_cast<std::chrono::microseconds>(b2a_ours - b2a_spu);
              if (lctx.get()->Rank() == 0) OfflineRecorder::StopRecorder();
              if (lctx.get()->Rank() == 0) std::cout << "------------------------ B2A, ours" << std::endl;
              if (lctx.get()->Rank() == 0) std::cout << "b2a micro seconds: " << duration_ours.count() << std::endl;
              if (lctx.get()->Rank() == 0) std::cout << "b2a comm: " << GetComm(ours).comm - comm << std::endl;
              if (lctx.get()->Rank() == 0) std::cout << "b2a latency: " << GetComm(ours).latency - latency << std::endl;

              // Check output.
              auto x_p_spu = OpenRef(MyUnwrapValue(a2b(sctx_aby3.get(), x_a)), aby3);
              auto x_p_ours = OpenRef(oursnp::A2BMultiFanIn(&kectx_ours, x_a_ours), ours);
              // if (lctx.get()->Rank() == 0) printResult(msbx_p.data(), "msb, spu");
              // if (lctx.get()->Rank() == 0) printResult(msbx_ours_p.data(), "msb, ours");
              if (lctx.get()->Rank() == 0) checkOutput(x_p_spu.data(), x_p_ours.data(), "b2a");
            }

            /**
             * ----------------------------------------------
             *                Test: A2B.
             * ----------------------------------------------
             */
            {
              if (lctx.get()->Rank() == 0) std::cout << \
              "##############################################\nA2B\n##############################################" \
              << std::endl;

              auto x_ari_aby3 = p2a(sctx_aby3.get(), x_p_aby3);
              auto x_ari_ours = p2a(sctx_ours.get(), x_p_ours);

              size_t comm = GetComm(aby3).comm;
              size_t latency = GetComm(aby3).latency;
              auto a2b_start = std::chrono::high_resolution_clock::now();

              spu::Value x_b;
              for (int i = 0; i < EXP_TIMES; i++) {
                #ifdef EQ_USE_BITWIDTH_16
                auto x_b_ref = aby3np::A2BForBitwidth16(&kectx_aby3, MyUnwrapValue(x_ari_aby3));
                x_b = MyWrapValue(x_b_ref);
                #else
                x_b = a2b(sctx_aby3.get(), x_ari_aby3);
                #endif
              }

              auto a2b_spu = std::chrono::high_resolution_clock::now();
              auto duration_spu = std::chrono::duration_cast<std::chrono::microseconds>(a2b_spu - a2b_start);
              if (lctx.get()->Rank() == 0) std::cout << "------------------------ A2B, spu" << std::endl;
              if (lctx.get()->Rank() == 0) std::cout << "a2b micro seconds: " << duration_spu.count() << std::endl;
              if (lctx.get()->Rank() == 0) std::cout << "a2b comm: " << GetComm(aby3).comm - comm << std::endl;
              if (lctx.get()->Rank() == 0) std::cout << "a2b latency: " << GetComm(aby3).latency - latency << std::endl;

              // ---------------------------------------------------------------          

              comm = GetComm(ours).comm;
              latency = GetComm(ours).latency;
              if (lctx.get()->Rank() == 0) OfflineRecorder::StartRecorder();

              spu::NdArrayRef x_b_ours;
              for (int i = 0; i < EXP_TIMES; i++) {
                #ifdef EQ_USE_BITWIDTH_16
                x_b_ours = oursnp::A2BMultiFanInForBitwidth16(&kectx_ours, MyUnwrapValue(x_ari_ours));
                #else
                x_b_ours = oursnp::A2BMultiFanIn(&kectx_ours, MyUnwrapValue(x_ari_ours));
                #endif
              }

              auto a2b_ours = std::chrono::high_resolution_clock::now();
              auto duration_ours = std::chrono::duration_cast<std::chrono::microseconds>(a2b_ours - a2b_spu);
              if (lctx.get()->Rank() == 0) OfflineRecorder::StopRecorder();
              if (lctx.get()->Rank() == 0) std::cout << "------------------------ A2B, ours" << std::endl;
              if (lctx.get()->Rank() == 0) std::cout << "a2b micro seconds: " << duration_ours.count() << std::endl;
              if (lctx.get()->Rank() == 0) std::cout << "a2b comm: " << GetComm(ours).comm - comm << std::endl;
              if (lctx.get()->Rank() == 0) std::cout << "a2b latency: " << GetComm(ours).latency - latency << std::endl;

              // Check output.
              auto x_p_spu = OpenRef(MyUnwrapValue(x_b), aby3);
              auto x_p_ours = OpenRef(x_b_ours, ours);
              // if (lctx.get()->Rank() == 0) printResult(msbx_p.data(), "msb, spu");
              // if (lctx.get()->Rank() == 0) printResult(msbx_ours_p.data(), "msb, ours");
              if (lctx.get()->Rank() == 0) checkOutput(x_p_spu.data(), x_p_ours.data(), "a2b");
            }

            /**
             * ----------------------------------------------
             *                Test: PPA.
             * ----------------------------------------------
             */
            {
              if (lctx.get()->Rank() == 0) std::cout << \
              "##############################################\nPPA\n##############################################" \
              << std::endl;

              size_t comm = GetComm(aby3).comm;
              size_t latency = GetComm(aby3).latency;
              auto b2a_start = std::chrono::high_resolution_clock::now();

              spu::Value x_a;
              for (int i = 0; i < EXP_TIMES; i++) {
                #ifdef EQ_USE_BITWIDTH_16
                x_a = add_bb(sctx_aby3.get(), x_sb_aby3, x_sb_aby3);
                #else
                x_a = add_bb(sctx_aby3.get(), x_sb_aby3, x_sb_aby3);
                #endif
              }

              auto b2a_spu = std::chrono::high_resolution_clock::now();
              auto duration_spu = std::chrono::duration_cast<std::chrono::microseconds>(b2a_spu - b2a_start);
              if (lctx.get()->Rank() == 0) std::cout << "------------------------ PPA, spu" << std::endl;
              if (lctx.get()->Rank() == 0) std::cout << "ppa micro seconds: " << duration_spu.count() << std::endl;
              if (lctx.get()->Rank() == 0) std::cout << "ppa comm: " << GetComm(aby3).comm - comm << std::endl;
              if (lctx.get()->Rank() == 0) std::cout << "ppa latency: " << GetComm(aby3).latency - latency << std::endl;

              // ---------------------------------------------------------------          

              comm = GetComm(ours).comm;
              latency = GetComm(ours).latency;
              if (lctx.get()->Rank() == 0) OfflineRecorder::StartRecorder();

              spu::NdArrayRef x_a_ours;
              for (int i = 0; i < EXP_TIMES; i++) {
                #ifdef EQ_USE_BITWIDTH_16
                x_a_ours = oursnp::PPATestForBitwidth16(&kectx_ours, MyUnwrapValue(x_sb_ours));
                #else
                x_a_ours = oursnp::PPATest(&kectx_ours, MyUnwrapValue(x_sb_ours));
                #endif
              }

              auto b2a_ours = std::chrono::high_resolution_clock::now();
              auto duration_ours = std::chrono::duration_cast<std::chrono::microseconds>(b2a_ours - b2a_spu);
              if (lctx.get()->Rank() == 0) OfflineRecorder::StopRecorder();
              if (lctx.get()->Rank() == 0) std::cout << "------------------------ PPA, ours" << std::endl;
              if (lctx.get()->Rank() == 0) std::cout << "ppa micro seconds: " << duration_ours.count() << std::endl;
              if (lctx.get()->Rank() == 0) std::cout << "ppa comm: " << GetComm(ours).comm - comm << std::endl;
              if (lctx.get()->Rank() == 0) std::cout << "ppa latency: " << GetComm(ours).latency - latency << std::endl;

              // Check output.
              // auto x_p_spu = OpenRef(MyUnwrapValue(a2b(sctx_aby3.get(), x_a)), aby3);
              // auto x_p_ours = OpenRef(oursnp::A2BMultiFanIn(&kectx_ours, x_a_ours), ours);
              // // if (lctx.get()->Rank() == 0) printResult(msbx_p.data(), "msb, spu");
              // // if (lctx.get()->Rank() == 0) printResult(msbx_ours_p.data(), "msb, ours");
              // if (lctx.get()->Rank() == 0) checkOutput(x_p_spu.data(), x_p_ours.data(), "b2a");
            }

            /**
             * ----------------------------------------------
             *                Test: Offline.
             * ----------------------------------------------
             */
            {
              if (lctx.get()->Rank() == 0) std::cout << \
              "##############################################\nOffline\n##############################################" \
              << std::endl;

              spu::Shape newkShape = {N, M};
              if (lctx.get()->Rank() == 0) std::cout << "Mult size: " << N * M << std::endl;
              size_t comm = GetComm(ours).comm;
              size_t latency = GetComm(ours).latency;
              auto offline_start = std::chrono::high_resolution_clock::now();
              if (lctx.get()->Rank() == 0) OfflineRecorder::StartRecorder();

              auto x_test_mss = make_p(sctx_aby3.get(), static_cast<uint128_t>(1ull << 7), newkShape);
              auto xb_test_mss = p2b(sctx_ours.get(), x_test_mss);
              auto xb_test_rss = oursnp::ResharingMss2Rss(&kectx_ours, MyUnwrapValue(xb_test_mss));

              spu::NdArrayRef x_b_ours;
              for (int i = 0; i < EXP_TIMES; i++) {
                auto xb_res_ass = oursnp::RssAnd2NoComm(&kectx_ours, xb_test_rss, xb_test_rss);
                auto xb_res_rss = oursnp::ResharingAss2Rss(&kectx_ours, xb_res_ass);
              }

              auto offline_ours = std::chrono::high_resolution_clock::now();
              auto duration_ours = std::chrono::duration_cast<std::chrono::microseconds>(offline_ours - offline_start);
              if (lctx.get()->Rank() == 0) std::cout << "------------------------ offline, ours" << std::endl;
              if (lctx.get()->Rank() == 0) std::cout << "offline micro seconds: " << duration_ours.count() << std::endl;
              if (lctx.get()->Rank() == 0) std::cout << "offline comm: " << GetComm(ours).comm - comm << std::endl;
              if (lctx.get()->Rank() == 0) std::cout << "offline latency: " << GetComm(ours).latency - latency << std::endl;
              if (lctx.get()->Rank() == 0) OfflineRecorder::StopRecorder();
            }

        });    
}