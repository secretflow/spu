// #define EQ_DO_NOT_USE_OFFLINE

#include <iostream>
#include <string>
#include <chrono>

// #define USE_ALBO

#include "libspu/spu.pb.h"
#include "libspu/core/ndarray_ref.h"

#include "libspu/mpc/api.h"
#include "libspu/mpc/ab_api.h"
#ifdef USE_ALBO
#include "libspu/mpc/albo/protocol.h"
#include "libspu/mpc/albo/conversion.h"
#include "libspu/mpc/albo/value.h"
#include "libspu/mpc/albo/type.h"
namespace oursnp = spu::mpc::albo;        // ours namespace
#define OURSP albo
#else
#include "libspu/mpc/alkaid/protocol.h"
#include "libspu/mpc/alkaid/conversion.h"
#include "libspu/mpc/alkaid/value.h"
#include "libspu/mpc/alkaid/type.h"
namespace oursnp = spu::mpc::alkaid;        // ours namespace
#define OURSP alkaid
#endif
#include "libspu/mpc/aby3/protocol.h"
#include "libspu/mpc/aby3/conversion.h"
#include "libspu/mpc/aby3/value.h"
#include "libspu/mpc/aby3/type.h"
#include "libspu/mpc/utils/simulate.h"

#include "libspu/mpc/common/communicator.h"

// hack wrap value for handle namespace error.
#define MyWrapValue(x) spu::Value(x, spu::DT_INVALID)
#define MyUnwrapValue(x) x.data()
#define NumOutputs 3
#define OpenRef(ndref, tag) b2p(sctx_##tag.get(), MyWrapValue(ndref))
#define N 24
#define M 24
#define GetComm(tag) sctx_##tag.get()->getState<Communicator>()->getStats()
#define RANDOM_INPUT 1

using ring2k_t = uint64_t;
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

            /**
             * ----------------------------------------------
             *                Input random x, y.
             * ----------------------------------------------
             */
            #ifdef RANDOM_INPUT
            auto x_p_aby3 = rand_p(sctx_aby3.get(), kShape);
            auto x_p_ours = x_p_aby3;
            // MyUnwrapValue(x_p_ours).eltype() = spu::makeType<oursnp::Pub2kTy>(field);
            auto y_p_aby3 = rand_p(sctx_aby3.get(), kShape);
            auto y_p_ours = y_p_aby3;
            // MyUnwrapValue(y_p_ours).eltype() = spu::makeType<oursnp::Pub2kTy>(field);
            #else
            // auto x_p = make_p(sctx_ours.get(), static_cast<uint128_t>(16813041318660192066ull), kShape);
            auto x_p = make_p(sctx_ours.get(), static_cast<uint128_t>(0ull), kShape);
            auto y_p = make_p(sctx_ours.get(), static_cast<uint128_t>(4), kShape);
            #endif
            if (lctx.get()->Rank() == 0) std::cout << "Public input initialize." << std::endl;
            auto x_s_aby3 = p2b(sctx_aby3.get(), x_p_aby3);
            auto y_s_aby3 = p2b(sctx_aby3.get(), y_p_aby3);
            if (lctx.get()->Rank() == 0) std::cout << "ABY3 sharing is converted." << std::endl;
            auto x_s_ours = p2b(sctx_ours.get(), x_p_ours);
            auto y_s_ours = p2b(sctx_ours.get(), y_p_ours);
            if (lctx.get()->Rank() == 0) std::cout << "Ours sharing is converted." << std::endl;
            if (lctx.get()->Rank() == 0) printResult(x_p_aby3.data(), "Input x: ");
            if (lctx.get()->Rank() == 0) printResult(y_p_aby3.data(), "Input y: ");

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

              auto msbx_s = msb_a2b(sctx_aby3.get(), x_ari_aby3);

              auto msb_spu = std::chrono::high_resolution_clock::now();
              auto duration_spu = std::chrono::duration_cast<std::chrono::microseconds>(msb_spu - msb_start);
              if (lctx.get()->Rank() == 0) std::cout << "------------------------ MSB, spu" << std::endl;
              if (lctx.get()->Rank() == 0) std::cout << "msb micro seconds: " << duration_spu.count() << std::endl;
              if (lctx.get()->Rank() == 0) std::cout << "msb sent: " << GetComm(aby3).comm - comm << std::endl;
              if (lctx.get()->Rank() == 0) std::cout << "msb latency: " << GetComm(aby3).latency - latency << std::endl;

              // ---------------------------------------------------------------              
              
              comm = GetComm(ours).comm;
              latency = GetComm(ours).latency;

              // auto x_ari_test = x_ari_ours;
              // MyUnwrapValue(x_ari_test).eltype() = spu::makeType<aby3::AShrTy>(field);
              // auto msbx_ours_s = MyUnwrapValue(msb_a2b(sctx_aby3.get(), x_ari));

              auto msbx_ours_s = oursnp::MsbA2BMultiFanIn(&kectx_ours, MyUnwrapValue(x_ari_ours));

              auto msb_ours = std::chrono::high_resolution_clock::now();
              auto duration_ours = std::chrono::duration_cast<std::chrono::microseconds>(msb_ours - msb_spu);
              if (lctx.get()->Rank() == 0) std::cout << "------------------------ MSB, ours" << std::endl;
              if (lctx.get()->Rank() == 0) std::cout << "msb micro seconds: " << duration_ours.count() << std::endl;
              if (lctx.get()->Rank() == 0) std::cout << "msb sent: " << GetComm(ours).comm - comm << std::endl;
              if (lctx.get()->Rank() == 0) std::cout << "msb latency: " << GetComm(ours).latency - latency << std::endl;

              // Check output.
              auto msb_p_spu = OpenRef(MyUnwrapValue(msbx_s), aby3);
              auto msb_p_ours = OpenRef(msbx_ours_s, ours);
              if (lctx.get()->Rank() == 0) printResult(msb_p_spu.data(), "msb, spu");
              if (lctx.get()->Rank() == 0) printResult(msb_p_ours.data(), "msb, ours");
              if (lctx.get()->Rank() == 0) checkOutput(msb_p_spu.data(), msb_p_ours.data(), "msb");
            }

            // /**
            //  * ----------------------------------------------
            //  *                Test: eqz.
            //  * ----------------------------------------------
            //  */
            // {
            //   auto x_ari_aby3 = p2a(sctx_aby3.get(), x_p_aby3);
            //   auto x_ari_ours = p2a(sctx_ours.get(), x_p_ours);

            //   size_t comm = GetComm(aby3).comm;
            //   size_t latency = GetComm(aby3).latency;
            //   auto eqz_start = std::chrono::high_resolution_clock::now();

            //   auto x_b = equal_ss(sctx_aby3.get(), x_ari_aby3, x_ari_aby3);

            //   auto eqz_spu = std::chrono::high_resolution_clock::now();
            //   auto duration_spu = std::chrono::duration_cast<std::chrono::microseconds>(eqz_spu - eqz_start);
            //   if (lctx.get()->Rank() == 0) std::cout << "------------------------ eqz, spu" << std::endl;
            //   if (lctx.get()->Rank() == 0) std::cout << "eqz micro seconds: " << duration_spu.count() << std::endl;
            //   if (lctx.get()->Rank() == 0) std::cout << "eqz sent: " << GetComm(aby3).comm - comm << std::endl;
            //   if (lctx.get()->Rank() == 0) std::cout << "eqz latency: " << GetComm(aby3).latency - latency << std::endl;

            //   // ---------------------------------------------------------------          

            //   comm = GetComm(ours).comm;
            //   latency = GetComm(ours).latency;

            //   auto x_b_ours = equal_ss(sctx_ours.get(), x_ari_ours, x_ari_ours);

            //   auto eqz_ours = std::chrono::high_resolution_clock::now();
            //   auto duration_ours = std::chrono::duration_cast<std::chrono::microseconds>(eqz_ours - eqz_spu);
            //   if (lctx.get()->Rank() == 0) std::cout << "------------------------ eqz, ours" << std::endl;
            //   if (lctx.get()->Rank() == 0) std::cout << "eqz micro seconds: " << duration_ours.count() << std::endl;
            //   if (lctx.get()->Rank() == 0) std::cout << "eqz sent: " << GetComm(ours).comm - comm << std::endl;
            //   if (lctx.get()->Rank() == 0) std::cout << "eqz latency: " << GetComm(ours).latency - latency << std::endl;

            //   // Check output.
            //   auto x_p_spu = OpenRef(MyUnwrapValue(x_b.value()), aby3);
            //   auto x_p_ours = OpenRef(MyUnwrapValue(x_b_ours.value()), ours);
            //   // if (lctx.get()->Rank() == 0) printResult(msbx_p.data(), "msb, spu");
            //   // if (lctx.get()->Rank() == 0) printResult(msbx_ours_p.data(), "msb, ours");
            //   if (lctx.get()->Rank() == 0) checkOutput(x_p_spu.data(), x_p_ours.data(), "eqz");
            // }

            /**
             * ----------------------------------------------
             *                Test: B2A.
             * ----------------------------------------------
             */
            {
              size_t comm = GetComm(aby3).comm;
              size_t latency = GetComm(aby3).latency;
              auto b2a_start = std::chrono::high_resolution_clock::now();

              auto x_a = b2a(sctx_aby3.get(), x_s_aby3);

              auto b2a_spu = std::chrono::high_resolution_clock::now();
              auto duration_spu = std::chrono::duration_cast<std::chrono::microseconds>(b2a_spu - b2a_start);
              if (lctx.get()->Rank() == 0) std::cout << "------------------------ B2A, spu" << std::endl;
              if (lctx.get()->Rank() == 0) std::cout << "b2a micro seconds: " << duration_spu.count() << std::endl;
              if (lctx.get()->Rank() == 0) std::cout << "b2a sent: " << GetComm(aby3).comm - comm << std::endl;
              if (lctx.get()->Rank() == 0) std::cout << "b2a latency: " << GetComm(aby3).latency - latency << std::endl;

              // ---------------------------------------------------------------          

              comm = GetComm(ours).comm;
              latency = GetComm(ours).latency;

              auto x_a_ours = oursnp::B2AMultiFanIn(&kectx_ours, MyUnwrapValue(x_s_ours));

              auto b2a_ours = std::chrono::high_resolution_clock::now();
              auto duration_ours = std::chrono::duration_cast<std::chrono::microseconds>(b2a_ours - b2a_spu);
              if (lctx.get()->Rank() == 0) std::cout << "------------------------ B2A, ours" << std::endl;
              if (lctx.get()->Rank() == 0) std::cout << "b2a micro seconds: " << duration_ours.count() << std::endl;
              if (lctx.get()->Rank() == 0) std::cout << "b2a sent: " << GetComm(ours).comm - comm << std::endl;
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
              auto x_ari_aby3 = p2a(sctx_aby3.get(), x_p_aby3);
              auto x_ari_ours = p2a(sctx_ours.get(), x_p_ours);

              size_t comm = GetComm(aby3).comm;
              size_t latency = GetComm(aby3).latency;
              auto a2b_start = std::chrono::high_resolution_clock::now();

              auto x_b = a2b(sctx_aby3.get(), x_ari_aby3);

              auto a2b_spu = std::chrono::high_resolution_clock::now();
              auto duration_spu = std::chrono::duration_cast<std::chrono::microseconds>(a2b_spu - a2b_start);
              if (lctx.get()->Rank() == 0) std::cout << "------------------------ A2B, spu" << std::endl;
              if (lctx.get()->Rank() == 0) std::cout << "a2b micro seconds: " << duration_spu.count() << std::endl;
              if (lctx.get()->Rank() == 0) std::cout << "a2b sent: " << GetComm(aby3).comm - comm << std::endl;
              if (lctx.get()->Rank() == 0) std::cout << "a2b latency: " << GetComm(aby3).latency - latency << std::endl;

              // ---------------------------------------------------------------          

              comm = GetComm(ours).comm;
              latency = GetComm(ours).latency;

              auto x_b_ours = oursnp::A2BMultiFanIn(&kectx_ours, MyUnwrapValue(x_ari_ours));

              auto a2b_ours = std::chrono::high_resolution_clock::now();
              auto duration_ours = std::chrono::duration_cast<std::chrono::microseconds>(a2b_ours - a2b_spu);
              if (lctx.get()->Rank() == 0) std::cout << "------------------------ A2B, ours" << std::endl;
              if (lctx.get()->Rank() == 0) std::cout << "a2b micro seconds: " << duration_ours.count() << std::endl;
              if (lctx.get()->Rank() == 0) std::cout << "a2b sent: " << GetComm(ours).comm - comm << std::endl;
              if (lctx.get()->Rank() == 0) std::cout << "a2b latency: " << GetComm(ours).latency - latency << std::endl;

              // Check output.
              auto x_p_spu = OpenRef(MyUnwrapValue(x_b), aby3);
              auto x_p_ours = OpenRef(x_b_ours, ours);
              // if (lctx.get()->Rank() == 0) printResult(msbx_p.data(), "msb, spu");
              // if (lctx.get()->Rank() == 0) printResult(msbx_ours_p.data(), "msb, ours");
              if (lctx.get()->Rank() == 0) checkOutput(x_p_spu.data(), x_p_ours.data(), "a2b");
            }

            // // 64-fan-in
            // {
            //   auto input00 = p2b(sctx.get(), rand_p(sctx.get(), {M * 32, N})); // 64 -> 32
            //   auto input01 = p2b(sctx.get(), rand_p(sctx.get(), {M * 32, N})); // 64 -> 32
            //   auto input10 = p2b(sctx.get(), rand_p(sctx.get(), {M * 16, N})); // 32 -> 16
            //   auto input11 = p2b(sctx.get(), rand_p(sctx.get(), {M * 16, N})); // 32 -> 16
            //   auto input12 = p2b(sctx.get(), rand_p(sctx.get(), {M * 16, N})); // 32 -> 16
            //   auto input13 = p2b(sctx.get(), rand_p(sctx.get(), {M * 16, N})); // 32 -> 16
            //   auto input20 = p2b(sctx.get(), rand_p(sctx.get(), {M * 8, N} )); // 16 -> 8
            //   auto input21 = p2b(sctx.get(), rand_p(sctx.get(), {M * 8, N} )); // 16 -> 8
            //   auto input30 = p2b(sctx.get(), rand_p(sctx.get(), {M * 4, N} )); // 8 -> 4
            //   auto input31 = p2b(sctx.get(), rand_p(sctx.get(), {M * 4, N} )); // 8 -> 4
            //   auto input32 = p2b(sctx.get(), rand_p(sctx.get(), {M * 4, N} )); // 8 -> 4
            //   auto input33 = p2b(sctx.get(), rand_p(sctx.get(), {M * 4, N} )); // 8 -> 4
            //   auto input40 = p2b(sctx.get(), rand_p(sctx.get(), {M * 2, N} )); // 4 -> 2
            //   auto input41 = p2b(sctx.get(), rand_p(sctx.get(), {M * 2, N} )); // 4 -> 2
            //   auto input50 = p2b(sctx.get(), rand_p(sctx.get(), {M, N}     )); // 2 -> 1
            //   auto input51 = p2b(sctx.get(), rand_p(sctx.get(), {M, N}     )); // 2 -> 1
            //   auto input52 = p2b(sctx.get(), rand_p(sctx.get(), {M, N}     )); // 2 -> 1
            //   auto input53 = p2b(sctx.get(), rand_p(sctx.get(), {M, N}     )); // 2 -> 1

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
            //   auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            //   if (lctx.get()->Rank() == 0) std::cout << "------------------------ 64-fan-in, spu" << std::endl;
            //   if (lctx.get()->Rank() == 0) std::cout << "64-fan-in micro seconds: " << duration.count() << std::endl;
            //   if (lctx.get()->Rank() == 0) std::cout << "64-fan-in sent: " << GetComm.comm - comm << std::endl;
            //   if (lctx.get()->Rank() == 0) std::cout << "64-fan-in latency: " << GetComm.latency - latency << std::endl;

            //   // ours naive and for 64 input
            //   auto input10_mss = oursnp::ResharingRss2Mss(&kectx, MyUnwrapValue(input10));
            //   auto input11_mss = oursnp::ResharingRss2Mss(&kectx, MyUnwrapValue(input11));
            //   auto input12_mss = oursnp::ResharingRss2Mss(&kectx, MyUnwrapValue(input12));
            //   auto input13_mss = oursnp::ResharingRss2Mss(&kectx, MyUnwrapValue(input13));
            //   auto input30_mss = oursnp::ResharingRss2Mss(&kectx, MyUnwrapValue(input30));
            //   auto input31_mss = oursnp::ResharingRss2Mss(&kectx, MyUnwrapValue(input31));
            //   auto input32_mss = oursnp::ResharingRss2Mss(&kectx, MyUnwrapValue(input32));
            //   auto input33_mss = oursnp::ResharingRss2Mss(&kectx, MyUnwrapValue(input33));
            //   auto input50_mss = oursnp::ResharingRss2Mss(&kectx, MyUnwrapValue(input50));
            //   auto input51_mss = oursnp::ResharingRss2Mss(&kectx, MyUnwrapValue(input51));
            //   auto input52_mss = oursnp::ResharingRss2Mss(&kectx, MyUnwrapValue(input52));
            //   auto input53_mss = oursnp::ResharingRss2Mss(&kectx, MyUnwrapValue(input53));

            //   start = std::chrono::high_resolution_clock::now();
            //   comm = GetComm.comm;
            //   latency = GetComm.latency;
            //   auto result1_ours = oursnp::MssAnd4NoComm(&kectx, input10_mss, input11_mss, input12_mss, input13_mss);
            //   auto result1_mss = oursnp::ResharingAss2Mss(&kectx, result1_ours);
            //   auto result3_ours = oursnp::MssAnd4NoComm(&kectx, input30_mss, input31_mss, input32_mss, input33_mss);
            //   auto result3_mss = oursnp::ResharingAss2Mss(&kectx, result3_ours);
            //   auto result5_ours = oursnp::MssAnd4NoComm(&kectx, input50_mss, input51_mss, input52_mss, input53_mss);
            //   auto result5_mss = oursnp::ResharingAss2Mss(&kectx, result5_ours);
            //   auto end_ours = std::chrono::high_resolution_clock::now();
            //   auto duration_ours = std::chrono::duration_cast<std::chrono::microseconds>(end_ours - start);
            //   if (lctx.get()->Rank() == 0) std::cout << "------------------------ 64-fan-in, ours" << std::endl;
            //   if (lctx.get()->Rank() == 0) std::cout << "64-fan-in micro seconds: " << duration_ours.count() << std::endl;
            //   if (lctx.get()->Rank() == 0) std::cout << "64-fan-in sent: " << GetComm.comm - comm << std::endl;
            //   if (lctx.get()->Rank() == 0) std::cout << "64-fan-in latency: " << GetComm.latency - latency << std::endl;
            //   // oursnp::MssAnd4NoComm(&kectx, x_mss, y_mss, a_mss, b_mss);

            // }
        });    
}