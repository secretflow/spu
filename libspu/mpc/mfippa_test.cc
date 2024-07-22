// #define EQ_DO_NOT_USE_OFFLINE

#include <iostream>
#include <string>
#include <chrono>

#include "libspu/spu.pb.h"
#include "libspu/core/ndarray_ref.h"

#include "libspu/mpc/api.h"
#include "libspu/mpc/ab_api.h"
#include "libspu/mpc/alkaid/protocol.h"
#include "libspu/mpc/alkaid/conversion.h"
#include "libspu/mpc/aby3/protocol.h"
#include "libspu/mpc/aby3/conversion.h"
#include "libspu/mpc/utils/simulate.h"

#include "libspu/mpc/common/communicator.h"

// hack wrap value for handle namespace error.
#define MyWrapValue(x) spu::Value(x, spu::DT_INVALID)
#define MyUnwrapValue(x) x.data()
#define NumOutputs 3
#define OpenRef(ndref) b2p(sctx.get(), MyWrapValue(ndref))
#define N 32
#define M 32
#define GetComm sctx.get()->getState<Communicator>()->getStats()
#define USE_ALKAID 1

using ring2k_t = uint64_t;
using pub_t = std::array<ring2k_t, 1>; 
using namespace spu::mpc;

spu::RuntimeConfig makeConfig(spu::FieldType field, bool use_alkaid=true) {
  spu::RuntimeConfig conf;
  if (use_alkaid) conf.set_protocol(spu::ProtocolKind::ALKAID);
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
             * ----------------------------------------------
             *                Sharing test.
             * ----------------------------------------------
             */
            auto field = spu::FieldType::FM64;        
            #ifdef USE_ALKAID
            spu::RuntimeConfig config = makeConfig(field, true);
            auto sctx = makeAlkaidProtocol(config, lctx); 
            auto kectx = spu::KernelEvalContext(sctx.get());    
            #else
            spu::RuntimeConfig config = makeConfig(field, false);
            auto sctx = makeAby3Protocol(config, lctx); 
            auto kectx = spu::KernelEvalContext(sctx.get());   
            #endif

            auto x_p = rand_p(sctx.get(), kShape);
            auto y_p = rand_p(sctx.get(), kShape);
            // auto x_p = make_p(sctx.get(), static_cast<uint128_t>(2), kShape);
            // auto y_p = make_p(sctx.get(), static_cast<uint128_t>(4), kShape);
            auto x_s = p2b(sctx.get(), x_p);
            auto y_s = p2b(sctx.get(), y_p);

            if (lctx.get()->Rank() == 0) printResult(x_p.data(), "input x");
            if (lctx.get()->Rank() == 0) printResult(y_p.data(), "input y");

            // msb test
            {
              auto x_as = b2a(sctx.get(), x_s);
              size_t comm = GetComm.comm;
              size_t latency = GetComm.latency;
              auto msb_start = std::chrono::high_resolution_clock::now();
              auto msbx_s = msb_a2b(sctx.get(), x_as);
              auto msb_spu = std::chrono::high_resolution_clock::now();
              auto duration_spu = std::chrono::duration_cast<std::chrono::microseconds>(msb_spu - msb_start);
              // auto msbx_p = b2p(sctx.get(), msbx_s);

              if (lctx.get()->Rank() == 0) std::cout << "------------------------ MSB, spu" << std::endl;
              // if (lctx.get()->Rank() == 0) printResult(msbx_p.data(), "msb, spu");
              if (lctx.get()->Rank() == 0) std::cout << "msb micro seconds: " << duration_spu.count() << std::endl;
              if (lctx.get()->Rank() == 0) std::cout << "msb sent: " << GetComm.comm - comm << std::endl;
              if (lctx.get()->Rank() == 0) std::cout << "msb latency: " << GetComm.latency - latency << std::endl;
              
              comm = GetComm.comm;
              latency = GetComm.latency;
              auto msbx_ours_s = alkaid::MsbA2BMultiFanIn(&kectx, MyUnwrapValue(x_as));
              auto msb_ours = std::chrono::high_resolution_clock::now();
              auto duration_ours = std::chrono::duration_cast<std::chrono::microseconds>(msb_ours - msb_spu);
              // auto msbx_ours_p = OpenRef(msbx_ours_s);
              if (lctx.get()->Rank() == 0) std::cout << "------------------------ MSB, ours" << std::endl;
              // if (lctx.get()->Rank() == 0) printResult(msbx_ours_p.data(), "msb, ours");
              if (lctx.get()->Rank() == 0) std::cout << "msb micro seconds: " << duration_ours.count() << std::endl;
              if (lctx.get()->Rank() == 0) std::cout << "msb sent: " << GetComm.comm - comm << std::endl;
              if (lctx.get()->Rank() == 0) std::cout << "msb latency: " << GetComm.latency - latency << std::endl;

              // auto x_ap = a2p(sctx.get(), x_as);

              // spu::NdArrayView<pub_t> view_msbx_p(msbx_p.data());
              // spu::NdArrayView<pub_t> view_msbx_ours_p(msbx_ours_p.data());

              // size_t match_count = 0;
              // size_t positive_count = 0;
              // size_t ours_positive_count = 0;
              // for (int i = 0; i < N * M; i++) 
              // {
              //   if (view_msbx_p[i][0] == view_msbx_ours_p[i][0]) match_count++;
              //   if (view_msbx_p[i][0] == 0) positive_count++;
              //   if (view_msbx_ours_p[i][0] == 0) ours_positive_count++;
              // }
              // if (lctx.get()->Rank() == 0) std::cout << "msb match count rate: " << static_cast<double>(match_count) / (N * M) << std::endl;
              // if (lctx.get()->Rank() == 0) std::cout << "msb positive count rate: " << static_cast<double>(positive_count) / (N * M) << std::endl;
              // if (lctx.get()->Rank() == 0) std::cout << "ours msb positive count rate: " << static_cast<double>(ours_positive_count) / (N * M) << std::endl;
            }

            // A2B test.
            {
              auto x_a = p2a(sctx.get(), x_p);
              auto x_b_spu = a2b(sctx.get(), x_a);
              auto x_r_spu = OpenRef(MyUnwrapValue(x_b_spu));
              auto x_b_ours = alkaid::A2BMultiFanIn(&kectx, MyUnwrapValue(x_a));
              auto x_r_ours = OpenRef(x_b_ours);

              if (lctx.get()->Rank() == 0) printResult(x_r_spu.data(), "a2b, spu");
              if (lctx.get()->Rank() == 0) printResult(x_r_ours.data(), "a2b, ours");
              if (lctx.get()->Rank() == 0) checkOutput(x_r_spu.data(), x_r_ours.data(), "a2b");
            }

            // 64-fan-in
            {
              auto input00 = p2b(sctx.get(), rand_p(sctx.get(), {M * 32, N})); // 64 -> 32
              auto input01 = p2b(sctx.get(), rand_p(sctx.get(), {M * 32, N})); // 64 -> 32
              auto input10 = p2b(sctx.get(), rand_p(sctx.get(), {M * 16, N})); // 32 -> 16
              auto input11 = p2b(sctx.get(), rand_p(sctx.get(), {M * 16, N})); // 32 -> 16
              auto input12 = p2b(sctx.get(), rand_p(sctx.get(), {M * 16, N})); // 32 -> 16
              auto input13 = p2b(sctx.get(), rand_p(sctx.get(), {M * 16, N})); // 32 -> 16
              auto input20 = p2b(sctx.get(), rand_p(sctx.get(), {M * 8, N} )); // 16 -> 8
              auto input21 = p2b(sctx.get(), rand_p(sctx.get(), {M * 8, N} )); // 16 -> 8
              auto input30 = p2b(sctx.get(), rand_p(sctx.get(), {M * 4, N} )); // 8 -> 4
              auto input31 = p2b(sctx.get(), rand_p(sctx.get(), {M * 4, N} )); // 8 -> 4
              auto input32 = p2b(sctx.get(), rand_p(sctx.get(), {M * 4, N} )); // 8 -> 4
              auto input33 = p2b(sctx.get(), rand_p(sctx.get(), {M * 4, N} )); // 8 -> 4
              auto input40 = p2b(sctx.get(), rand_p(sctx.get(), {M * 2, N} )); // 4 -> 2
              auto input41 = p2b(sctx.get(), rand_p(sctx.get(), {M * 2, N} )); // 4 -> 2
              auto input50 = p2b(sctx.get(), rand_p(sctx.get(), {M, N}     )); // 2 -> 1
              auto input51 = p2b(sctx.get(), rand_p(sctx.get(), {M, N}     )); // 2 -> 1
              auto input52 = p2b(sctx.get(), rand_p(sctx.get(), {M, N}     )); // 2 -> 1
              auto input53 = p2b(sctx.get(), rand_p(sctx.get(), {M, N}     )); // 2 -> 1

              // spu naive and for 64 input
              auto start = std::chrono::high_resolution_clock::now();
              size_t comm = GetComm.comm;
              size_t latency = GetComm.latency;
              auto result0 = and_bb(sctx.get(), input00, input01);
              auto result1 = and_bb(sctx.get(), input10, input11);
              auto result2 = and_bb(sctx.get(), input20, input21);
              auto result3 = and_bb(sctx.get(), input30, input31);
              auto result4 = and_bb(sctx.get(), input40, input41);
              auto result5 = and_bb(sctx.get(), input50, input51);
              auto end = std::chrono::high_resolution_clock::now();
              auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
              if (lctx.get()->Rank() == 0) std::cout << "------------------------ 64-fan-in, spu" << std::endl;
              if (lctx.get()->Rank() == 0) std::cout << "64-fan-in micro seconds: " << duration.count() << std::endl;
              if (lctx.get()->Rank() == 0) std::cout << "64-fan-in sent: " << GetComm.comm - comm << std::endl;
              if (lctx.get()->Rank() == 0) std::cout << "64-fan-in latency: " << GetComm.latency - latency << std::endl;

              // ours naive and for 64 input
              auto input10_mss = alkaid::ResharingRss2Mss(&kectx, MyUnwrapValue(input10));
              auto input11_mss = alkaid::ResharingRss2Mss(&kectx, MyUnwrapValue(input11));
              auto input12_mss = alkaid::ResharingRss2Mss(&kectx, MyUnwrapValue(input12));
              auto input13_mss = alkaid::ResharingRss2Mss(&kectx, MyUnwrapValue(input13));
              auto input30_mss = alkaid::ResharingRss2Mss(&kectx, MyUnwrapValue(input30));
              auto input31_mss = alkaid::ResharingRss2Mss(&kectx, MyUnwrapValue(input31));
              auto input32_mss = alkaid::ResharingRss2Mss(&kectx, MyUnwrapValue(input32));
              auto input33_mss = alkaid::ResharingRss2Mss(&kectx, MyUnwrapValue(input33));
              auto input50_mss = alkaid::ResharingRss2Mss(&kectx, MyUnwrapValue(input50));
              auto input51_mss = alkaid::ResharingRss2Mss(&kectx, MyUnwrapValue(input51));
              auto input52_mss = alkaid::ResharingRss2Mss(&kectx, MyUnwrapValue(input52));
              auto input53_mss = alkaid::ResharingRss2Mss(&kectx, MyUnwrapValue(input53));

              start = std::chrono::high_resolution_clock::now();
              comm = GetComm.comm;
              latency = GetComm.latency;
              auto result1_ours = alkaid::MssAnd4NoComm(&kectx, input10_mss, input11_mss, input12_mss, input13_mss);
              auto result1_mss = alkaid::ResharingAss2Mss(&kectx, result1_ours);
              auto result3_ours = alkaid::MssAnd4NoComm(&kectx, input30_mss, input31_mss, input32_mss, input33_mss);
              auto result3_mss = alkaid::ResharingAss2Mss(&kectx, result3_ours);
              auto result5_ours = alkaid::MssAnd4NoComm(&kectx, input50_mss, input51_mss, input52_mss, input53_mss);
              auto result5_mss = alkaid::ResharingAss2Mss(&kectx, result5_ours);
              auto end_ours = std::chrono::high_resolution_clock::now();
              auto duration_ours = std::chrono::duration_cast<std::chrono::microseconds>(end_ours - start);
              if (lctx.get()->Rank() == 0) std::cout << "------------------------ 64-fan-in, ours" << std::endl;
              if (lctx.get()->Rank() == 0) std::cout << "64-fan-in micro seconds: " << duration_ours.count() << std::endl;
              if (lctx.get()->Rank() == 0) std::cout << "64-fan-in sent: " << GetComm.comm - comm << std::endl;
              if (lctx.get()->Rank() == 0) std::cout << "64-fan-in latency: " << GetComm.latency - latency << std::endl;
              // alkaid::MssAnd4NoComm(&kectx, x_mss, y_mss, a_mss, b_mss);

            }
        });    
}