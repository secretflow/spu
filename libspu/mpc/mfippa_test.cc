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
#include "libspu/mpc/alkaid/value.h"
#include "libspu/mpc/alkaid/type.h"
#include "libspu/mpc/aby3/protocol.h"
#include "libspu/mpc/aby3/conversion.h"
#include "libspu/mpc/utils/simulate.h"

#include "libspu/mpc/common/communicator.h"

// hack wrap value for handle namespace error.
#define MyWrapValue(x) spu::Value(x, spu::DT_INVALID)
#define MyUnwrapValue(x) x.data()
#define NumOutputs 3
#define OpenRef(ndref, tag) b2p(sctx_##tag.get(), MyWrapValue(ndref))
#define N 32
#define M 32
#define GetComm(tag) sctx_##tag.get()->getState<Communicator>()->getStats()
#define RANDOM_INPUT 1

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
             * ----------------------------------------------
             *                Syetem setup.
             * ----------------------------------------------
             */
            auto field = spu::FieldType::FM64;        
            spu::RuntimeConfig config_al = makeConfig(field, true);
            auto sctx_al = makeAlkaidProtocol(config_al, lctx); 
            auto kectx_al = spu::KernelEvalContext(sctx_al.get());    
            spu::RuntimeConfig config_aby = makeConfig(field, false);
            auto sctx_aby = makeAby3Protocol(config_aby, lctx); 
            auto kectx_aby = spu::KernelEvalContext(sctx_aby.get());   

            /**
             * ----------------------------------------------
             *                Input random x, y.
             * ----------------------------------------------
             */
            #ifdef RANDOM_INPUT
            auto x_p = rand_p(sctx_aby.get(), kShape);
            auto y_p = rand_p(sctx_aby.get(), kShape);
            #else
            auto x_p = make_p(sctx_al.get(), static_cast<uint128_t>(16813041318660192066ull), kShape);
            auto y_p = make_p(sctx_al.get(), static_cast<uint128_t>(4), kShape);
            #endif
            auto x_s = p2b(sctx_aby.get(), x_p);
            auto y_s = p2b(sctx_aby.get(), y_p);
            if (lctx.get()->Rank() == 0) printResult(x_p.data(), "input x");
            if (lctx.get()->Rank() == 0) printResult(y_p.data(), "input y");

            /**
             * ----------------------------------------------
             *                Test: ExtMsb.
             * ----------------------------------------------
             */
            {

              auto x_ari = b2a(sctx_aby.get(), x_s);
              auto x_ari_al = x_ari.data().clone();
              x_ari_al.eltype() = spu::makeType<alkaid::AShrTy>(field);

              size_t comm = GetComm(aby).comm;
              size_t latency = GetComm(aby).latency;
              auto msb_start = std::chrono::high_resolution_clock::now();

              auto msbx_s = msb_a2b(sctx_aby.get(), x_ari);

              auto msb_spu = std::chrono::high_resolution_clock::now();
              auto duration_spu = std::chrono::duration_cast<std::chrono::microseconds>(msb_spu - msb_start);
              if (lctx.get()->Rank() == 0) std::cout << "------------------------ MSB, spu" << std::endl;
              if (lctx.get()->Rank() == 0) std::cout << "msb micro seconds: " << duration_spu.count() << std::endl;
              if (lctx.get()->Rank() == 0) std::cout << "msb sent: " << GetComm(aby).comm - comm << std::endl;
              if (lctx.get()->Rank() == 0) std::cout << "msb latency: " << GetComm(aby).latency - latency << std::endl;

              // ---------------------------------------------------------------              
              
              comm = GetComm(al).comm;
              latency = GetComm(al).latency;

              auto msbx_ours_s = alkaid::MsbA2BMultiFanIn(&kectx_al, x_ari_al);

              auto msb_ours = std::chrono::high_resolution_clock::now();
              auto duration_ours = std::chrono::duration_cast<std::chrono::microseconds>(msb_ours - msb_spu);
              if (lctx.get()->Rank() == 0) std::cout << "------------------------ MSB, ours" << std::endl;
              if (lctx.get()->Rank() == 0) std::cout << "msb micro seconds: " << duration_ours.count() << std::endl;
              if (lctx.get()->Rank() == 0) std::cout << "msb sent: " << GetComm(al).comm - comm << std::endl;
              if (lctx.get()->Rank() == 0) std::cout << "msb latency: " << GetComm(al).latency - latency << std::endl;

              // Check output.
              auto msb_p_spu = OpenRef(MyUnwrapValue(msbx_s), aby);
              auto msb_p_ours = OpenRef(msbx_ours_s, al);
              if (lctx.get()->Rank() == 0) printResult(msb_p_spu.data(), "msb, spu");
              if (lctx.get()->Rank() == 0) printResult(msb_p_ours.data(), "msb, ours");
              if (lctx.get()->Rank() == 0) checkOutput(msb_p_spu.data(), msb_p_ours.data(), "msb");
            }

            /**
             * ----------------------------------------------
             *                Test: A2B.
             * ----------------------------------------------
             */
            {
              auto x_ari = p2a(sctx_aby.get(), x_p);
              auto x_ari_al = x_ari;
              x_ari_al.storage_type() = spu::makeType<alkaid::AShrTy>(field);

              size_t comm = GetComm(aby).comm;
              size_t latency = GetComm(aby).latency;
              auto a2b_start = std::chrono::high_resolution_clock::now();

              auto x_b = a2b(sctx_aby.get(), x_ari);

              auto a2b_spu = std::chrono::high_resolution_clock::now();
              auto duration_spu = std::chrono::duration_cast<std::chrono::microseconds>(a2b_spu - a2b_start);
              if (lctx.get()->Rank() == 0) std::cout << "------------------------ A2B, spu" << std::endl;
              if (lctx.get()->Rank() == 0) std::cout << "a2b micro seconds: " << duration_spu.count() << std::endl;
              if (lctx.get()->Rank() == 0) std::cout << "a2b sent: " << GetComm(aby).comm - comm << std::endl;
              if (lctx.get()->Rank() == 0) std::cout << "a2b latency: " << GetComm(aby).latency - latency << std::endl;

              // ---------------------------------------------------------------          

              comm = GetComm(al).comm;
              latency = GetComm(al).latency;

              auto x_b_ours = alkaid::A2BMultiFanIn(&kectx_al, MyUnwrapValue(x_ari_al));

              auto a2b_ours = std::chrono::high_resolution_clock::now();
              auto duration_ours = std::chrono::duration_cast<std::chrono::microseconds>(a2b_ours - a2b_spu);
              if (lctx.get()->Rank() == 0) std::cout << "------------------------ A2B, ours" << std::endl;
              if (lctx.get()->Rank() == 0) std::cout << "a2b micro seconds: " << duration_ours.count() << std::endl;
              if (lctx.get()->Rank() == 0) std::cout << "a2b sent: " << GetComm(al).comm - comm << std::endl;
              if (lctx.get()->Rank() == 0) std::cout << "a2b latency: " << GetComm(al).latency - latency << std::endl;

              // Check output.
              auto x_p_spu = OpenRef(MyUnwrapValue(x_b), aby);
              auto x_p_ours = OpenRef(x_b_ours, al);
              // if (lctx.get()->Rank() == 0) printResult(msbx_p.data(), "msb, spu");
              // if (lctx.get()->Rank() == 0) printResult(msbx_ours_p.data(), "msb, ours");
              if (lctx.get()->Rank() == 0) checkOutput(x_p_spu.data(), x_p_ours.data(), "a2b");
            }

            /**
             * ----------------------------------------------
             *                Test: eqz.
             * ----------------------------------------------
             */
            {
              auto x_ari = p2a(sctx_aby.get(), x_p);
              auto x_ari_al = x_ari;
              x_ari_al.storage_type() = spu::makeType<alkaid::AShrTy>(field);

              size_t comm = GetComm(aby).comm;
              size_t latency = GetComm(aby).latency;
              auto eqz_start = std::chrono::high_resolution_clock::now();

              auto x_b = equal_ss(sctx_aby.get(), x_ari, x_ari);

              auto eqz_spu = std::chrono::high_resolution_clock::now();
              auto duration_spu = std::chrono::duration_cast<std::chrono::microseconds>(eqz_spu - eqz_start);
              if (lctx.get()->Rank() == 0) std::cout << "------------------------ eqz, spu" << std::endl;
              if (lctx.get()->Rank() == 0) std::cout << "eqz micro seconds: " << duration_spu.count() << std::endl;
              if (lctx.get()->Rank() == 0) std::cout << "eqz sent: " << GetComm(aby).comm - comm << std::endl;
              if (lctx.get()->Rank() == 0) std::cout << "eqz latency: " << GetComm(aby).latency - latency << std::endl;

              // ---------------------------------------------------------------          

              comm = GetComm(al).comm;
              latency = GetComm(al).latency;

              auto x_b_ours = equal_ss(sctx_al.get(), x_ari_al, x_ari_al);

              auto eqz_ours = std::chrono::high_resolution_clock::now();
              auto duration_ours = std::chrono::duration_cast<std::chrono::microseconds>(eqz_ours - eqz_spu);
              if (lctx.get()->Rank() == 0) std::cout << "------------------------ eqz, ours" << std::endl;
              if (lctx.get()->Rank() == 0) std::cout << "eqz micro seconds: " << duration_ours.count() << std::endl;
              if (lctx.get()->Rank() == 0) std::cout << "eqz sent: " << GetComm(al).comm - comm << std::endl;
              if (lctx.get()->Rank() == 0) std::cout << "eqz latency: " << GetComm(al).latency - latency << std::endl;

              // Check output.
              auto x_p_spu = OpenRef(MyUnwrapValue(x_b.value()), aby);
              auto x_p_ours = OpenRef(MyUnwrapValue(x_b_ours.value()), al);
              // if (lctx.get()->Rank() == 0) printResult(msbx_p.data(), "msb, spu");
              // if (lctx.get()->Rank() == 0) printResult(msbx_ours_p.data(), "msb, ours");
              if (lctx.get()->Rank() == 0) checkOutput(x_p_spu.data(), x_p_ours.data(), "eqz");
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
            //   auto input10_mss = alkaid::ResharingRss2Mss(&kectx, MyUnwrapValue(input10));
            //   auto input11_mss = alkaid::ResharingRss2Mss(&kectx, MyUnwrapValue(input11));
            //   auto input12_mss = alkaid::ResharingRss2Mss(&kectx, MyUnwrapValue(input12));
            //   auto input13_mss = alkaid::ResharingRss2Mss(&kectx, MyUnwrapValue(input13));
            //   auto input30_mss = alkaid::ResharingRss2Mss(&kectx, MyUnwrapValue(input30));
            //   auto input31_mss = alkaid::ResharingRss2Mss(&kectx, MyUnwrapValue(input31));
            //   auto input32_mss = alkaid::ResharingRss2Mss(&kectx, MyUnwrapValue(input32));
            //   auto input33_mss = alkaid::ResharingRss2Mss(&kectx, MyUnwrapValue(input33));
            //   auto input50_mss = alkaid::ResharingRss2Mss(&kectx, MyUnwrapValue(input50));
            //   auto input51_mss = alkaid::ResharingRss2Mss(&kectx, MyUnwrapValue(input51));
            //   auto input52_mss = alkaid::ResharingRss2Mss(&kectx, MyUnwrapValue(input52));
            //   auto input53_mss = alkaid::ResharingRss2Mss(&kectx, MyUnwrapValue(input53));

            //   start = std::chrono::high_resolution_clock::now();
            //   comm = GetComm.comm;
            //   latency = GetComm.latency;
            //   auto result1_ours = alkaid::MssAnd4NoComm(&kectx, input10_mss, input11_mss, input12_mss, input13_mss);
            //   auto result1_mss = alkaid::ResharingAss2Mss(&kectx, result1_ours);
            //   auto result3_ours = alkaid::MssAnd4NoComm(&kectx, input30_mss, input31_mss, input32_mss, input33_mss);
            //   auto result3_mss = alkaid::ResharingAss2Mss(&kectx, result3_ours);
            //   auto result5_ours = alkaid::MssAnd4NoComm(&kectx, input50_mss, input51_mss, input52_mss, input53_mss);
            //   auto result5_mss = alkaid::ResharingAss2Mss(&kectx, result5_ours);
            //   auto end_ours = std::chrono::high_resolution_clock::now();
            //   auto duration_ours = std::chrono::duration_cast<std::chrono::microseconds>(end_ours - start);
            //   if (lctx.get()->Rank() == 0) std::cout << "------------------------ 64-fan-in, ours" << std::endl;
            //   if (lctx.get()->Rank() == 0) std::cout << "64-fan-in micro seconds: " << duration_ours.count() << std::endl;
            //   if (lctx.get()->Rank() == 0) std::cout << "64-fan-in sent: " << GetComm.comm - comm << std::endl;
            //   if (lctx.get()->Rank() == 0) std::cout << "64-fan-in latency: " << GetComm.latency - latency << std::endl;
            //   // alkaid::MssAnd4NoComm(&kectx, x_mss, y_mss, a_mss, b_mss);

            // }
        });    
}