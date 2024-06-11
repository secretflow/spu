#include <iostream>
#include <string>

#include "libspu/spu.pb.h"
#include "libspu/core/ndarray_ref.h"

#include "libspu/mpc/api.h"
#include "libspu/mpc/ab_api.h"
#include "libspu/mpc/aby3/protocol.h"
#include "libspu/mpc/aby3/conversion.h"
#include "libspu/mpc/utils/simulate.h"

// hack wrap value for handle namespace error.
#define MyWrapValue(x) spu::Value(x, spu::DT_INVALID)
#define MyUnwrapValue(x) x.data()
#define NumOutputs 3
#define OpenRef(ndref) b2p(sctx.get(), MyWrapValue(ndref))
#define N 20
#define M 30

using ring2k_t = uint64_t;
using pub_t = std::array<ring2k_t, 1>; 
using namespace spu::mpc;

spu::RuntimeConfig makeConfig(spu::FieldType field) {
  spu::RuntimeConfig conf;
  conf.set_protocol(spu::ProtocolKind::ABY3);
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
            auto field = spu::FieldType::FM64;        
            spu::RuntimeConfig config = makeConfig(field);
            auto sctx = makeAby3Protocol(config, lctx); 
            auto kectx = spu::KernelEvalContext(sctx.get());        

            // sharing test
            auto x_p = rand_p(sctx.get(), kShape);
            auto y_p = rand_p(sctx.get(), kShape);
            // auto x_p = make_p(sctx.get(), static_cast<uint128_t>(2), kShape);
            // auto y_p = make_p(sctx.get(), static_cast<uint128_t>(4), kShape);
            auto a_p = make_p(sctx.get(), static_cast<uint128_t>(6), kShape);
            auto b_p = make_p(sctx.get(), static_cast<uint128_t>(15), kShape);
            auto x_s = p2b(sctx.get(), x_p);
            auto y_s = p2b(sctx.get(), y_p);
            auto a_s = p2b(sctx.get(), a_p);
            auto b_s = p2b(sctx.get(), b_p);
            auto x_mss = aby3::ResharingRss2Mss(&kectx, MyUnwrapValue(x_s));
            auto y_mss = aby3::ResharingRss2Mss(&kectx, MyUnwrapValue(y_s));

            if (lctx.get()->Rank() == 0) printResult(x_p.data(), "input x");
            if (lctx.get()->Rank() == 0) printResult(y_p.data(), "input y");

            // and test
            {
              auto z_s = and_bb(sctx.get(), x_s, y_s);
              auto z_p = b2p(sctx.get(), z_s);
              auto zours_ass = aby3::RssAnd2NoComm(&kectx, MyUnwrapValue(x_s), MyUnwrapValue(y_s));
              auto zours_rss = aby3::ResharingAss2Rss(&kectx, zours_ass);
              auto zours_p = b2p(sctx.get(), MyWrapValue(zours_rss));
              
              auto xy_rss = aby3::MssAnd2NoComm(&kectx, x_mss, y_mss);
              auto xy_p = b2p(sctx.get(), MyWrapValue(xy_rss));

              if (lctx.get()->Rank() == 0) printResult(z_p.data(), "x & y, spu");
              if (lctx.get()->Rank() == 0) printResult(zours_p.data(), "x & y, RssAnd2NoComm");
              if (lctx.get()->Rank() == 0) printResult(xy_p.data(), "x & y, MssAnd2Comm");

              if (lctx.get()->Rank() == 0) checkOutput(z_p.data(), zours_p.data(), "and2-rss-based");
              if (lctx.get()->Rank() == 0) checkOutput(z_p.data(), xy_p.data(), "and2-mss-based");
            }

            // mutli-fan-in and test
            {
              auto res_lo_s = and_bb(sctx.get(), x_s, y_s);
              auto res_hi_s = and_bb(sctx.get(), a_s, b_s);
              auto res_s = and_bb(sctx.get(), res_lo_s, a_s);
              auto res_p = b2p(sctx.get(), res_s);
              auto a_mss = aby3::ResharingRss2Mss(&kectx, MyUnwrapValue(a_s));
              auto b_mss = aby3::ResharingRss2Mss(&kectx, MyUnwrapValue(b_s));
              auto res_ass = aby3::MssAnd4NoComm(&kectx, x_mss, y_mss, a_mss, b_mss);
              auto res_rss = aby3::ResharingAss2Rss(&kectx, res_ass);
              auto res_ours_p = b2p(sctx.get(), MyWrapValue(res_rss));
              // auto xy_rss = aby3::MssAnd2NoComm(&kectx, x_mss, y_mss);
              // auto xy_rss_p = OpenRef(xy_rss);
              // auto a_rss = aby3::ResharingMss2Rss(&kectx, a_mss);
              // auto a_rss_p = OpenRef(a_rss);
              // auto xya_ass = aby3::RssAnd2NoComm(&kectx, xy_rss, a_rss);
              // auto xya_rss = aby3::ResharingAss2Rss(&kectx, xya_ass);
              // auto xya_rss_p = b2p(sctx.get(), MyWrapValue(xya_rss));

              if (lctx.get()->Rank() == 0) printResult(res_p.data(), "and4, spu");
              if (lctx.get()->Rank() == 0) printResult(res_ours_p.data(), "and4, ours");
              // if (lctx.get()->Rank() == 0) printResult(xy_rss_p.data(), "and4, step1 ");
              // if (lctx.get()->Rank() == 0) printResult(a_rss_p.data(), "and4, step2 ");
              // if (lctx.get()->Rank() == 0) printResult(xya_rss_p.data(), "and4, step3 ");

              if (lctx.get()->Rank() == 0) checkOutput(res_p.data(), res_ours_p.data(), "and4");
            }

            // resharing test
            {
              auto z_s = and_bb(sctx.get(), x_s, y_s);
              auto z_p = b2p(sctx.get(), z_s);
              auto zours_ass = aby3::RssAnd2NoComm(&kectx, MyUnwrapValue(x_s), MyUnwrapValue(y_s));
              auto zours_rss_2 = aby3::ResharingAss2Rss(&kectx, zours_ass);
              auto zours_mss_2 = aby3::ResharingRss2Mss(&kectx, zours_rss_2);
              auto zours_rss_3 = aby3::ResharingMss2Rss(&kectx, zours_mss_2);
              auto zours_ass_3 = aby3::ResharingRss2Ass(&kectx, zours_rss_3);
              auto zours_mss_3 = aby3::ResharingAss2Mss(&kectx, zours_ass_3);
              auto zours_rss_4 = aby3::ResharingMss2Rss(&kectx, zours_mss_3);
              auto zours_p2 = b2p(sctx.get(), MyWrapValue(zours_rss_4));

              if (lctx.get()->Rank() == 0) printResult(zours_p2.data(), "resharing");
              if (lctx.get()->Rank() == 0) checkOutput(z_p.data(), zours_p2.data(), "resharing");
            }

            // msb test
            {
              auto x_as = b2a(sctx.get(), x_s);
              auto msbx_s = msb_a2b(sctx.get(), x_as);
              auto msbx_p = b2p(sctx.get(), msbx_s);

              if (lctx.get()->Rank() == 0) printResult(msbx_p.data(), "msb, spu");
              
              auto msbx_ours_s = aby3::MsbA2BMultiFanIn(&kectx, MyUnwrapValue(x_as));
              auto msbx_ours_p = OpenRef(msbx_ours_s);

              if (lctx.get()->Rank() == 0) printResult(msbx_ours_p.data(), "msb, ours");

              // auto x_ap = a2p(sctx.get(), x_as);

              spu::NdArrayView<pub_t> view_msbx_p(msbx_p.data());
              spu::NdArrayView<pub_t> view_msbx_ours_p(msbx_ours_p.data());

              size_t match_count = 0;
              size_t positive_count = 0;
              size_t ours_positive_count = 0;
              for (int i = 0; i < N * M; i++) 
              {
                if (view_msbx_p[i][0] == view_msbx_ours_p[i][0]) match_count++;
                if (view_msbx_p[i][0] == 0) positive_count++;
                if (view_msbx_ours_p[i][0] == 0) ours_positive_count++;
              }
              if (lctx.get()->Rank() == 0) std::cout << "msb match count rate: " << static_cast<double>(match_count) / (N * M) << std::endl;
              if (lctx.get()->Rank() == 0) std::cout << "msb positive count rate: " << static_cast<double>(positive_count) / (N * M) << std::endl;
              if (lctx.get()->Rank() == 0) std::cout << "ours msb positive count rate: " << static_cast<double>(ours_positive_count) / (N * M) << std::endl;
            }

        });    
}