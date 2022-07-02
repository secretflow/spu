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

#include "spu/hal/simulation/fxp_sim.h"

#include "spu/core/xt_helper.h"
#include "spu/hal/constants.h"
#include "spu/hal/context.h"
#include "spu/hal/polymorphic.h"
#include "spu/hal/prot_wrapper.h"
#include "spu/mpc/util/simulate.h"

namespace spu::hal::simulation {

constexpr size_t kNumParties = 3;
constexpr ProtocolKind kProtocol = ProtocolKind::ABY3;
constexpr FieldType kField = FieldType::FM64;

template <typename T>
std::vector<T> toVector(const NdArrayRef& in) {
  YASL_ENFORCE(in.isCompact());

  const PtType pt_ty = in.eltype().as<PtTy>()->pt_type();
  return DISPATCH_ALL_PT_TYPES(pt_ty, "_", [&]() {
    auto in_xt = xt::cast<T>((xt_adapt<_PtTypeT>(in)));
    return std::vector<T>(in_xt.begin(), in_xt.end());
  });
}

RuntimeConfig makeConfig() {
  RuntimeConfig hcfg;
  hcfg.set_protocol(kProtocol);
  hcfg.set_field(kField);
  hcfg.set_sigmoid_mode(RuntimeConfig::SIGMOID_REAL);
  return hcfg;
}

#define SIM_DEFINE_UNARY(FNAME)                                       \
  std::vector<float> FNAME##_sim(const std::vector<float>& x,         \
                                 Visibility x_vis) {                  \
    std::vector<NdArrayRef> results = mpc::util::simulate(            \
        kNumParties, [&](std::shared_ptr<yasl::link::Context> lctx) { \
          HalContext hctx(makeConfig(), lctx);                        \
                                                                      \
          Value _x = make_value(&hctx, x_vis, x);                     \
          Value _y = FNAME(&hctx, _x);                                \
                                                                      \
          if (_y.isSecret()) {                                        \
            _y = _s2p(&hctx, _y).setDtype(_y.dtype());                \
          }                                                           \
          YASL_ENFORCE(_y.isPublic());                                \
                                                                      \
          return dump_public(&hctx, _y);                              \
        });                                                           \
                                                                      \
    /* check all party has the same result. */                        \
    YASL_ENFORCE(results.size() == kNumParties);                      \
                                                                      \
    return toVector<float>(results[0]);                               \
  }

SIM_DEFINE_UNARY(exp);
SIM_DEFINE_UNARY(log);
SIM_DEFINE_UNARY(reciprocal);
SIM_DEFINE_UNARY(logistic);

#define SIM_DEFINE_BINARY(FNAME)                                      \
  std::vector<float> FNAME##_sim(                                     \
      const std::vector<float>& x, Visibility x_vis,                  \
      const std::vector<float>& y, Visibility y_vis) {                \
    std::vector<NdArrayRef> results = mpc::util::simulate(            \
        kNumParties, [&](std::shared_ptr<yasl::link::Context> lctx) { \
          HalContext hctx(makeConfig(), lctx);                        \
                                                                      \
          Value _x = make_value(&hctx, x_vis, x);                     \
          Value _y = make_value(&hctx, y_vis, y);                     \
          Value _z = FNAME(&hctx, _x, _y);                            \
                                                                      \
          if (_z.isSecret()) {                                        \
            _z = _s2p(&hctx, _z).setDtype(_z.dtype());                \
          }                                                           \
          YASL_ENFORCE(_z.isPublic());                                \
                                                                      \
          return dump_public(&hctx, _z);                              \
        });                                                           \
                                                                      \
    /* check all party has the same result. */                        \
    YASL_ENFORCE(results.size() == kNumParties);                      \
                                                                      \
    return toVector<float>(results[0]);                               \
  }

SIM_DEFINE_BINARY(div);

}  // namespace spu::hal::simulation
