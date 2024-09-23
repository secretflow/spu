

#include "libspu/mpc/fantastic4/protocol.h"

#include "libspu/mpc/ab_api.h"
#include "libspu/mpc/ab_api_test.h"
#include "libspu/mpc/api.h"
#include "libspu/mpc/api_test.h"

namespace spu::mpc::test {
namespace {

RuntimeConfig makeConfig(FieldType field) {
  RuntimeConfig conf;
  conf.set_protocol(ProtocolKind::FANTASTIC4);
  conf.set_field(field);
  return conf;
}

}  // namespace

// INSTANTIATE_TEST_SUITE_P(
//     Fantastic4, ApiTest,
//     testing::Combine(testing::Values(makeFantastic4Protocol),              //
//                      testing::Values(makeConfig(FieldType::FM32),    //
//                                      makeConfig(FieldType::FM64),    //
//                                      makeConfig(FieldType::FM128)),  //
//                      testing::Values(4)),                            //
//     [](const testing::TestParamInfo<ApiTest::ParamType>& p) {
//       return fmt::format("{}x{}", std::get<1>(p.param).field(),
//                          std::get<2>(p.param));
//     });

INSTANTIATE_TEST_SUITE_P(
    Fantastic4, ArithmeticTest,
    testing::Combine(testing::Values(makeFantastic4Protocol),              //
                     testing::Values(makeConfig(FieldType::FM32),    //
                                     makeConfig(FieldType::FM64),    //
                                     makeConfig(FieldType::FM128)),  //

                    // /////////////////////////
                    // npc = 4
                    // ////////////////////////
                     testing::Values(4)),                            //
    [](const testing::TestParamInfo<ArithmeticTest::ParamType>& p) {
      return fmt::format("{}x{}", std::get<1>(p.param).field(),
                         std::get<2>(p.param));
    });

INSTANTIATE_TEST_SUITE_P(
    Fantastic4, BooleanTest,
    testing::Combine(testing::Values(makeFantastic4Protocol),              //
                     testing::Values(makeConfig(FieldType::FM32),    //
                                     makeConfig(FieldType::FM64),    //
                                     makeConfig(FieldType::FM128)),  //
                     testing::Values(4)),                            //
    [](const testing::TestParamInfo<BooleanTest::ParamType>& p) {
      return fmt::format("{}x{}", std::get<1>(p.param).field(),
                         std::get<2>(p.param));
    });

INSTANTIATE_TEST_SUITE_P(
    Fantastic4, ConversionTest,
    testing::Combine(testing::Values(makeFantastic4Protocol),              //
                     testing::Values(makeConfig(FieldType::FM32),    //
                                     makeConfig(FieldType::FM64),    //
                                     makeConfig(FieldType::FM128)),  //
                     testing::Values(4)),                            //
    [](const testing::TestParamInfo<BooleanTest::ParamType>& p) {
      return fmt::format("{}x{}", std::get<1>(p.param).field(),
                         std::get<2>(p.param));
    });

}  // namespace spu::mpc::test
