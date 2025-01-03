#include "libspu/mpc/io_test.h"

#include "libspu/mpc/fantastic4/io.h"

namespace spu::mpc::fantastic4 {

INSTANTIATE_TEST_SUITE_P(
    Fantastic4IoTest, IoTest,
    testing::Combine(testing::Values(makeFantastic4Io),  //
                     testing::Values(4),           //
                     testing::Values(FieldType::FM32, FieldType::FM64,
                                     FieldType::FM128)),
    [](const testing::TestParamInfo<IoTest::ParamType>& p) {
      return fmt::format("{}x{}", std::get<1>(p.param), std::get<2>(p.param));
    });

} 