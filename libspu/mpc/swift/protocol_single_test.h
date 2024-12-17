#include "gtest/gtest.h"
#include "yacl/link/link.h"

#include "libspu/mpc/api_test_params.h"

namespace spu::mpc::test {

class ArithmeticTest : public ::testing::TestWithParam<OpTestParams> {};

class BooleanTest : public ::testing::TestWithParam<OpTestParams> {};

}  // namespace spu::mpc::test
