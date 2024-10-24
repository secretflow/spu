// Copyright 2023 Ant Group Co., Ltd.
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

#include "libspu/psi/utils/progress.h"

#include "gtest/gtest.h"

namespace spu::psi {

class ProgressTest : public ::testing::Test {};

TEST_F(ProgressTest, ProgressSinglePhase_Normal) {
  auto progress = std::make_shared<Progress>();

  auto data = progress->Get();
  EXPECT_EQ(data.percentage, 0);
  EXPECT_EQ(data.total, 1);
  EXPECT_EQ(data.finished, 0);
  EXPECT_EQ(data.running, 0);
  EXPECT_EQ(data.description, "0%");

  progress->Update(50);
  data = progress->Get();
  EXPECT_EQ(data.percentage, 50);
  EXPECT_EQ(data.description, "50%");

  progress->Done();
  data = progress->Get();
  EXPECT_EQ(data.percentage, 100);
  EXPECT_EQ(data.description, "100%");
}

TEST_F(ProgressTest, ProgressMultiPhase_Normal) {
  auto progress = std::make_shared<Progress>();
  progress->SetWeights({10, 90});
  auto data = progress->Get();
  EXPECT_EQ(data.percentage, 0);
  EXPECT_EQ(data.total, 2);
  EXPECT_EQ(data.finished, 0);
  EXPECT_EQ(data.running, 0);
  EXPECT_EQ(data.description, "");

  progress->Update(50);  // do nothing
  data = progress->Get();
  EXPECT_EQ(data.percentage, 0);
  EXPECT_EQ(data.description, "");

  auto step1 = progress->NextSubProgress("Step1");
  data = progress->Get();
  EXPECT_EQ(data.percentage, 0);
  EXPECT_EQ(data.description, "Step1, 0%");

  step1->Update(50);
  data = progress->Get();
  EXPECT_EQ(data.percentage, 5);
  EXPECT_EQ(data.description, "Step1, 50%");

  auto step2 = progress->NextSubProgress("Step2");
  data = progress->Get();
  EXPECT_EQ(data.percentage, 10);
  EXPECT_EQ(data.description, "Step2, 0%");

  step2->Update(50);
  data = progress->Get();
  EXPECT_EQ(data.percentage, 55);  // 10 + 90 * 0.5
  EXPECT_EQ(data.description, "Step2, 50%");

  step2->Done();
  data = progress->Get();
  EXPECT_EQ(data.percentage, 100);
  EXPECT_EQ(data.description, "Step2, 100%");
}

}  // namespace spu::psi
