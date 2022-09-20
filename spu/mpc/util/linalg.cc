// Copyright 2022 Ant Group Co., Ltd.
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

#include "spu/mpc/util/linalg.h"

#include <memory>

#include "llvm/Support/Threading.h"

namespace spu::mpc::linalg {

struct EigenThreadPool {
  EigenThreadPool() {
    // std::thread::hardware_concurrency() counts hyper-threads as well, use
    // llvm one to get a more accurate count
    // Eigen suggests to use number of physical cores for better performance
    // See https://eigen.tuxfamily.org/dox/TopicMultiThreading.html
    auto nProc =
        llvm::heavyweight_hardware_concurrency().compute_thread_count();
    pool_ = std::make_unique<Eigen::ThreadPool>(nProc);
    device_ = std::make_unique<Eigen::ThreadPoolDevice>(pool_.get(), nProc);
  }
  std::unique_ptr<Eigen::ThreadPool> pool_;
  std::unique_ptr<Eigen::ThreadPoolDevice> device_;
};

Eigen::ThreadPoolDevice* getEigenThreadPoolDevice() {
  static EigenThreadPool pool;
  return pool.device_.get();
}

}  // namespace spu::mpc::linalg