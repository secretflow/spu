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

#include "libspu/mpc/alkaid_boolean/arithmetic_gpu_ext.h"

#include "spdlog/spdlog.h"

#include "libspu/cuda_support/kernels.h"
#include "libspu/cuda_support/utils.h"

namespace spu::mpc::alkaid_boolean {

void matmul_aa_gpu(const NdArrayRef& x, const NdArrayRef& y, NdArrayRef& ret) {
  auto M = x.shape()[0];
  auto K = x.shape()[1];
  auto N = y.shape()[1];

  auto x_share_bytes = x.numel() * x.elsize();
  auto y_share_bytes = y.numel() * y.elsize();
  auto result_bytes = M * N * x.elsize();

  auto g_x = cuda::allocate(x_share_bytes);
  auto g_y = cuda::allocate(y_share_bytes);
  auto g_ret = cuda::allocate(result_bytes);

  auto* g_x2 = g_x.get() + x_share_bytes / 2;
  auto* g_y2 = g_y.get() + y_share_bytes / 2;
  auto* result_2 = g_ret.get() + result_bytes / 2;

  // x1
  cuda::DeinterleaveCopyToCudaDevice(x.data<std::byte>(),
                                     x.buf()->size() - x.offset(), x.shape(),
                                     x.strides(), g_x.get(), g_x2, x.elsize());

  // y1
  cuda::DeinterleaveCopyToCudaDevice(y.data<std::byte>(),
                                     y.buf()->size() - y.offset(), y.shape(),
                                     y.strides(), g_y.get(), g_y2, y.elsize());

  // x1*(y1+y2) + x2*y1 + k1
  // y1 + y2 - > y2
  cuda::add(reinterpret_cast<uint64_t*>(g_y2),
            reinterpret_cast<uint64_t*>(g_y.get()), y.shape().numel());

  // x1*y2
  cuda::matmul(M, N, K, reinterpret_cast<uint64_t*>(g_x.get()),
               reinterpret_cast<uint64_t*>(g_y2),
               reinterpret_cast<uint64_t*>(g_ret.get()));

  // x2*y1
  cuda::matmul(M, N, K, reinterpret_cast<uint64_t*>(g_x2),
               reinterpret_cast<uint64_t*>(g_y.get()),
               reinterpret_cast<uint64_t*>(result_2));

  // result1 + result2
  cuda::add(reinterpret_cast<uint64_t*>(g_ret.get()),
            reinterpret_cast<uint64_t*>(result_2), M * N);

  // Copy back
  cuda::CopyFromCudaDevice(g_ret.get(), ret.data<std::byte>(), ret.numel(),
                           ret.elsize(), ret.strides()[1]);
}

}  // namespace spu::mpc::alkaid_boolean