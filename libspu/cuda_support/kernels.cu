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

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

#include "libspu/cuda_support/kernels.h"

#define KERNEL_CALL(funcname, n) funcname<<<((n) + 255) / 256, 256>>>
#define GLOBAL_INDEX (blockDim.x * blockIdx.x + threadIdx.x)

namespace spu::cuda {

namespace {

// NN means both A and B is not transposed. That is: C = A * B.
cudaError_t cutlassMatmul_u64(int64_t M, int64_t N, int64_t K, uint64_t alpha,
                              uint64_t const* A, int64_t lda, uint64_t const* B,
                              int64_t ldb, uint64_t beta, uint64_t* C,
                              int64_t ldc) {
  using CutlassGemm = cutlass::gemm::device::Gemm<
      uint64_t, cutlass::layout::RowMajor, uint64_t, cutlass::layout::RowMajor,
      uint64_t, cutlass::layout::RowMajor, uint64_t, cutlass::arch::OpClassSimt,
      cutlass::arch::Sm80>;
  CutlassGemm gemm_operator;
  CutlassGemm::Arguments args({(int)M, (int)N, (int)K}, {A, lda}, {B, ldb},
                              {C, ldc}, {C, ldc}, {alpha, beta});
  cutlass::Status status = gemm_operator(args);
  if (status != cutlass::Status::kSuccess) {
    return cudaErrorUnknown;
  }
  return cudaSuccess;
}

__global__ void deviceArrayAddInplaceKernel(uint64_t* A, const uint64_t* B,
                                            int64_t numel) {
  size_t idx = GLOBAL_INDEX;
  if (idx < numel) {
    A[idx] += B[idx];
  }
}

}  // namespace

// The matrices are in row-major format.
// They are copied to the GPU memory to perform device matrix multiplication.
void matmul(int64_t M, int64_t N, int64_t K, const uint64_t* A, uint64_t* B,
            uint64_t* C) {
  cutlassMatmul_u64(M, N, K, 1, A, K, B, N, 0, C, N);
}

void add(uint64_t* A, const uint64_t* B, int64_t numel) {
  KERNEL_CALL(deviceArrayAddInplaceKernel, numel)(A, B, numel);
}

}  // namespace spu::cuda
