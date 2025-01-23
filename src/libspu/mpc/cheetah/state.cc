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

#include "libspu/mpc/cheetah/state.h"

#include <future>

#include "libspu/core/context.h"
#include "libspu/core/ndarray_ref.h"
#include "libspu/core/prelude.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::cheetah {

namespace {
// Return num_workers for the given size of jobs
size_t InitOTState(KernelEvalContext* ctx, size_t njobs) {
  constexpr size_t kMinWorkSize = 2048;
  if (njobs == 0) {
    return 0;
  }
  auto* comm = ctx->getState<Communicator>();
  auto* ot_state = ctx->getState<CheetahOTState>();
  size_t nworker =
      std::min(ot_state->maximum_instances(), CeilDiv(njobs, kMinWorkSize));
  for (size_t w = 0; w < nworker; ++w) {
    ot_state->LazyInit(comm, w);
  }
  return nworker;
}
}  // namespace

void CheetahMulState::makeSureCacheSize(FieldType field, int64_t numel) {
  // NOTE(juhou): make sure the lock is obtained
  SPU_ENFORCE(numel > 0);
  if (field_ != field) {
    // drop all previous cache
    cached_sze_ = 0;
  }

  if (cached_sze_ >= numel) {
    return;
  }

  //  create one batch OLE which then converted to Beavers
  //  Math:
  //   Alice samples rand0 and views it as rand0 = a0||b0
  //   Bob samples rand1 and views it as rand1 = b1||a0
  //   The multiplication rand0 * rand1 gives two cross term a0*b1||a1*b0
  //   Then the beaver (a0, b0, c0) and (a1, b1, c1)
  //   where c0 = a0*b0 + <a0*b1> + <a1*b0>
  //         c1 = a1*b1 + <a0*b1> + <a1*b0>
  mul_prot_->LazyInitKeys(field);
  const int rank = mul_prot_->Rank();
  const int64_t ole_sze = mul_prot_->OLEBatchSize();
  const int64_t num_ole = CeilDiv<size_t>(2 * numel, ole_sze);
  const int64_t num_beaver = (num_ole * ole_sze) / 2;
  auto rand = ring_rand(field, {num_ole * ole_sze});
  auto cross = mul_prot_->MulOLE(rand, rank == 0);

  NdArrayRef beaver[3];
  NdArrayRef a0b1;
  NdArrayRef a1b0;
  if (rank == 0) {
    beaver[0] = rand.slice({0}, {num_beaver}, {1});
    beaver[1] = rand.slice({num_beaver}, {num_beaver * 2}, {1});
  } else {
    beaver[0] = rand.slice({num_beaver}, {num_beaver * 2}, {1});
    beaver[1] = rand.slice({0}, {num_beaver}, {1});
  }
  a0b1 = cross.slice({0}, {num_beaver}, {1});
  a1b0 = cross.slice({num_beaver}, {num_beaver * 2}, {1});

  beaver[2] =
      ring_add(ring_add(cross.slice({0}, {num_beaver}, {1}),
                        cross.slice({num_beaver}, {2 * num_beaver}, {1})),
               ring_mul(beaver[0], beaver[1]));

  DISPATCH_ALL_FIELDS(field, [&]() {
    for (size_t i : {0, 1, 2}) {
      auto tmp = ring_zeros(field, {num_beaver + cached_sze_});
      NdArrayView<ring2k_t> new_cache(tmp);

      // concate two array
      if (cached_sze_ > 0) {
        NdArrayView<const ring2k_t> old_cache(cached_beaver_[i]);
        pforeach(0, cached_sze_,
                 [&](int64_t j) { new_cache[j] = old_cache[j]; });
      }

      NdArrayView<const ring2k_t> _beaver(beaver[i]);
      pforeach(0, num_beaver,
               [&](int64_t j) { new_cache[cached_sze_ + j] = _beaver[j]; });
      cached_beaver_[i] = tmp;
    }
  });

  field_ = field;
  cached_sze_ += num_beaver;

  SPU_ENFORCE(cached_sze_ >= numel);
}

std::array<NdArrayRef, 3> CheetahMulState::TakeCachedBeaver(FieldType field,
                                                            int64_t numel) {
  SPU_ENFORCE(numel > 0);
  std::unique_lock guard(lock_);
  makeSureCacheSize(field, numel);

  std::array<NdArrayRef, 3> ret;
  for (size_t i : {0, 1, 2}) {
    SPU_ENFORCE(cached_beaver_[i].numel() >= numel);
    ret[i] = cached_beaver_[i].slice({0}, {numel}, {1});
    if (cached_sze_ == numel) {
      // empty cache now
      // NOTE(lwj): should use `{0}` as empty while `{}` is a scalar
      cached_beaver_[i] = NdArrayRef(cached_beaver_[i].eltype(), {0});
    } else {
      cached_beaver_[i] = cached_beaver_[i].slice({numel}, {cached_sze_}, {1});
    }
  }

  cached_sze_ -= numel;
  return ret;
}

NdArrayRef TiledDispatchOTFunc(KernelEvalContext* ctx, const NdArrayRef& x,
                               OTUnaryFunc func) {
  const Shape& shape = x.shape();
  SPU_ENFORCE(shape.numel() > 0);
  // (lazy) init OT
  int64_t numel = x.numel();
  int64_t nworker = InitOTState(ctx, numel);
  int64_t workload = nworker == 0 ? 0 : CeilDiv(numel, nworker);

  if (shape.ndim() != 1) {
    // TiledDispatchOTFunc over flatten input
    return TiledDispatchOTFunc(ctx, x.reshape({numel}), func)
        .reshape(x.shape());
  }

  std::vector<NdArrayRef> outs(nworker);
  std::vector<std::future<void>> futures;

  int64_t slice_end = 0;
  for (int64_t wi = 0; wi + 1 < nworker; ++wi) {
    int64_t slice_bgn = wi * workload;
    slice_end = std::min(numel, slice_bgn + workload);
    auto slice_input = x.slice({slice_bgn}, {slice_end}, {});
    futures.emplace_back(std::async(
        [&](int64_t idx, const NdArrayRef& input) {
          auto ot_instance = ctx->getState<CheetahOTState>()->get(idx);
          outs[idx] = func(input, ot_instance);
        },
        wi, slice_input));
  }

  auto slice_input = x.slice({slice_end}, {numel}, {1});
  auto ot_instance = ctx->getState<CheetahOTState>()->get(nworker - 1);
  outs[nworker - 1] = func(slice_input, ot_instance);

  for (auto&& f : futures) {
    f.get();
  }

  NdArrayRef out(outs[0].eltype(), x.shape());
  int64_t offset = 0;

  for (auto& out_slice : outs) {
    std::memcpy(out.data<std::byte>() + offset, out_slice.data(),
                out_slice.numel() * out.elsize());
    offset += out_slice.numel() * out.elsize();
  }

  return out;
}

NdArrayRef TiledDispatchOTFunc(KernelEvalContext* ctx, const NdArrayRef& x,
                               const NdArrayRef& y, OTBinaryFunc func) {
  const Shape& shape = x.shape();
  SPU_ENFORCE(shape.numel() > 0);
  SPU_ENFORCE_EQ(shape, y.shape());
  // (lazy) init OT
  int64_t numel = x.numel();
  int64_t nworker = InitOTState(ctx, numel);
  int64_t workload = nworker == 0 ? 0 : CeilDiv(numel, nworker);

  if (shape.ndim() != 1) {
    // TiledDispatchOTFunc over flatten input
    return TiledDispatchOTFunc(ctx, x.reshape({numel}), y.reshape({numel}),
                               func)
        .reshape(x.shape());
  }

  std::vector<NdArrayRef> outs(nworker);
  std::vector<std::future<void>> futures;

  int64_t slice_end = 0;
  for (int64_t wi = 0; wi + 1 < nworker; ++wi) {
    int64_t slice_bgn = wi * workload;
    slice_end = std::min(numel, slice_bgn + workload);
    auto x_slice = x.slice({slice_bgn}, {slice_end}, {1});
    auto y_slice = y.slice({slice_bgn}, {slice_end}, {1});
    futures.emplace_back(std::async(
        [&](int64_t idx, const NdArrayRef& inp0, const NdArrayRef& inp1) {
          auto ot_instance = ctx->getState<CheetahOTState>()->get(idx);
          outs[idx] = func(inp0, inp1, ot_instance);
        },
        wi, x_slice, y_slice));
  }

  auto x_slice = x.slice({slice_end}, {numel}, {});
  auto y_slice = y.slice({slice_end}, {numel}, {});
  auto ot_instance = ctx->getState<CheetahOTState>()->get(nworker - 1);
  outs[nworker - 1] = func(x_slice, y_slice, ot_instance);

  for (auto&& f : futures) {
    f.get();
  }

  NdArrayRef out(outs[0].eltype(), x.shape());
  int64_t offset = 0;

  for (auto& out_slice : outs) {
    std::memcpy(out.data<std::byte>() + offset, out_slice.data(),
                out_slice.numel() * out.elsize());
    offset += out_slice.numel() * out.elsize();
  }

  return out;
}

NdArrayRef TiledDispatchOTFunc(KernelEvalContext* ctx,
                               absl::Span<const uint8_t> x,
                               OTUnaryFuncWithU8 func) {
  SPU_ENFORCE(not x.empty());
  // (lazy) init OT
  int64_t numel = x.size();
  int64_t nworker = InitOTState(ctx, numel);
  int64_t workload = nworker == 0 ? 0 : CeilDiv(numel, nworker);

  std::vector<NdArrayRef> outs(nworker);
  std::vector<std::future<void>> futures;

  int64_t slice_end = 0;
  for (int64_t wi = 0; wi + 1 < nworker; ++wi) {
    int64_t slice_bgn = wi * workload;
    slice_end = std::min(numel, slice_bgn + workload);
    auto slice_input = x.subspan(slice_bgn, slice_end - slice_bgn);
    futures.emplace_back(std::async(
        [&](int64_t idx, absl::Span<const uint8_t> input) {
          auto ot_instance = ctx->getState<CheetahOTState>()->get(idx);
          outs[idx] = func(input, ot_instance);
        },
        wi, slice_input));
  }

  auto slice_input = x.subspan(slice_end, numel - slice_end);
  auto ot_instance = ctx->getState<CheetahOTState>()->get(nworker - 1);
  outs[nworker - 1] = func(slice_input, ot_instance);

  for (auto&& f : futures) {
    f.get();
  }

  NdArrayRef out(outs[0].eltype(), {numel});
  int64_t offset = 0;

  for (auto& out_slice : outs) {
    std::memcpy(out.data<std::byte>() + offset, out_slice.data(),
                out_slice.numel() * out.elsize());
    offset += out_slice.numel() * out.elsize();
  }

  return out;
}

NdArrayRef TiledDispatchOTFunc(KernelEvalContext* ctx, const NdArrayRef& x,
                               absl::Span<const uint8_t> y,
                               OTBinaryFuncWithU8 func) {
  const Shape& shape = x.shape();
  SPU_ENFORCE(shape.numel() > 0);
  SPU_ENFORCE_EQ(shape.numel(), (int64_t)y.size());
  // (lazy) init OT
  int64_t numel = x.numel();
  int64_t nworker = InitOTState(ctx, numel);
  int64_t workload = nworker == 0 ? 0 : CeilDiv(numel, nworker);

  if (shape.ndim() != 1) {
    // TiledDispatchOTFunc over flatten input
    return TiledDispatchOTFunc(ctx, x.reshape({numel}), y, func)
        .reshape(x.shape());
  }

  std::vector<NdArrayRef> outs(nworker);
  std::vector<std::future<void>> futures;

  int64_t slice_end = 0;
  for (int64_t wi = 0; wi + 1 < nworker; ++wi) {
    int64_t slice_bgn = wi * workload;
    slice_end = std::min(numel, slice_bgn + workload);
    auto x_slice = x.slice({slice_bgn}, {slice_end}, {1});
    auto y_slice = y.subspan(slice_bgn, slice_end - slice_bgn);
    futures.emplace_back(std::async(
        [&](int64_t idx, const NdArrayRef& inp0,
            absl::Span<const uint8_t> inp1) {
          auto ot_instance = ctx->getState<CheetahOTState>()->get(idx);
          outs[idx] = func(inp0, inp1, ot_instance);
        },
        wi, x_slice, y_slice));
  }

  auto x_slice = x.slice({slice_end}, {numel}, {});
  auto y_slice = y.subspan(slice_end, numel - slice_end);
  auto ot_instance = ctx->getState<CheetahOTState>()->get(nworker - 1);
  outs[nworker - 1] = func(x_slice, y_slice, ot_instance);

  for (auto&& f : futures) {
    f.get();
  }

  NdArrayRef out(outs[0].eltype(), x.shape());
  int64_t offset = 0;

  for (auto& out_slice : outs) {
    std::memcpy(out.data<std::byte>() + offset, out_slice.data(),
                out_slice.numel() * out.elsize());
    offset += out_slice.numel() * out.elsize();
  }

  return out;
}

}  // namespace spu::mpc::cheetah
