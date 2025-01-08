

#pragma once

#include "libspu/core/ndarray_ref.h"
#include "libspu/core/type_util.h"

namespace spu::mpc::fantastic4 {

NdArrayRef getShare(const NdArrayRef& in, int64_t share_idx);

NdArrayRef getFirstShare(const NdArrayRef& in);

NdArrayRef getSecondShare(const NdArrayRef& in);

NdArrayRef getThirdShare(const NdArrayRef& in);

NdArrayRef makeAShare(const NdArrayRef& s1, const NdArrayRef& s2, const NdArrayRef& s3, 
                      FieldType field);

PtType calcBShareBacktype(size_t nbits);

template <typename T>
std::vector<T> getShareAs(const NdArrayRef& in, size_t share_idx) {
  SPU_ENFORCE(share_idx == 0 || share_idx == 1 || share_idx == 2);

  NdArrayRef share = getShare(in, share_idx);
  SPU_ENFORCE(share.elsize() == sizeof(T));

  auto numel = in.numel();

  std::vector<T> res(numel);
  DISPATCH_UINT_PT_TYPES(share.eltype().as<PtTy>()->pt_type(), [&]() {
    NdArrayView<ScalarT> _share(share);
    for (auto idx = 0; idx < numel; ++idx) {
      res[idx] = _share[idx];
    }
  });

  return res;
}

#define PFOR_GRAIN_SIZE 8192

} 
