

#include "libspu/mpc/fantastic4/value.h"

#include "libspu/core/prelude.h"
#include "libspu/mpc/fantastic4/type.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::fantastic4 {

NdArrayRef getShare(const NdArrayRef& in, int64_t share_idx) {
  SPU_ENFORCE(share_idx == 0 || share_idx == 1 || share_idx == 2);

  auto new_strides = in.strides();
  std::transform(new_strides.cbegin(), new_strides.cend(), new_strides.begin(),
                 [](int64_t s) { return 3 * s; });

  if (in.eltype().isa<AShrTy>()) {
    const auto field = in.eltype().as<AShrTy>()->field();
    const auto ty = makeType<RingTy>(field);

    return NdArrayRef(
        in.buf(), ty, in.shape(), new_strides,
        in.offset() + share_idx * static_cast<int64_t>(ty.size()));
  } 
//   else if (in.eltype().isa<OShrTy>()) {
//     const auto field = in.eltype().as<OShrTy>()->field();
//     const auto ty = makeType<RingTy>(field);

//     return NdArrayRef(
//         in.buf(), ty, in.shape(), new_strides,
//         in.offset() + share_idx * static_cast<int64_t>(ty.size()));
//   } 
  else if (in.eltype().isa<BShrTy>()) {
    const auto stype = in.eltype().as<BShrTy>()->getBacktype();
    const auto ty = makeType<PtTy>(stype);
    return NdArrayRef(
        in.buf(), ty, in.shape(), new_strides,
        in.offset() + share_idx * static_cast<int64_t>(ty.size()));
  } 
//   else if (in.eltype().isa<PShrTy>()) {
//     const auto field = in.eltype().as<PShrTy>()->field();
//     const auto ty = makeType<RingTy>(field);

//     return NdArrayRef(
//         in.buf(), ty, in.shape(), new_strides,
//         in.offset() + share_idx * static_cast<int64_t>(ty.size()));
//   } 
  else {
    SPU_THROW("unsupported type {}", in.eltype());
  }
}

NdArrayRef getFirstShare(const NdArrayRef& in) { return getShare(in, 0); }

NdArrayRef getSecondShare(const NdArrayRef& in) { return getShare(in, 1); }

NdArrayRef getThirdShare(const NdArrayRef& in) { return getShare(in, 2); }

// 3 shares
NdArrayRef makeAShare(const NdArrayRef& s1, const NdArrayRef& s2, const NdArrayRef& s3,
                      FieldType field) {
  const Type ty = makeType<AShrTy>(field);

  SPU_ENFORCE(s3.eltype().as<Ring2k>()->field() == field);
  SPU_ENFORCE(s2.eltype().as<Ring2k>()->field() == field);
  SPU_ENFORCE(s1.eltype().as<Ring2k>()->field() == field);

  SPU_ENFORCE(s1.shape() == s2.shape(), "got s1={}, s2={}", s1, s2);
  SPU_ENFORCE(s2.shape() == s3.shape(), "got s2={}, s3={}", s2, s3);

  // 3 elements
  SPU_ENFORCE(ty.size() == 3 * s1.elsize());

  NdArrayRef res(ty, s1.shape());

  if (res.numel() != 0) {
    auto res_s1 = getFirstShare(res);
    auto res_s2 = getSecondShare(res);
    auto res_s3 = getThirdShare(res);

    ring_assign(res_s1, s1);
    ring_assign(res_s2, s2);
    ring_assign(res_s3, s3);
  }

  return res;
}

PtType calcBShareBacktype(size_t nbits) {
  if (nbits <= 8) {
    return PT_U8;
  }
  if (nbits <= 16) {
    return PT_U16;
  }
  if (nbits <= 32) {
    return PT_U32;
  }
  if (nbits <= 64) {
    return PT_U64;
  }
  if (nbits <= 128) {
    return PT_U128;
  }
  SPU_THROW("invalid number of bits={}", nbits);
}

}  // namespace spu::mpc::fantastic4
