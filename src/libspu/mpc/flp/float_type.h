// Shared floating-point representation (secret-shared) for FLP kernels.
// Migrated from cheetah/fp/float_type.h and adapted to `spu::mpc::flp`.

#ifndef SPU_MPC_FLP_FLOAT_TYPE_H_
#define SPU_MPC_FLP_FLOAT_TYPE_H_

#include "libspu/core/value.h"

namespace spu::mpc::flp {

struct SharedFloat {
  ::spu::Value z;  // zero-bit: 1 if value is zero
  ::spu::Value s;  // sign bit: 0/1
  ::spu::Value e;  // exponent (unbiased) as integer representation
  ::spu::Value m;  // mantissa stored explicitly with q+1 bits

  int p = 0;  // exponent parameter
  int q = 0;  // mantissa fraction bits (so stored width is q+1)

  SharedFloat() = default;

  SharedFloat(const ::spu::Value& z_, const ::spu::Value& s_,
              const ::spu::Value& e_, const ::spu::Value& m_,
              int p_ = 0, int q_ = 0)
      : z(z_), s(s_), e(e_), m(m_), p(p_), q(q_) {}

};

}  // namespace spu::mpc::flp

#endif  // SPU_MPC_FLP_FLOAT_TYPE_H_
