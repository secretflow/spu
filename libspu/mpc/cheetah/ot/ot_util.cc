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

#include "libspu/mpc/cheetah/ot/ot_util.h"

#include <numeric>

#include "ot_util.h"

#include "libspu/core/prelude.h"

namespace spu::mpc::cheetah {

uint8_t BoolToU8(absl::Span<const uint8_t> bits) {
  size_t len = bits.size();
  SPU_ENFORCE(len >= 1 && len <= 8);
  return std::accumulate(
      bits.data(), bits.data() + len,
      /*init*/ static_cast<uint8_t>(0),
      [](uint8_t init, uint8_t next) { return (init << 1) | (next & 1); });
}

void U8ToBool(absl::Span<uint8_t> bits, uint8_t u8) {
  size_t len = std::min(8UL, bits.size());
  SPU_ENFORCE(len >= 1);
  for (size_t i = 0; i < len; ++i) {
    bits[i] = (u8 & 1);
    u8 >>= 1;
  }
}

NdArrayRef OpenShare(const NdArrayRef &shr, ReduceOp op, size_t nbits,
                     std::shared_ptr<Communicator> conn) {
  SPU_ENFORCE(conn != nullptr);
  SPU_ENFORCE(shr.eltype().isa<Ring2k>());
  SPU_ENFORCE(op == ReduceOp::ADD or op == ReduceOp::XOR);

  auto field = shr.eltype().as<Ring2k>()->field();
  size_t fwidth = SizeOf(field) * 8;
  if (nbits == 0) {
    nbits = fwidth;
  }
  SPU_ENFORCE(nbits <= fwidth, "nbits out-of-bound");
  bool packable = fwidth > nbits;
  if (not packable) {
    return conn->allReduce(op, shr, "open");
  }

  size_t numel = shr.numel();
  size_t compact_numel = CeilDiv(numel * nbits, fwidth);

  NdArrayRef out(shr.eltype(), {(int64_t)numel});
  DISPATCH_ALL_FIELDS(field, [&]() {
    auto inp = absl::MakeConstSpan(&shr.at<ring2k_t>(0), numel);
    auto oup = absl::MakeSpan(&out.at<ring2k_t>(0), compact_numel);

    size_t used = ZipArray(inp, nbits, oup);
    SPU_ENFORCE_EQ(used, compact_numel);

    std::vector<ring2k_t> opened;
    if (op == ReduceOp::XOR) {
      opened = conn->allReduce<ring2k_t, std::bit_xor>(oup, "open");
    } else {
      opened = conn->allReduce<ring2k_t, std::plus>(oup, "open");
    }

    oup = absl::MakeSpan(&out.at<ring2k_t>(0), numel);
    UnzipArray(absl::MakeConstSpan(opened), nbits, oup);
  });
  return out.reshape(shr.shape());
}

#ifdef __x86_64__
#include <immintrin.h>
#elif __aarch64__
#include "sse2neon.h"
#endif

#define INP(x, y) inp[(x) * ncols / 8 + (y) / 8]
#define OUT(x, y) out[(y) * nrows / 8 + (x) / 8]

#ifdef __x86_64__
__attribute__((target("sse2")))
#endif
void SseTranspose(uint8_t *out, uint8_t const *inp, uint64_t nrows, uint64_t ncols) {
  uint64_t rr;
  uint64_t cc;
  int i;
  int h;
  union {
    __m128i x;
    uint8_t b[16];
  } tmp;
  __m128i vec;
  SPU_ENFORCE(nrows % 8 == 0 && ncols % 8 == 0);

  // Do the main body in 16x8 blocks:
  for (rr = 0; rr + 16 <= nrows; rr += 16) {
    for (cc = 0; cc < ncols; cc += 8) {
      vec = _mm_set_epi8(INP(rr + 15, cc), INP(rr + 14, cc), INP(rr + 13, cc),
                         INP(rr + 12, cc), INP(rr + 11, cc), INP(rr + 10, cc),
                         INP(rr + 9, cc), INP(rr + 8, cc), INP(rr + 7, cc),
                         INP(rr + 6, cc), INP(rr + 5, cc), INP(rr + 4, cc),
                         INP(rr + 3, cc), INP(rr + 2, cc), INP(rr + 1, cc),
                         INP(rr + 0, cc));
      for (i = 8; --i >= 0; vec = _mm_slli_epi64(vec, 1))
        *(uint16_t *)&OUT(rr, cc + i) = _mm_movemask_epi8(vec);
    }
  }
  if (rr == nrows) {
    return;
  }

  // The remainder is a block of 8x(16n+8) bits (n may be 0).
  //  Do a PAIR of 8x8 blocks in each step:
  if ((ncols % 8 == 0 && ncols % 16 != 0) ||
      (nrows % 8 == 0 && nrows % 16 != 0)) {
    // The fancy optimizations in the else-branch don't work if the above
    // if-condition holds, so we use the simpler non-simd variant for that case.
    for (cc = 0; cc + 16 <= ncols; cc += 16) {
      for (i = 0; i < 8; ++i) {
        tmp.b[i] = h = *(uint16_t const *)&INP(rr + i, cc);
        tmp.b[i + 8] = h >> 8;
      }
      for (i = 8; --i >= 0; tmp.x = _mm_slli_epi64(tmp.x, 1)) {
        OUT(rr, cc + i) = h = _mm_movemask_epi8(tmp.x);
        OUT(rr, cc + i + 8) = h >> 8;
      }
    }
  } else {
    for (cc = 0; cc + 16 <= ncols; cc += 16) {
      vec = _mm_set_epi16(*(uint16_t const *)&INP(rr + 7, cc),
                          *(uint16_t const *)&INP(rr + 6, cc),
                          *(uint16_t const *)&INP(rr + 5, cc),
                          *(uint16_t const *)&INP(rr + 4, cc),
                          *(uint16_t const *)&INP(rr + 3, cc),
                          *(uint16_t const *)&INP(rr + 2, cc),
                          *(uint16_t const *)&INP(rr + 1, cc),
                          *(uint16_t const *)&INP(rr + 0, cc));
      for (i = 8; --i >= 0; vec = _mm_slli_epi64(vec, 1)) {
        OUT(rr, cc + i) = h = _mm_movemask_epi8(vec);
        OUT(rr, cc + i + 8) = h >> 8;
      }
    }
  }
  if (cc == ncols) return;

  //  Do the remaining 8x8 block:
  for (i = 0; i < 8; ++i) {
    tmp.b[i] = INP(rr + i, cc);
  }
  for (i = 8; --i >= 0; tmp.x = _mm_slli_epi64(tmp.x, 1)) {
    OUT(rr, cc + i) = _mm_movemask_epi8(tmp.x);
  }
}

}  // namespace spu::mpc::cheetah
