#pragma once

#include "libspu/mpc/io_interface.h"

namespace spu::mpc::fantastic4 {

class Fantastic4Io final : public BaseIo {
 public:
  using BaseIo::BaseIo;

  std::vector<NdArrayRef> toShares(const NdArrayRef& raw, Visibility vis,
                                   int owner_rank) const override;

  Type getShareType(Visibility vis, int owner_rank = -1) const override;

  NdArrayRef fromShares(const std::vector<NdArrayRef>& shares) const override;

  std::vector<NdArrayRef> makeBitSecret(const PtBufferView& in) const override;

  size_t getBitSecretShareSize(size_t numel) const override;

  bool hasBitSecretSupport() const override { return true; }
};

std::unique_ptr<Fantastic4Io> makeFantastic4Io(FieldType field, size_t npc);

} 