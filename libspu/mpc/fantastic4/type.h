
#pragma once

#include "libspu/core/type.h"

namespace spu::mpc::fantastic4 {

class AShrTy : public TypeImpl<AShrTy, RingTy, Secret, AShare> {
  using Base = TypeImpl<AShrTy, RingTy, Secret, AShare>;

 public:
  using Base::Base;
  static std::string_view getStaticId() { return "fantastic4.AShr"; }

  explicit AShrTy(FieldType field) { field_ = field; }

  // 3-out-of-4 shares
  size_t size() const override { return SizeOf(GetStorageType(field_)) * 3; }
};

class BShrTy : public TypeImpl<BShrTy, TypeObject, Secret, BShare> {
  using Base = TypeImpl<BShrTy, TypeObject, Secret, BShare>;
  PtType back_type_ = PT_INVALID;

 public:
  using Base::Base;
  explicit BShrTy(PtType back_type, size_t nbits) {
    SPU_ENFORCE(SizeOf(back_type) * 8 >= nbits,
                "backtype={} has not enough bits={}", back_type, nbits);
    back_type_ = back_type;
    nbits_ = nbits;
  }

  PtType getBacktype() const { return back_type_; }

  static std::string_view getStaticId() { return "fantastic4.BShr"; }

  void fromString(std::string_view detail) override {
    auto comma = detail.find_first_of(',');
    auto back_type_str = detail.substr(0, comma);
    auto nbits_str = detail.substr(comma + 1);
    SPU_ENFORCE(PtType_Parse(std::string(back_type_str), &back_type_),
                "parse failed from={}", detail);
    nbits_ = std::stoul(std::string(nbits_str));
  }

  std::string toString() const override {
    return fmt::format("{},{}", PtType_Name(back_type_), nbits_);
  }

  // 3-out-of-4 shares
  size_t size() const override { return SizeOf(back_type_) * 3; }

  bool equals(TypeObject const* other) const override {
    auto const* derived_other = dynamic_cast<BShrTy const*>(other);
    SPU_ENFORCE(derived_other);
    return getBacktype() == derived_other->getBacktype() &&
           nbits() == derived_other->nbits();
  }
};

void registerTypes();

}  // namespace spu::mpc::fantastic4