
#pragma once

#include "libspu/core/type.h"
#include "libspu/mpc/object.h"

namespace spu::mpc::spdzwisefield {

class AShrTy : public TypeImpl<AShrTy, RingTy, Secret, AShare> {
    using Base = TypeImpl<AShrTy, RingTy, Secret, AShare>;

public:
    using Base::Base;
    static std::string_view getStaticId() { return "spdzwisefield.AShr"; }

    explicit AShrTy(FieldType field, int owner_rank = -1) {
        field_ = field;
        owner_ = owner_rank;
    }

    size_t size() const override { return SizeOf(GetStorageType(field_)) * 2; }

};

class BShrTy : public TypeImpl<BShrTy, RingTy, Secret, BShare> {
    using Base = TypeImpl<BShrTy, RingTy, Secret, BShare>;

    static constexpr size_t kDefaultNumBits = std::numeric_limits<size_t>::max();

    public:
    using Base::Base;
    explicit BShrTy(FieldType field, size_t nbits = kDefaultNumBits) {
        field_ = field;
        nbits_ = nbits == kDefaultNumBits ? SizeOf(field) * 8 : nbits;
        YACL_ENFORCE(nbits_ <= SizeOf(field) * 8);
    }

    static std::string_view getStaticId() { return "spdzwisefield.BShr"; }

    void fromString(std::string_view detail) override {
        auto comma = detail.find_first_of(',');
        auto field_str = detail.substr(0, comma);
        auto nbits_str = detail.substr(comma + 1);
        YACL_ENFORCE(FieldType_Parse(std::string(field_str), &field_),
                    "parse failed from={}", detail);
        nbits_ = std::stoul(std::string(nbits_str));
    };

    std::string toString() const override {
        return fmt::format("{},{}", FieldType_Name(field()), nbits_);
    }

};

void registerTypes();

}