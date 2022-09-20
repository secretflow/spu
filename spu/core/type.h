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

#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include "yasl/base/exception.h"

#include "spu/core/type_util.h"

namespace spu {

////////////////////////////////////////////////////////////////////////////
// Type interfaces.
////////////////////////////////////////////////////////////////////////////

// This trait means the data is maintained in ring 2^k, it DOES NOT mean that
// data storage is k-bits.
// For example, some protocol works in Zq, but the data it manipulates on is
// [0, 2^k).
class Ring2k {
 protected:
  FieldType field_{FT_INVALID};

 public:
  virtual ~Ring2k() = default;
  FieldType field() const { return field_; }
};

// The public interface.
//
// The value of this type is public visible for parties.
class Public {
 public:
  virtual ~Public() = default;
};

// The secret interface.
//
// The value of this type is a secret, invisible for parties.
class Secret {
 protected:
  // When in colocated mode, the data-provider also host a piece of spu
  // evaluator, it can infeed a cleartext as a 'private-secret`, a secret which
  // only know by itself. Some ops could benifit from this setting. i.e. If two
  // secret are provided by the same owner, secret multiplication could be done
  // without communication.
  //
  // def mul(x: Secret, y: Secret):
  //   if not x.owner or not y.owner or x.owner != y.owner:
  //     return mul_with_beaver(x, y)
  //
  //   # x, y are owned by the same owner, no beaver triple required.
  //   if rank() == x.owner:
  //     return x * y
  //   else:
  //     return 0
  //
  // In out-sourcing mode, this tag should be ignored.
  // In colocated mode, this tag describes this owner's rank information.
  //
  // Kernel implementation note:
  // This tag is `optional` for kernels, it's used to optimize certain kernels,
  // but without the optimiazation, the semantic should still be correct. i.e.
  //
  // Given four variables [x], [y], [z], [w] that [x] is a pure secret, [y] and
  // [z] are private-secret owned by Alice, [w] is a secret owned by Bob.
  //         Alice     Bob
  //   [x]    x0       x1        ; owner_ = -1
  //   [y]    y*       0         ; owner_ = 0
  //   [z]    z*       0         ; owner_ = 0
  //   [w]    0        w*        ; owner_ = 1
  //
  // and a kernel f, accept two secrets and return another secret.
  //   f :: ([a], [a]) -> [a]
  //
  // As a kernel implementer, we have to first make sure the `general case`,
  // that is, without care about `owner`, the functionality is correct.
  //   f ([x], [x])
  //   f ([x], [y])
  //   f ([y], [w])
  //
  // Then we may optimize it under certain case, for example
  //   f ([y], [z]), we can optimize it since y&z share the same owner.
  int64_t owner_ = -1;

 public:
  virtual ~Secret() = default;

  int64_t owner() const { return owner_; }
};

class AShare {
 public:
  virtual ~AShare() = default;
};

class BShare {
 protected:
  // Theoretically, `B share` works in F2 (mod 2) ring, which means it is
  // represented by a single bit. But just like other processing unit, data are
  // normally batched processed with `byte`, SPU also manipulate multi-bits
  // together.
  //
  // But for boolean circuit, number of bits is has critical performance impact,
  // so we record number of bit here to hint MPC protocol for perf improvement.
  //
  // This member represents the number of valid bits for a multi-bits storage,
  // the least significant of nbit_ are valid.
  size_t nbits_ = 0;

 public:
  virtual ~BShare() = default;

  size_t nbits() const { return nbits_; }

  void setNbits(size_t nbits) { nbits_ = nbits; }
};

////////////////////////////////////////////////////////////////////////////
// Type interfaces end.
////////////////////////////////////////////////////////////////////////////

// TODO(jint) document me, how to add a new type.
class TypeObject {
 public:
  friend class Type;

  virtual ~TypeObject() = default;

  // Return the unique type if of this class.
  //
  // This is a static method, which helps the system to identify the derived
  // type statically, it's used to do reflection.
  static std::string_view getStaticId();

  // Return type size in bytes.
  virtual size_t size() const = 0;

  // Return the string representation of this concept.
  virtual std::string toString() const = 0;

  // De-serialize from a string.
  virtual void fromString(std::string_view str) = 0;

  // Return true if two type object equals.
  virtual bool equals(TypeObject const* other) const = 0;

  // The below methods should be automatically filled when inherit from TypeImpl
  // Return the `id` of this type, which must be THE SAME as getStaticId.
  virtual std::string_view getId() const = 0;

  // Clone self.
  virtual std::unique_ptr<TypeObject> clone() const = 0;
};

// A value semantic type.
class Type final {
  std::unique_ptr<TypeObject> model_;

 public:
  // default constructable, as the void type.
  Type();
  explicit Type(std::unique_ptr<TypeObject> model);

  // copy and move constructable
  Type(const Type& other) : model_(other.model_->clone()) {}
  Type& operator=(const Type& other);
  Type(Type&& other) = default;
  Type& operator=(Type&& other) = default;

  // equal test
  bool operator==(Type const& other) const;
  bool operator!=(Type const& other) const { return !(*this == other); }

  // serialize and reflection.
  std::string toString() const;
  static Type fromString(std::string_view repr);

  // return object of this type's size in bytes.
  size_t size() const { return model_->size(); }

  // object oriented relationship
  template <typename T>
  T const* as() const;
  template <typename T>
  T* as();
  template <typename T>
  bool isa() const;
};

template <typename T>
T const* Type::as() const {
  T const* concrete_type = dynamic_cast<T*>(model_.get());
  YASL_ENFORCE(concrete_type, "casting from {} to {} failed", model_->getId(),
               typeid(T).name());
  return concrete_type;
}

template <typename T>
T* Type::as() {
  T* concrete_type = dynamic_cast<T*>(model_.get());
  YASL_ENFORCE(concrete_type, "casting from {} to {} failed", model_->getId(),
               typeid(T).name());
  return concrete_type;
}

template <typename T>
bool Type::isa() const {
  T const* concrete_type = dynamic_cast<T*>(model_.get());
  return concrete_type != nullptr;
}

std::ostream& operator<<(std::ostream& os, const Type& type);

template <typename ModelT, typename... Args>
Type makeType(Args&&... args) {
  return Type{std::make_unique<ModelT>(std::forward<Args>(args)...)};
}

template <typename DerivedT, typename BaseT, typename... InterfaceT>
class TypeImpl : public BaseT, public InterfaceT... {
 public:
  std::string_view getId() const override { return DerivedT::getStaticId(); }

  std::unique_ptr<TypeObject> clone() const override {
    return std::make_unique<DerivedT>(static_cast<DerivedT const&>(*this));
  }
};

// Builtin type, void
class VoidTy : public TypeImpl<VoidTy, TypeObject> {
  using Base = TypeImpl<VoidTy, TypeObject>;

 public:
  using Base::Base;

  static std::string_view getStaticId() { return "Void"; }

  size_t size() const override { return 0u; }

  void fromString(std::string_view detail) override {
    YASL_ENFORCE(detail.empty(), "expect empty, got={}", detail);
  };

  std::string toString() const override { return ""; }

  bool equals(TypeObject const* other) const override { return true; }
};

// Builtin type, plaintext types.
class PtTy : public TypeImpl<PtTy, TypeObject> {
  using Base = TypeImpl<PtTy, TypeObject>;
  PtType pt_type_;

 public:
  using Base::Base;
  explicit PtTy(PtType pt_type) : pt_type_(pt_type) {}
  PtType pt_type() const { return pt_type_; }

  static std::string_view getStaticId() { return "Plaintext"; };

  bool equals(TypeObject const* other) const override {
    auto const* derived_other = dynamic_cast<PtTy const*>(other);
    YASL_ENFORCE(derived_other);
    return pt_type() == derived_other->pt_type();
  }

  size_t size() const override { return SizeOf(pt_type_); }

  std::string toString() const override { return PtType_Name(pt_type_); }

  void fromString(std::string_view detail) override {
    YASL_ENFORCE(PtType_Parse(std::string(detail), &pt_type_),
                 "parse failed from={}", detail);
  }
};

inline Type makePtType(PtType etype) { return makeType<PtTy>(etype); }

template <typename T>
Type makePtType() {
  return makePtType(PtTypeToEnum<T>::value);
}

bool isFloatTy(const Type& type);
bool isIntTy(const Type& type);

// predefine plaintext types.
extern Type Void;
extern Type I8;
extern Type U8;
extern Type I16;
extern Type U16;
extern Type I32;
extern Type U32;
extern Type I64;
extern Type U64;
extern Type F32;
extern Type F64;
extern Type I128;
extern Type U128;

class RingTy : public TypeImpl<RingTy, TypeObject, Ring2k> {
  using Base = TypeImpl<RingTy, TypeObject, Ring2k>;

 public:
  using Base::Base;
  explicit RingTy(FieldType field) { field_ = field; }

  static std::string_view getStaticId() { return "Ring"; }

  size_t size() const override { return SizeOf(GetStorageType(field_)); }

  void fromString(std::string_view detail) override {
    YASL_ENFORCE(FieldType_Parse(std::string(detail), &field_),
                 "parse failed from={}", detail);
  };

  std::string toString() const override { return FieldType_Name(field()); }

  bool equals(TypeObject const* other) const override {
    auto const* derived_other = dynamic_cast<RingTy const*>(other);
    YASL_ENFORCE(derived_other);
    return field() == derived_other->field();
  }
};

class TypeContext {
 public:
  using TypeCreateFn =
      std::function<std::unique_ptr<TypeObject>(std::string_view)>;

  TypeContext() {
    addTypes<VoidTy, PtTy, RingTy>();  // Base types that we need to register
  }

  template <typename T>
  void addType() {
    std::unique_lock<std::mutex> lock(creator_mutex);
    creators[T::getStaticId()] =
        [](std::string_view detail) -> std::unique_ptr<T> {
      auto x = std::make_unique<T>();
      x->fromString(detail);
      return x;
    };
  }

  template <typename... Args>
  void addTypes() {
    (void)std::initializer_list<int>{0, (TypeContext::addType<Args>(), 0)...};
  }

  static TypeContext* getTypeContext() {
    static TypeContext ctx;
    return &ctx;
  }

  TypeCreateFn getTypeCreateFunction(std::string_view keyword) {
    auto fctor = creators.find(keyword);
    YASL_ENFORCE(fctor != creators.end(), "type not found, {}", keyword);
    return fctor->second;
  }

 private:
  std::unordered_map<std::string_view, TypeCreateFn> creators;
  std::mutex creator_mutex;
};

}  // namespace spu
