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

#include "absl/strings/str_split.h"
#include "fmt/ostream.h"

#include "libspu/core/prelude.h"
#include "libspu/core/type_util.h"

namespace spu {

////////////////////////////////////////////////////////////////////////////
// Type interfaces.
////////////////////////////////////////////////////////////////////////////

class BaseRingType {
 protected:
  int16_t valid_bits_ = 0;
  // Type that reflects what this `Type`'s semantic behavior
  SemanticType semantic_type_ = SE_INVALID;
  // Type that reflects actual storage for this `Type`
  StorageType storage_type_ = ST_INVALID;

 public:
  virtual ~BaseRingType() = default;
  SemanticType semantic_type() const { return semantic_type_; }
  void set_semantic_type(SemanticType type) { semantic_type_ = type; }
  StorageType storage_type() const { return storage_type_; }
  void set_storage_type(StorageType type) { storage_type_ = type; }

  int64_t valid_bits() const { return valid_bits_; }
  void set_valid_bits(int16_t val) { valid_bits_ = val; }
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
 public:
  virtual ~Secret() = default;
};

class Private {
 protected:
  // When in colocated mode, the data-provider also host a piece of spu
  // evaluator, it can infeed a cleartext as a 'private-secret`, a secret which
  // only know by itself. Some ops could benefit from this setting. i.e. If two
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
  // but without the optimization, the semantic should still be correct. i.e.
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
  virtual ~Private() = default;

  int64_t owner() const { return owner_; }
};

class ArithShare {
 public:
  virtual ~ArithShare() = default;
};

class BoolShare {
 protected:
  // Note: Packed means the underlying storage using builtin scalar types to
  // store a set of bit sharings. Make sure your know what you are doing when
  // set it to true.
  bool packed_{false};

 public:
  virtual ~BoolShare() = default;
  bool is_packed() const { return packed_; }
};

class OramShare {
 public:
  virtual ~OramShare() = default;
};

class OramPubShare {
 public:
  virtual ~OramPubShare() = default;
};

// Permutation share, a secret permutation can be a composition of a series of
// individual permutations hold by different parties. Each individual
// permutation is represented as a PShare in SPU. PShare is a secret type.
// We use the letter m for naming PShare values in order to be distinguished
// from public values.
class PermShare {
 public:
  virtual ~PermShare() = default;
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
  //
  // Warn: change attributes of a type object should NOT affect the object size.
  virtual size_t size() const = 0;

  // Return actual storage type.
  //
  // Warn: change attributes of a type object should NOT affect the object size.
  virtual StorageType storage_type() const = 0;

  // Return semantic ring type.
  //
  // Warn: change attributes of a type object should NOT affect the object size.
  virtual SemanticType semantic_type() const = 0;

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
  std::unique_ptr<TypeObject> impl_;

 public:
  // default constructable, as the void type.
  Type();
  explicit Type(std::unique_ptr<TypeObject> impl);

  // copy and move constructable
  Type(const Type& other) : impl_(other.impl_->clone()) {}
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
  inline size_t size() const { return impl_->size(); }
  // return the actual plaintext type for backend storage
  inline StorageType storage_type() const { return impl_->storage_type(); }
  // return the semantic type
  inline SemanticType semantic_type() const { return impl_->semantic_type(); }

  // object oriented relationship
  template <typename T>
  T const* as() const {
    T const* concrete_type = dynamic_cast<T const*>(impl_.get());
    SPU_ENFORCE(concrete_type, "casting from {} to {} failed", impl_->getId(),
                typeid(T).name());
    return concrete_type;
  }

  template <typename T>
  T* as() {
    T* concrete_type = dynamic_cast<T*>(impl_.get());
    SPU_ENFORCE(concrete_type, "casting from {} to {} failed", impl_->getId(),
                typeid(T).name());
    return concrete_type;
  }

  template <typename T>
  bool isa() const {
    T const* concrete_type = dynamic_cast<T const*>(impl_.get());
    return concrete_type != nullptr;
  }
};

std::ostream& operator<<(std::ostream& os, const Type& type);

template <typename ModelT, typename... Args>
Type makeType(Args&&... args) {
  return Type(std::make_unique<ModelT>(std::forward<Args>(args)...));
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

  size_t size() const override { return 0U; }

  StorageType storage_type() const override { return ST_INVALID; }
  SemanticType semantic_type() const override { return SE_INVALID; }

  void fromString(std::string_view detail) override {
    SPU_ENFORCE(detail.empty(), "expect empty, got={}", detail);
  };

  std::string toString() const override { return ""; }

  bool equals(TypeObject const*) const override { return true; }
};

class RingTy : public TypeImpl<RingTy, TypeObject, BaseRingType> {
  using Base = TypeImpl<RingTy, TypeObject, BaseRingType>;

 public:
  using Base::Base;
  explicit RingTy(SemanticType semantic_type, size_t width) {
    semantic_type_ = semantic_type;
    valid_bits_ = width;
    storage_type_ = GetStorageType(valid_bits_);
  }

  static std::string_view getStaticId() { return "Ring"; }

  size_t size() const override { return SizeOf(storage_type_); }

  StorageType storage_type() const override { return storage_type_; }
  SemanticType semantic_type() const override { return semantic_type_; }

  void fromString(std::string_view detail) override {
    std::vector<std::string> tokens = absl::StrSplit(detail, ',');
    SemanticType_Parse(tokens[0], &semantic_type_);
    valid_bits_ = std::stoul(tokens[1]);
    storage_type_ = GetStorageType(valid_bits_);
  };

  std::string toString() const override {
    return fmt::format("{},{}", semantic_type_, valid_bits_);
  }

  bool equals(TypeObject const* other) const override {
    auto const* derived_other = dynamic_cast<RingTy const*>(other);
    SPU_ENFORCE(derived_other);
    return valid_bits_ == derived_other->valid_bits_ &&
           semantic_type_ == derived_other->semantic_type_;
  }
};

class TypeContext final {
 public:
  using TypeCreateFn =
      std::function<std::unique_ptr<TypeObject>(std::string_view)>;

 private:
  std::unordered_map<std::string_view, TypeCreateFn> creators_;
  std::mutex creator_mutex_;

 public:
  TypeContext() {
    addTypes<VoidTy, RingTy>();  // Base types that we need to register
  }

  template <typename T>
  void addType() {
    std::unique_lock<std::mutex> lock(creator_mutex_);
    creators_[T::getStaticId()] =
        [](std::string_view detail) -> std::unique_ptr<T> {
      auto x = std::make_unique<T>();
      x->fromString(detail);
      return x;
    };
  }

  template <typename... Args>
  void addTypes() {
    (TypeContext::addType<Args>(), ...);
  }

  static TypeContext* getTypeContext() {
    static TypeContext ctx;
    return &ctx;
  }

  TypeCreateFn getTypeCreateFunction(std::string_view keyword) {
    auto fctor = creators_.find(keyword);
    SPU_ENFORCE(fctor != creators_.end(), "type not found, {}", keyword);
    return fctor->second;
  }
};

inline auto format_as(const Type& t) { return fmt::streamed(t); }

}  // namespace spu
