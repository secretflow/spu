// Copyright 2024 Ant Group Co., Ltd.
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

#include "libspu/dialect/ring/IR/bytecode.h"

#include "llvm/ADT/TypeSwitch.h"

#include "libspu/core/prelude.h"
#include "libspu/dialect/ring/IR/dialect.h"
#include "libspu/dialect/ring/IR/types.h"
#include "libspu/dialect/ring/IR/version.h"
#include "libspu/dialect/utils/utils.h"

namespace mlir::spu::ring {

enum TypeCode : uint64_t {
  kSecretType = 0,
};

class RingDialectBytecodeInterface : public BytecodeDialectInterface {
 public:
  using BytecodeDialectInterface::BytecodeDialectInterface;

  explicit RingDialectBytecodeInterface(Dialect *dialect)
      : BytecodeDialectInterface(dialect) {}

  Attribute readAttribute(DialectBytecodeReader &reader) const override {
    Attribute attr;
    if (failed(reader.readAttribute(attr))) {
      reader.emitError() << "Unknown attribute: "
                         << spu::mlirObjectToString(attr);
      return Attribute();
    }

    return attr;
  }

  LogicalResult writeAttribute(Attribute attr,
                               DialectBytecodeWriter &writer) const override {
    writer.writeAttribute(attr);
    return success();
  }

  Type readType(DialectBytecodeReader &reader) const override {
    uint64_t code = 0;
    if (failed(reader.readVarInt(code))) {
      return Type();
    }

    switch (code) {
      case TypeCode::kSecretType: {
        Type baseType;
        if (failed(reader.readType(baseType))) {
          return Type();
        }

        return SecretType::get(baseType);
      }
      default: {
        reader.emitError() << "Unknown type: " << code;
        return Type();
      }
    }

    return Type();
  }

  LogicalResult writeType(Type type,
                          DialectBytecodeWriter &writer) const override {
    return TypeSwitch<Type, LogicalResult>(type)
        .Case<SecretType>([&](SecretType type) {
          writer.writeVarInt(TypeCode::kSecretType);
          writer.writeType(type.getBaseType());
          return success();
        })
        .Default([&](Type) {
          SPU_THROW("Unknown type: ", spu::mlirObjectToString(type));
          return failure();
        });
  }

  void writeVersion(DialectBytecodeWriter &writer) const override {
    if (auto version = cast<RingDialect>(getDialect())->getVersion();
        version && version->isValid()) {
      writer.writeVarInt(version->major);
      writer.writeVarInt(version->minor);
      writer.writeVarInt(version->patch);
    }
  }

  std::unique_ptr<DialectVersion> readVersion(
      DialectBytecodeReader &reader) const override {
    auto version = std::make_unique<RingDialectVersion>();

    if (!version || failed(reader.readVarInt(version->major)) ||
        failed(reader.readVarInt(version->minor)) ||
        failed(reader.readVarInt(version->patch)) || !version->isValid()) {
      reader.emitError() << "failed to read version";
      return nullptr;
    }

    return version;
  }
};

void addBytecodeInterface(RingDialect *dialect) {
  dialect->addInterfaces<RingDialectBytecodeInterface>();
}

}  // namespace mlir::spu::ring
