// Copyright 2023 Ant Group Co., Ltd.
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

#include "examples/cpp/pir/keyword_pir_utils.h"

#include <filesystem>
#include <memory>

#include "libspu/psi/io/io.h"
#include "libspu/psi/utils/batch_provider.h"

namespace spu {
namespace pir {
namespace examples {
namespace utils {

namespace {

constexpr size_t kEccKeySize = 32;

}

std::vector<uint8_t> ReadEcSecretKeyFile(const std::string &file_path) {
  size_t file_byte_size = 0;
  try {
    file_byte_size = std::filesystem::file_size(file_path);
  } catch (std::filesystem::filesystem_error &e) {
    SPU_THROW("ReadEcSecretKeyFile {} Error: {}", file_path, e.what());
  }
  SPU_ENFORCE(file_byte_size == kEccKeySize,
              "error format: key file bytes is not {}", kEccKeySize);

  std::vector<uint8_t> secret_key(kEccKeySize);

  auto in =
      spu::psi::io::BuildInputStream(spu::psi::io::FileIoOptions(file_path));
  in->Read(secret_key.data(), kEccKeySize);
  in->Close();

  return secret_key;
}

size_t CsvFileDataCount(const std::string &file_path,
                        const std::vector<std::string> &ids) {
  size_t data_count = 0;

  std::shared_ptr<spu::psi::IBatchProvider> batch_provider =
      std::make_shared<spu::psi::CsvBatchProvider>(file_path, ids);

  while (true) {
    auto batch = batch_provider->ReadNextBatch(4096);
    if (batch.empty()) {
      break;
    }
    data_count += batch.size();
  }

  return data_count;
}

void WritePsiParams(const std::string file_path,
                    const apsi::PSIParams &psi_params) {
  yacl::Buffer params_buffer = spu::psi::PsiParamsToBuffer(psi_params);

  auto out =
      spu::psi::io::BuildOutputStream(spu::psi::io::FileIoOptions(file_path));

  out->Write(params_buffer.data(), params_buffer.size());
  out->Close();
}

apsi::PSIParams ReadPsiParams(const std::string &file_path) {
  size_t file_byte_size = 0;
  try {
    file_byte_size = std::filesystem::file_size(file_path);
  } catch (std::filesystem::filesystem_error &e) {
    SPU_THROW("ReadPsiParams {} Error: {}", file_path, e.what());
  }

  yacl::Buffer params_buffer(file_byte_size);

  auto in =
      spu::psi::io::BuildInputStream(spu::psi::io::FileIoOptions(file_path));
  in->Read(params_buffer.data(), params_buffer.size());
  in->Close();

  apsi::PSIParams psi_params = spu::psi::ParsePsiParamsProto(params_buffer);

  return psi_params;
}

}  // namespace utils
}  // namespace examples
}  // namespace pir
}  // namespace spu
