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

// clang-format off
// build generate_pir_data
// > bazel build //examples/cpp/pir:generate_pir_data -c opt
//
// To run the example, start two terminals:
// > ./generate_pir_data -data_count 10000 -label_len 32 -server_out_path pir_server.csv -client_out_path pir_client.csv
// clang-format on

#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "absl/strings/escaping.h"
#include "examples/cpp/utils.h"
#include "fmt/format.h"
#include "spdlog/spdlog.h"
#include "yacl/crypto/utils/rand.h"

llvm::cl::opt<int> DataCountOpt("data_count", llvm::cl::init(100000),
                                llvm::cl::desc("example data count"));

llvm::cl::opt<int> LabelLengthOpt("label_len", llvm::cl::init(288),
                                  llvm::cl::desc("label data length"));

llvm::cl::opt<float> QueryRateOpt(
    "query_rate", llvm::cl::init(0.001),
    llvm::cl::desc("rate of client data in serer data num"));

llvm::cl::opt<std::string> ServerOutPathOpt(
    "server_out_path", llvm::cl::init("pir_server_data.csv"),
    llvm::cl::desc("[out] server output path for pir example data"));

llvm::cl::opt<std::string> ClientOutPathOpt(
    "client_out_path", llvm::cl::init("pir_client_data.csv"),
    llvm::cl::desc("[out] client output path for pir example data"));

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv);

  size_t alice_item_size = DataCountOpt.getValue();
  size_t label_size = std::max<size_t>(1, LabelLengthOpt.getValue() / 2);

  uint64_t seed = yacl::crypto::RandU64();
  std::mt19937 rand(seed);

  double q1 = QueryRateOpt.getValue();

  SPDLOG_INFO("sample bernoulli_distribution: {}", q1);

  std::bernoulli_distribution dist1(q1);

  std::vector<size_t> bernoulli_items_idx;

  std::string id1_data = "111111";

  std::ofstream psi1_out_file;
  std::ofstream psi2_out_file;
  psi1_out_file.open(ServerOutPathOpt.getValue(), std::ios::out);
  psi2_out_file.open(ClientOutPathOpt.getValue(), std::ios::out);

  psi1_out_file << "id,id1"
                << ",label,label1" << '\r' << std::endl;
  psi2_out_file << "id,id1" << '\r' << std::endl;

  for (size_t idx = 0; idx < alice_item_size; idx++) {
    std::string a_item = fmt::format("{:010d}{:08d}", idx, idx + 900000000);
    std::string b_item;
    if (dist1(rand)) {
      psi2_out_file << a_item << "," << id1_data << '\r' << std::endl;
    }
    std::vector<uint8_t> label_bytes = yacl::crypto::RandBytes(label_size);
    std::vector<uint8_t> label_bytes2 = yacl::crypto::RandBytes(label_size);
    psi1_out_file << a_item << "," << id1_data << ","
                  << absl::BytesToHexString(absl::string_view(
                         reinterpret_cast<char *>(label_bytes.data()),
                         label_bytes.size()))
                  << ","
                  << absl::BytesToHexString(absl::string_view(
                         reinterpret_cast<char *>(label_bytes2.data()),
                         label_bytes2.size()))
                  << '\r' << std::endl;
  }

  psi1_out_file.close();
  psi2_out_file.close();

  return 0;
}