// Copyright 2022 Ant Group Co., Ltd.
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

#include <exception>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "llvm/Support/CommandLine.h"
#include "spdlog/spdlog.h"

#include "libspu/psi/core/ecdh_psi.h"
#include "libspu/psi/core/mini_psi.h"

llvm::cl::opt<int> RoleOpt("role", llvm::cl::init(1),
                           llvm::cl::desc("sender:0, receiver: 1"));

llvm::cl::opt<int> RankOpt("rank", llvm::cl::init(0),
                           llvm::cl::desc("self rank 0/1"));

llvm::cl::opt<std::string> InPathOpt("in", llvm::cl::init("in.csv"),
                                     llvm::cl::desc("input file"));

llvm::cl::opt<std::string> OutPathOpt("out", llvm::cl::init("out.csv"),
                                      llvm::cl::desc("psi out file"));

llvm::cl::opt<std::string> IdOpt("id", llvm::cl::init("id"),
                                 llvm::cl::desc("id of the csv"));

llvm::cl::opt<std::string> LocalOpt("local", llvm::cl::init("127.0.0.1:1234"),
                                    llvm::cl::desc("local address and port"));

llvm::cl::opt<std::string> RemoteOpt("remote", llvm::cl::init("127.0.0.1:1235"),
                                     llvm::cl::desc("remote address and port"));

llvm::cl::opt<std::string> ProtocolOpt("protocol",
                                       llvm::cl::init("semi-honest"),
                                       llvm::cl::desc("semi-honest/malicious"));

namespace {
constexpr uint32_t kLinkRecvTimeout = 30 * 60 * 1000;

class CSVRow {
 public:
  std::string_view operator[](std::size_t index) const {
    return std::string_view(&m_line_[m_data_[index] + 1],
                            m_data_[index + 1] - (m_data_[index] + 1));
  }
  std::size_t size() const { return m_data_.size() - 1; }
  void readNextRow(std::istream& str) {
    std::getline(str, m_line_);

    m_data_.clear();
    m_data_.emplace_back(-1);
    std::string::size_type pos = 0;
    while ((pos = m_line_.find(',', pos)) != std::string::npos) {
      m_data_.emplace_back(pos);
      ++pos;
    }
    // This checks for a trailing comma with no data after it.
    pos = m_line_.size();
    m_data_.emplace_back(pos);
  }

 private:
  std::string m_line_;
  std::vector<int> m_data_;
};

std::istream& operator>>(std::istream& str, CSVRow& data) {
  data.readNextRow(str);
  return str;
}

std::vector<std::string> ReadCsvData(const std::string& file_name) {
  std::vector<std::string> items;
  std::ifstream file(file_name);

  CSVRow row;
  // read header
  file >> row;
  while (file >> row) {
    items.emplace_back(row[0]);
  }
  return items;
}

void WriteCsvData(const std::string& file_name,
                  const std::vector<std::string>& items) {
  std::ofstream out_file;
  out_file.open(file_name, std::ios::out);
  out_file << "id" << '\r' << std::endl;
  for (const auto& item : items) {
    out_file << item << '\r' << std::endl;
  }
  out_file.close();
}

std::shared_ptr<yacl::link::Context> CreateContext(
    int self_rank, yacl::link::ContextDesc& lctx_desc) {
  std::shared_ptr<yacl::link::Context> link_ctx;

  yacl::link::FactoryBrpc factory;
  link_ctx = factory.CreateContext(lctx_desc, self_rank);
  link_ctx->ConnectToMesh();

  return link_ctx;
}

std::shared_ptr<yacl::link::Context> CreateLinks(const std::string& local_addr,
                                                 const std::string& remote_addr,
                                                 int self_rank) {
  yacl::link::ContextDesc lctx_desc;

  // int self_rank = 0;

  if (self_rank == 0) {
    std::string id = fmt::format("party{}", 0);
    lctx_desc.parties.push_back({id, local_addr});
    id = fmt::format("party{}", 1);
    lctx_desc.parties.push_back({id, remote_addr});
  } else {
    std::string id = fmt::format("party{}", 0);
    lctx_desc.parties.push_back({id, remote_addr});
    id = fmt::format("party{}", 1);
    lctx_desc.parties.push_back({id, local_addr});
  }

  return CreateContext(self_rank, lctx_desc);
}

}  // namespace

//
// script generate_psi.py
//    used to generate test data 18 digits id  50% intersection rate
//
// psi demo
//  -- sender
// ./bazel-bin/psi/core/mini_psi_demo -in ./100m/psi_1.csv -local
// "127.0.0.1:1234" -remote "127.0.0.1:2222" -rank 0 -role 0 -protocol
// semi-honest/malicious
//
//  -- receiver
// ./bazel-bin/psi/core/mini_psi_demo -in ./100m/psi_2.csv -remote
// "127.0.0.1:1234" -local "127.0.0.1:2222" -rank 1 -role 1 --protocol
// semi-honest
//
int main(int argc, char** argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv);

  std::cout << RoleOpt.getValue() << "," << RankOpt.getValue() << std::endl;
  std::cout << InPathOpt.getValue() << "," << OutPathOpt.getValue()
            << std::endl;
  std::cout << LocalOpt.getValue() << "," << RemoteOpt.getValue() << std::endl;
  std::cout << ProtocolOpt.getValue() << "," << IdOpt.getValue() << std::endl;

  std::vector<std::string> items = ReadCsvData(InPathOpt.getValue());
  std::cout << items.size() << std::endl;

  try {
    std::shared_ptr<yacl::link::Context> link_ctx = CreateLinks(
        InPathOpt.getValue(), RemoteOpt.getValue(), RankOpt.getValue());
    link_ctx->SetRecvTimeout(kLinkRecvTimeout);

    std::string file_name = ProtocolOpt.getValue();
    file_name.append("_").append(OutPathOpt.getValue());

    std::vector<std::string> intersection;
    if (ProtocolOpt.getValue() == "semi-honest") {
      intersection = spu::psi::RunEcdhPsi(link_ctx, items, 1);
      if (RankOpt.getValue() == 1) {
        SPDLOG_INFO("intersection size:{}", intersection.size());

        WriteCsvData(file_name, intersection);
      }
    } else if (ProtocolOpt.getValue() == "malicious") {
      if (RoleOpt.getValue() == 0) {
        spu::psi::MiniPsiSendBatch(link_ctx, items);
      } else if (RoleOpt.getValue() == 1) {
        intersection = spu::psi::MiniPsiRecvBatch(link_ctx, items);
        SPDLOG_INFO("intersection size:{}", intersection.size());
        WriteCsvData(file_name, intersection);
      }
    }
  } catch (std::exception& e) {
    SPDLOG_INFO("exception {}", e.what());
  }

  return 0;
}
