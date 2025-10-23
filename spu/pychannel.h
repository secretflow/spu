// Copyright 2025 Ant Group Co., Ltd.
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

#include "pybind11/pybind11.h"
#include "yacl/link/transport/channel.h"

class PyChannel : public yacl::link::transport::IChannel,
                  public pybind11::trampoline_self_life_support {
 public:
  using yacl::link::transport::IChannel::IChannel;

  void SendAsync(const std::string& key, yacl::Buffer buf) override {
    PYBIND11_OVERRIDE_PURE(void, yacl::link::transport::IChannel, SendAsync,
                           key, std::move(buf));
  }

  void SendAsyncThrottled(const std::string& key, yacl::Buffer buf) override {
    PYBIND11_OVERRIDE_PURE(void, yacl::link::transport::IChannel,
                           SendAsyncThrottled, key, std::move(buf));
  }

  void Send(const std::string& key, yacl::ByteContainerView value) override {
    PYBIND11_OVERRIDE_PURE(void, yacl::link::transport::IChannel, Send, key,
                           value);
  }

  yacl::Buffer Recv(const std::string& key) override {
    PYBIND11_OVERRIDE_PURE(yacl::Buffer, yacl::link::transport::IChannel, Recv,
                           key);
  }

  void SetRecvTimeout(uint64_t timeout_ms) override {
    PYBIND11_OVERRIDE_PURE(void, yacl::link::transport::IChannel,
                           SetRecvTimeout, timeout_ms);
  }

  uint64_t GetRecvTimeout() const override {
    PYBIND11_OVERRIDE_PURE(uint64_t, yacl::link::transport::IChannel,
                           GetRecvTimeout);
  }

  virtual void WaitLinkTaskFinish() override {
    PYBIND11_OVERRIDE_PURE(void, yacl::link::transport::IChannel,
                           WaitLinkTaskFinish);
  }

  virtual void Abort() override {
    PYBIND11_OVERRIDE_PURE(void, yacl::link::transport::IChannel, Abort);
  }

  virtual void SetThrottleWindowSize(size_t size) override {
    PYBIND11_OVERRIDE_PURE(void, yacl::link::transport::IChannel,
                           SetThrottleWindowSize, size);
  }

  void TestSend(uint32_t timeout) override {
    PYBIND11_OVERRIDE_PURE(void, yacl::link::transport::IChannel, TestSend,
                           timeout);
  }

  void TestRecv() override {
    PYBIND11_OVERRIDE_PURE(void, yacl::link::transport::IChannel, TestRecv);
  }

  void SetChunkParallelSendSize(size_t size) override {
    PYBIND11_OVERRIDE_PURE(void, yacl::link::transport::IChannel,
                           SetChunkParallelSendSize, size);
  }
};