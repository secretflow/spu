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

#include <cstddef>
#include <cstring>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "pybind11/iostream.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "spdlog/spdlog.h"
#include "spu/pychannel.h"
#include "yacl/base/exception.h"
#include "yacl/link/algorithm/allgather.h"
#include "yacl/link/algorithm/barrier.h"
#include "yacl/link/algorithm/broadcast.h"
#include "yacl/link/algorithm/gather.h"
#include "yacl/link/algorithm/scatter.h"
#include "yacl/link/context.h"
#include "yacl/link/factory.h"

#include "libspu/compiler/compile.h"
#include "libspu/core/context.h"
#include "libspu/core/logging.h"
#include "libspu/core/value.h"
#include "libspu/device/api.h"
#include "libspu/device/io.h"
#include "libspu/device/pphlo/pphlo_executor.h"
#include "libspu/device/symbol_table.h"
#include "libspu/mpc/factory.h"
#include "libspu/spu.h"
#include "libspu/version.h"

// Add missing includes for brpc and fmt
#include "butil/macros.h"
#include "fmt/format.h"

#ifdef CHECK_AVX
#include "cpu_features/cpuinfo_x86.h"
#endif

namespace py = pybind11;

// Forward declare brpc FLAGS
namespace brpc {
extern uint64_t FLAGS_max_body_size;
extern int64_t FLAGS_socket_max_unwritten_bytes;
}  // namespace brpc

namespace spu {

namespace {

[[maybe_unused]] std::string FormatMissingCpuFeatureMsg(const char* name) {
  return fmt::format(
      "This version of SPU was built using {} instructions, which your "
      "CPU and/or operating system do not support. You may be able to work "
      "around this issue by building SPU from source.",
      name);
}

}  // namespace

// Convert yacl::Buffer to py::array_t using zero-copy with move semantics
py::array_t<uint8_t> BufferToArray(yacl::Buffer buffer) {
  // Create a capsule that takes ownership of the buffer data using move
  // semantics This achieves zero-copy by transferring ownership instead of
  // copying
  auto* buf_ptr = new yacl::Buffer(std::move(buffer));
  py::capsule capsule(
      buf_ptr, [](void* data) { delete static_cast<yacl::Buffer*>(data); });

  // Create numpy array as a view of the buffer data without copying
  return py::array_t<uint8_t>({buf_ptr->size()},         // shape
                              {1},                       // strides
                              buf_ptr->data<uint8_t>(),  // data pointer
                              capsule);  // capsule to manage lifetime
}

// Generic buffer to numpy array conversion with type information (zero-copy)
template <typename T>
py::array_t<T> BufferToTypedArray(yacl::Buffer buffer) {
  // Calculate number of elements
  size_t num_elements = buffer.size() / sizeof(T);
  SPU_ENFORCE(buffer.size() % sizeof(T) == 0,
              "Buffer size {} is not divisible by sizeof({})", buffer.size(),
              sizeof(T));

  // Create a capsule that takes ownership of the buffer using move semantics
  auto* buf_ptr = new yacl::Buffer(std::move(buffer));
  py::capsule capsule(
      buf_ptr, [](void* data) { delete static_cast<yacl::Buffer*>(data); });

  // Create typed numpy array that shares the buffer memory
  return py::array_t<T>(
      {num_elements},                                  // shape
      {1},                                             // strides
      reinterpret_cast<T*>(buf_ptr->data<uint8_t>()),  // data pointer
      capsule);  // capsule to manage lifetime
}

// Specialization for uint8_t (most common case)
template <>
py::array_t<uint8_t> BufferToTypedArray<uint8_t>(yacl::Buffer buffer) {
  return BufferToArray(
      std::move(buffer));  // Use the existing zero-copy implementation
}

#define NO_GIL py::call_guard<py::gil_scoped_release>()

void BindLink(py::module& m) {
  using yacl::link::CertInfo;
  using yacl::link::Context;
  using yacl::link::ContextDesc;
  using yacl::link::RetryOptions;
  using yacl::link::SSLOptions;
  using yacl::link::VerifyOptions;
  using yacl::link::transport::IChannel;

  // TODO(jint) expose this tag to python?
  constexpr char PY_CALL_TAG[] = "PY_CALL";

  m.doc() = R"pbdoc(
              SPU Link Library
                  )pbdoc";

  py::class_<IChannel, std::shared_ptr<IChannel>, PyChannel>(m, "IChannel")
      .def(py::init<>())
      .def("send_async",
           static_cast<void (IChannel::*)(const std::string&, yacl::Buffer)>(
               &IChannel::SendAsync))
      .def("send_async_throttled",
           static_cast<void (IChannel::*)(const std::string&, yacl::Buffer)>(
               &IChannel::SendAsyncThrottled))
      .def("send", &IChannel::Send)
      .def("recv", &IChannel::Recv)
      .def("test_send", &IChannel::TestSend)
      .def("test_recv", &IChannel::TestRecv)
      .def("set_throttle_window_size", &IChannel::SetThrottleWindowSize)
      .def("set_chunk_parallel_send_size", &IChannel::SetChunkParallelSendSize)
      .def("wait_link_task_finish", &IChannel::WaitLinkTaskFinish)
      .def("abort", &IChannel::Abort);

  py::class_<CertInfo>(m, "CertInfo", "The config info used for certificate")
      .def_readwrite("certificate_path", &CertInfo::certificate_path,
                     "certificate file path")
      .def_readwrite("private_key_path", &CertInfo::private_key_path,
                     "private key file path");

  py::class_<VerifyOptions>(m, "VerifyOptions",
                            "The options used for verify certificate")
      .def_readwrite("verify_depth", &VerifyOptions::verify_depth,
                     "maximum depth of the certificate chain for verification")
      .def_readwrite("ca_file_path", &VerifyOptions::ca_file_path,
                     "the trusted CA file path");

  py::class_<RetryOptions>(m, "RetryOptions",
                           "The options used for channel retry")
      .def_readwrite("max_retry", &RetryOptions::max_retry, "max retry count")
      .def_readwrite("retry_interval_ms", &RetryOptions::retry_interval_ms,
                     "first retry interval")
      .def_readwrite("retry_interval_incr_ms",
                     &RetryOptions::retry_interval_incr_ms,
                     "the amount of time to increase between retries")
      .def_readwrite("max_retry_interval_ms",
                     &RetryOptions::max_retry_interval_ms,
                     "the max interval between retries")
      .def_readwrite("error_codes", &RetryOptions::error_codes,
                     "retry on these error codes, if empty, retry on all codes")
      .def_readwrite(
          "http_codes", &RetryOptions::http_codes,
          "retry on these http codes, if empty, retry on all http codes")
      .def_readwrite("aggressive_retry", &RetryOptions::aggressive_retry,
                     "do aggressive retry");

  py::class_<SSLOptions>(m, "SSLOptions", "The options used for ssl")
      .def_readwrite("cert", &SSLOptions::cert,
                     "certificate used for authentication")
      .def_readwrite("verify", &SSLOptions::verify,
                     "options used to verify the peer's certificate");

  py::class_<ContextDesc::Party>(
      m, "Party", "The party that participate the secure computation")
      .def_readonly("id", &ContextDesc::Party::id, "the id, unique per link")
      .def_readonly("host", &ContextDesc::Party::host, "host address")
      .def("__repr__", [](const ContextDesc::Party& self) {
        return fmt::format("Party(id={}, host={})", self.id, self.host);
      });

  py::class_<ContextDesc>(
      m, "Desc", "Link description, describes parties which joins the link")
      .def(py::init<>())
      .def_readwrite("id", &ContextDesc::id, "the uuid")
      .def_readonly("parties", &ContextDesc::parties,
                    "the parties that joins the computation")
      .def_readwrite("connect_retry_times", &ContextDesc::connect_retry_times,
                     "connect to mesh retry time")
      .def_readwrite("connect_retry_interval_ms",
                     &ContextDesc::connect_retry_interval_ms)
      .def_readwrite("recv_timeout_ms", &ContextDesc::recv_timeout_ms)
      .def_readwrite("http_max_payload_size",
                     &ContextDesc::http_max_payload_size)
      .def_readwrite("http_timeout_ms", &ContextDesc::http_timeout_ms)
      .def_readwrite("brpc_channel_protocol",
                     &ContextDesc::brpc_channel_protocol)
      .def_readwrite("brpc_channel_connection_type",
                     &ContextDesc::brpc_channel_connection_type)
      .def_readwrite("throttle_window_size", &ContextDesc::throttle_window_size)
      .def_readwrite("enable_ssl", &ContextDesc::enable_ssl)
      .def_readwrite("client_ssl_opts", &ContextDesc::client_ssl_opts)
      .def_readwrite("server_ssl_opts", &ContextDesc::server_ssl_opts)
      .def_readwrite("link_type", &ContextDesc::link_type)
      .def_readwrite("retry_opts", &ContextDesc::retry_opts)
      .def_readwrite("disable_msg_seq_id", &ContextDesc::disable_msg_seq_id)
      .def(
          "add_party",
          [](ContextDesc& desc, std::string id, std::string host) {
            desc.parties.push_back({std::move(id), std::move(host)});
          },
          "add a party to the link");

  // expose shared_ptr<Context> to py
  py::class_<Context, std::shared_ptr<Context>>(m, "Context", "the link handle")
      .def("__repr__",
           [](const Context* self) {
             return fmt::format("Link(id={}, rank={}/{})", self->Id(),
                                self->Rank(), self->WorldSize());
           })
      .def(
          "id", [](const Context* self) { return self->Id(); },
          "the unique link id")
      .def_property_readonly(
          "rank", [](const Context* self) { return self->Rank(); },
          py::return_value_policy::copy, "my rank of the link")
      .def_property_readonly(
          "world_size", [](const Context* self) { return self->WorldSize(); },
          py::return_value_policy::copy, "the number of parties")
      .def(
          "spawn",
          [](const std::shared_ptr<Context>& self) {
            return std::shared_ptr<Context>(self->Spawn());
          },
          NO_GIL, "spawn a sub-link, advanced skill")
      .def(
          "barrier",
          [&PY_CALL_TAG](const std::shared_ptr<Context>& self) -> void {
            return yacl::link::Barrier(self, PY_CALL_TAG);
          },
          NO_GIL,
          "Blocks until all parties have reached this routine, aka MPI_Barrier")
      .def(
          "all_gather",
          [&PY_CALL_TAG](const std::shared_ptr<Context>& self,
                         const std::string& in) -> std::vector<std::string> {
            auto bufs = yacl::link::AllGather(self, in, PY_CALL_TAG);
            std::vector<std::string> ret(bufs.size());
            for (size_t idx = 0; idx < bufs.size(); ++idx) {
              ret[idx] = std::string(bufs[idx].data<char>(), bufs[idx].size());
            }
            return ret;
          },
          NO_GIL,
          "Gathers data from all parties and distribute the combined data to "
          "all parties, aka MPI_AllGather")
      .def(
          "gather",
          [&PY_CALL_TAG](const std::shared_ptr<Context>& self,
                         const std::string& in,
                         size_t root) -> std::vector<std::string> {
            auto bufs = yacl::link::Gather(self, in, root, PY_CALL_TAG);
            std::vector<std::string> ret(bufs.size());
            for (size_t idx = 0; idx < bufs.size(); ++idx) {
              ret[idx] = std::string(bufs[idx].data<char>(), bufs[idx].size());
            }
            return ret;
          },
          NO_GIL, "Gathers values from other parties, aka MPI_Gather")
      .def(
          "broadcast",
          [&PY_CALL_TAG](const std::shared_ptr<Context>& self,
                         const std::string& in, size_t root) -> std::string {
            auto buf = yacl::link::Broadcast(self, in, root, PY_CALL_TAG);
            return {buf.data<char>(), static_cast<size_t>(buf.size())};
          },
          NO_GIL,
          "Broadcasts a message from the party with rank 'root' to all other "
          "parties, aka MPI_Bcast")
      .def(
          "stop_link",
          [](const std::shared_ptr<Context>& self) -> void {
            return self->WaitLinkTaskFinish();
          },
          NO_GIL, "Blocks until all link is safely stoped")
      .def(
          "scatter",
          [&PY_CALL_TAG](const std::shared_ptr<Context>& self,
                         const std::vector<std::string>& in,
                         size_t root) -> std::string {
            auto buf = yacl::link::Scatter(self, {in.begin(), in.end()}, root,
                                           PY_CALL_TAG);
            return {buf.data<char>(), static_cast<size_t>(buf.size())};
          },
          NO_GIL,
          "Sends data from one party to all other parties, aka MPI_Scatter")
      .def(
          "send",
          [&PY_CALL_TAG](const std::shared_ptr<Context>& self, size_t dst_rank,
                         const std::string& in) {
            self->Send(dst_rank, in, PY_CALL_TAG);
          },
          NO_GIL, "Sends data to dst_rank")
      .def(
          "send_async",
          [&PY_CALL_TAG](const std::shared_ptr<Context>& self, size_t dst_rank,
                         const std::string& in) {
            self->SendAsync(dst_rank, yacl::Buffer(in), PY_CALL_TAG);
          },
          NO_GIL, "Sends data to dst_rank asynchronously")
      .def(
          "recv",
          [&PY_CALL_TAG](const std::shared_ptr<Context>& self,
                         size_t src_rank) -> py::bytes {
            py::gil_scoped_release release;
            yacl::Buffer buf = self->Recv(src_rank, PY_CALL_TAG);
            py::gil_scoped_acquire acquire;
            return py::bytes{buf.data<char>(), static_cast<size_t>(buf.size())};
          },  // Since it uses py bytes, we cannot release GIL here
          "Receives data from src_rank")
      .def(
          "next_rank",
          [](const std::shared_ptr<Context>& self, size_t strides = 1) {
            return self->NextRank(strides);
          },
          NO_GIL, "Gets next party rank", py::arg("strides") = 1);

  m.def(
      "create_brpc",
      [](const ContextDesc& desc, size_t self_rank,
         bool log_details) -> std::shared_ptr<Context> {
        py::gil_scoped_release release;
        brpc::FLAGS_max_body_size = std::numeric_limits<uint64_t>::max();
        brpc::FLAGS_socket_max_unwritten_bytes =
            std::numeric_limits<int64_t>::max() / 2;

        auto ctx = yacl::link::FactoryBrpc().CreateContext(desc, self_rank);
        ctx->ConnectToMesh(log_details ? spdlog::level::info
                                       : spdlog::level::debug);
        return ctx;
      },
      py::arg("desc"), py::arg("self_rank"), py::kw_only(),
      py::arg("log_details") = false);

  m.def("create_mem",
        [](const ContextDesc& desc,
           size_t self_rank) -> std::shared_ptr<Context> {
          py::gil_scoped_release release;

          auto ctx = yacl::link::FactoryMem().CreateContext(desc, self_rank);
          ctx->ConnectToMesh();
          return ctx;
        });
  m.def(
      "create_with_channels",
      [](const ContextDesc& desc, size_t self_rank,
         std::vector<std::shared_ptr<IChannel>> channels) {
        py::gil_scoped_release release;
        auto ctx = std::make_shared<yacl::link::Context>(
            desc, self_rank, std::move(channels), nullptr, false);
        ctx->ConnectToMesh();
        return ctx;
      },
      py::arg("desc"), py::arg("self_rank"), py::arg("channels"));
}

struct PyBindShare {
  py::bytes meta;
  std::vector<py::bytes> share_chunks;
};

static spu::Value ValueFromPyBindShare(const PyBindShare& py_share) {
  spu::ValueProto value;
  pb::ValueMetaProto meta;
  SPU_ENFORCE(meta.ParseFromString(py_share.meta));
  value.meta.Swap(&meta);
  for (const auto& s : py_share.share_chunks) {
    pb::ValueChunkProto chunk;
    SPU_ENFORCE(chunk.ParseFromString(s));
    value.chunks.emplace_back(std::move(chunk));
  }
  return Value::fromProto(value);
}

static PyBindShare ValueToPyBindShare(const spu::Value& value,
                                      size_t max_chunk_size) {
  PyBindShare ret;

  const auto value_pb = value.toProto(max_chunk_size);
  ret.meta = value_pb.meta.SerializeAsString();
  ret.share_chunks.reserve(value_pb.chunks.size());
  for (const auto& s : value_pb.chunks) {
    ret.share_chunks.emplace_back(s.SerializeAsString());
  }
  return ret;
}

// Wrap Runtime, it's workaround for protobuf pybind11/protoc conflict.
class RuntimeWrapper {
  std::unique_ptr<spu::SPUContext> sctx_;

  // the globals, could be used to cross session stuffs.
  spu::device::SymbolTable env_;

  size_t max_chunk_size_;

 public:
  explicit RuntimeWrapper(const std::shared_ptr<yacl::link::Context>& lctx,
                          const RuntimeConfig& config) {
    // first, fill protobuf default value with implementation defined value.
    // populateRuntimeConfig(config);

    sctx_ = std::make_unique<spu::SPUContext>(config, lctx);
    mpc::Factory::RegisterProtocol(sctx_.get(), lctx);
    max_chunk_size_ = config.share_max_chunk_size;
    if (max_chunk_size_ == 0) {
      max_chunk_size_ = 128UL * 1024 * 1024;
    }
  }

  void Run(const spu::ExecutableProto& exec) {
    spu::device::pphlo::PPHloExecutor executor;
    spu::device::execute(&executor, sctx_.get(), exec, &env_);
  }

  void SetVar(const std::string& name, const PyBindShare& share) {
    env_.setVar(name, ValueFromPyBindShare(share));
  }

  PyBindShare GetVar(const std::string& name) const {
    return ValueToPyBindShare(env_.getVar(name), max_chunk_size_);
  }

  size_t GetVarChunksCount(const std::string& name) {
    return env_.getVar(name).chunksCount(max_chunk_size_);
  }

  pb::ValueMetaProto GetVarMeta(const std::string& name) const {
    return env_.getVar(name).toMetaProto();
  }

  void DelVar(const std::string& name) { env_.delVar(name); }

  void Clear() { env_.clear(); }
};

// numpy type naming:
// https://numpy.org/doc/stable/reference/arrays.scalars.html#sized-aliases
#define FOR_PY_FORMATS(FN) \
  FN("int8", PT_I8)        \
  FN("int16", PT_I16)      \
  FN("int32", PT_I32)      \
  FN("int64", PT_I64)      \
  FN("uint8", PT_U8)       \
  FN("uint16", PT_U16)     \
  FN("uint32", PT_U32)     \
  FN("uint64", PT_U64)     \
  FN("float16", PT_F16)    \
  FN("float32", PT_F32)    \
  FN("float64", PT_F64)    \
  FN("bool", PT_I1)        \
  FN("complex64", PT_CF32) \
  FN("complex128", PT_CF64)

// https://docs.python.org/3/library/struct.html#format-characters
// https://numpy.org/doc/stable/reference/arrays.scalars.html#built-in-scalar-types
// Note: python and numpy has different type string, here pybind11 uses numpy's
// definition
spu::PtType PyFormatToPtType(const std::string& format) {
#define CASE(FORMAT, PT_TYPE) \
  if (format == (FORMAT)) return PT_TYPE;

  if (false) {  // NOLINT: macro trick
  }
  FOR_PY_FORMATS(CASE)

#undef CASE
  SPU_THROW("unknown py format={}", format);
}

std::string PtTypeToPyFormat(spu::PtType pt_type) {
#define CASE(FORMAT, PT_TYPE) \
  if (pt_type == (PT_TYPE)) return FORMAT;

  if (false) {  // NOLINT: macro trick
  }
  FOR_PY_FORMATS(CASE)

#undef CASE
  SPU_THROW("unknown pt_type={}", pt_type);
}

template <typename Iter>
Strides ByteToElementStrides(const Iter& begin, const Iter& end,
                             size_t elsize) {
  Strides ret(std::distance(begin, end));
  std::transform(begin, end, ret.begin(), [&](int64_t c) -> int64_t {
    SPU_ENFORCE(c % elsize == 0);
    return c / elsize;
  });
  return ret;
}

constexpr void SizeCheck() {
  static_assert(sizeof(intptr_t) == 8, "SPU only supports 64-bit system");
  static_assert(sizeof(long long) == 8,  // NOLINT
                "SPU assumes size of longlong == 8");
  static_assert(sizeof(unsigned long long) == 8,  // NOLINT
                "SPU assumes size of ulonglong == 8");
}

class IoWrapper {
 private:
  std::unique_ptr<spu::device::IoClient> ptr_;
  size_t max_chunk_size_;

 public:
  IoWrapper(size_t world_size, const spu::RuntimeConfig& config) {
    ptr_ = std::make_unique<spu::device::IoClient>(world_size, config);
    max_chunk_size_ = config.share_max_chunk_size;
    if (max_chunk_size_ == 0) {
      max_chunk_size_ = 128UL * 1024 * 1024;
    }
  }

  size_t GetShareChunkCount(const py::array& arr, int visibility,
                            int owner_rank) {
    const py::buffer_info& binfo = arr.request();
    const PtType pt_type = PyFormatToPtType(py::str(arr.dtype()));

    spu::PtBufferView view(
        binfo.ptr, pt_type, Shape(binfo.shape.begin(), binfo.shape.end()),
        ByteToElementStrides(binfo.strides.begin(), binfo.strides.end(),
                             binfo.itemsize));
    const size_t share_size = ptr_->getShareSize(
        view, static_cast<spu::Visibility>(visibility), owner_rank);
    size_t num_chunks = (share_size + max_chunk_size_ - 1) / max_chunk_size_;
    return num_chunks;
  }

  std::vector<PyBindShare> MakeShares(const py::array& arr, int visibility,
                                      int owner_rank = -1) {
    // When working with Python, do a static size check, this has no runtime
    // cost
    SizeCheck();

    const PtType pt_type = PyFormatToPtType(py::str(arr.dtype()));

    const py::buffer_info& binfo = arr.request();

    spu::PtBufferView view(
        binfo.ptr, pt_type, Shape(binfo.shape.begin(), binfo.shape.end()),
        ByteToElementStrides(binfo.strides.begin(), binfo.strides.end(),
                             binfo.itemsize));

    const auto shares = ptr_->makeShares(
        view, static_cast<spu::Visibility>(visibility), owner_rank);

    std::vector<PyBindShare> serialized;
    serialized.reserve(shares.size());
    for (const auto& share : shares) {
      serialized.emplace_back(ValueToPyBindShare(share, max_chunk_size_));
    }

    return serialized;
  }

  py::array Reconstruct(const std::vector<PyBindShare>& vals) {
    std::vector<spu::Value> shares;
    SPU_ENFORCE(!vals.empty());
    shares.reserve(vals.size());
    for (const auto& val : vals) {
      shares.emplace_back(ValueFromPyBindShare(val));
    }
    // sanity
    for (size_t idx = 1; idx < shares.size(); ++idx) {
      const auto& cur = shares[idx];
      const auto& prev = shares[idx - 1];
      SPU_ENFORCE(cur.storage_type() == prev.storage_type(),
                  "storage type mismatch, {} {}", cur.storage_type(),
                  prev.storage_type());
      SPU_ENFORCE(cur.dtype() == prev.dtype(), "data type mismatch, {} {}",
                  cur.dtype(), prev.dtype());
    }

    const PtType pt_type = ptr_->getPtType(shares);
    std::vector<size_t> shape = {shares.front().shape().begin(),
                                 shares.front().shape().end()};

    py::array ret(py::dtype(PtTypeToPyFormat(pt_type)), shape);
    const py::buffer_info& binfo = ret.request();

    spu::PtBufferView ret_view(
        binfo.ptr, pt_type, Shape(binfo.shape.begin(), binfo.shape.end()),
        ByteToElementStrides(binfo.strides.begin(), binfo.strides.end(),
                             binfo.itemsize));

    ptr_->combineShares(shares, &ret_view);
    return ret;
  }
};

void BindSPU(py::module& m) {
  m.doc() = R"pbdoc(
        SPU Library
            )pbdoc";

  // bind enum
  py::enum_<DataType>(m, "DataType")
      .value("DT_INVALID", DataType::DT_INVALID)
      .value("DT_I1", DataType::DT_I1)
      .value("DT_I8", DataType::DT_I8)
      .value("DT_U8", DataType::DT_U8)
      .value("DT_I16", DataType::DT_I16)
      .value("DT_U16", DataType::DT_U16)
      .value("DT_I32", DataType::DT_I32)
      .value("DT_U32", DataType::DT_U32)
      .value("DT_I64", DataType::DT_I64)
      .value("DT_U64", DataType::DT_U64)
      .value("DT_F16", DataType::DT_F16)
      .value("DT_F32", DataType::DT_F32)
      .value("DT_F64", DataType::DT_F64)
      .export_values();

  py::enum_<Visibility>(m, "Visibility")
      .value("VIS_INVALID", Visibility::VIS_INVALID)
      .value("VIS_PUBLIC", Visibility::VIS_PUBLIC)
      .value("VIS_SECRET", Visibility::VIS_SECRET)
      .value("VIS_PRIVATE", Visibility::VIS_PRIVATE)
      .export_values();

  py::enum_<FieldType>(m, "FieldType")
      .value("FT_INVALID", FieldType::FT_INVALID)
      .value("FM32", FieldType::FM32)
      .value("FM64", FieldType::FM64)
      .value("FM128", FieldType::FM128)
      .export_values();

  py::enum_<ProtocolKind>(m, "ProtocolKind")
      .value("PROT_INVALID", ProtocolKind::PROT_INVALID)
      .value("REF2K", ProtocolKind::REF2K)
      .value("SEMI2K", ProtocolKind::SEMI2K)
      .value("ABY3", ProtocolKind::ABY3)
      .value("CHEETAH", ProtocolKind::CHEETAH)
      .value("SECURENN", ProtocolKind::SECURENN)
      // .value("SWIFT", ProtocolKind::SWIFT)
      .export_values();

  // bind RuntimeConfig
  py::class_<ClientSSLConfig>(m, "ClientSSLConfig")
      .def(py::init<>())
      .def(py::init<std::string, std::string, std::string, int32_t>(),
           py::arg("certificate") = "", py::arg("private_key") = "",
           py::arg("ca_file_path") = "", py::arg("verify_depth") = 0)
      .def_readwrite("certificate", &ClientSSLConfig::certificate)
      .def_readwrite("private_key", &ClientSSLConfig::private_key)
      .def_readwrite("ca_file_path", &ClientSSLConfig::ca_file_path)
      .def_readwrite("verify_depth", &ClientSSLConfig::verify_depth);

  py::class_<TTPBeaverConfig>(m, "TTPBeaverConfig")
      .def(py::init<>())
      .def(py::init<std::string, int32_t, std::string, std::string, std::string,
                    std::shared_ptr<ClientSSLConfig>>(),
           py::arg("server_host") = "", py::arg("adjust_rank") = 0,
           py::arg("asym_crypto_schema") = "",
           py::arg("server_public_key") = "",
           py::arg("transport_protocol") = "", py::arg("ssl_config") = nullptr)
      .def_readwrite("server_host", &TTPBeaverConfig::server_host)
      .def_readwrite("adjust_rank", &TTPBeaverConfig::adjust_rank)
      .def_readwrite("asym_crypto_schema", &TTPBeaverConfig::asym_crypto_schema)
      .def_readwrite("server_public_key", &TTPBeaverConfig::server_public_key)
      .def_readwrite("transport_protocol", &TTPBeaverConfig::transport_protocol)
      .def_readwrite("ssl_config", &TTPBeaverConfig::ssl_config);

  py::class_<CheetahConfig>(m, "CheetahConfig")
      .def(py::init<>())
      .def(py::init<bool, bool, CheetahOtKind>(),
           py::arg("disable_matmul_pack") = false,
           py::arg("enable_mul_lsb_error") = false, py::arg("ot_kind") = 0)
      .def_readwrite("disable_matmul_pack", &CheetahConfig::disable_matmul_pack)
      .def_readwrite("enable_mul_lsb_error",
                     &CheetahConfig::enable_mul_lsb_error)
      .def_readwrite("ot_kind", &CheetahConfig::ot_kind);

  py::class_<RuntimeConfig> rt_cls(m, "RuntimeConfig");

  py::enum_<RuntimeConfig::SortMethod>(rt_cls, "SortMethod")
      .value("SORT_DEFAULT", RuntimeConfig::SORT_DEFAULT)
      .value("SORT_RADIX", RuntimeConfig::SORT_RADIX)
      .value("SORT_QUICK", RuntimeConfig::SORT_QUICK)
      .value("SORT_NETWORK", RuntimeConfig::SORT_NETWORK)
      .export_values();

  py::enum_<RuntimeConfig::ExpMode>(rt_cls, "ExpMode")
      .value("EXP_DEFAULT", RuntimeConfig::EXP_DEFAULT)
      .value("EXP_PADE", RuntimeConfig::EXP_PADE)
      .value("EXP_TAYLOR", RuntimeConfig::EXP_TAYLOR)
      .value("EXP_PRIME", RuntimeConfig::EXP_PRIME)
      .export_values();

  py::enum_<RuntimeConfig::LogMode>(rt_cls, "LogMode")
      .value("LOG_DEFAULT", RuntimeConfig::LOG_DEFAULT)
      .value("LOG_PADE", RuntimeConfig::LOG_PADE)
      .value("LOG_NEWTON", RuntimeConfig::LOG_NEWTON)
      .value("LOG_MINMAX", RuntimeConfig::LOG_MINMAX)
      .export_values();

  py::enum_<RuntimeConfig::SigmoidMode>(rt_cls, "SigmoidMode")
      .value("SIGMOID_DEFAULT", RuntimeConfig::SIGMOID_DEFAULT)
      .value("SIGMOID_MM1", RuntimeConfig::SIGMOID_MM1)
      .value("SIGMOID_SEG3", RuntimeConfig::SIGMOID_SEG3)
      .value("SIGMOID_REAL", RuntimeConfig::SIGMOID_REAL)
      .export_values();

  py::enum_<RuntimeConfig::BeaverType>(rt_cls, "BeaverType")
      .value("TrustedFirstParty", RuntimeConfig::TrustedFirstParty)
      .value("TrustedThirdParty", RuntimeConfig::TrustedThirdParty)
      .value("MultiParty", RuntimeConfig::MultiParty)
      .export_values();

  rt_cls.def(py::init<>())
      .def(py::init<ProtocolKind, FieldType, int64_t>(), py::arg("protocol"),
           py::arg("field"), py::arg("fxp_fraction_bits") = 0)
      .def(py::init<const RuntimeConfig&>())
      .def("ParseFromJsonString", &RuntimeConfig::ParseFromJsonString)
      .def("ParseFromString",
           [](RuntimeConfig& self, py::bytes data) {
             std::string str_data = data.cast<std::string>();
             return self.ParseFromString(str_data);
           })
      .def("SerializeToString",
           [](const RuntimeConfig& self) {
             return py::bytes(self.SerializeAsString());
           })
      .def("__str__", &RuntimeConfig::DumpToString)
      .def_readwrite("protocol", &RuntimeConfig::protocol)
      .def_readwrite("field", &RuntimeConfig::field)
      .def_readwrite("fxp_fraction_bits", &RuntimeConfig::fxp_fraction_bits)
      .def_readwrite("max_concurrency", &RuntimeConfig::max_concurrency)
      .def_readwrite("enable_action_trace", &RuntimeConfig::enable_action_trace)
      .def_readwrite("enable_type_checker", &RuntimeConfig::enable_type_checker)
      .def_readwrite("enable_pphlo_trace", &RuntimeConfig::enable_pphlo_trace)
      .def_readwrite("enable_runtime_snapshot",
                     &RuntimeConfig::enable_runtime_snapshot)
      .def_readwrite("snapshot_dump_dir", &RuntimeConfig::snapshot_dump_dir)
      .def_readwrite("enable_pphlo_profile",
                     &RuntimeConfig::enable_pphlo_profile)
      .def_readwrite("enable_hal_profile", &RuntimeConfig::enable_hal_profile)
      .def_readwrite("public_random_seed", &RuntimeConfig::public_random_seed)
      .def_readwrite("share_max_chunk_size",
                     &RuntimeConfig::share_max_chunk_size)
      .def_readwrite("sort_method", &RuntimeConfig::sort_method)
      .def_readwrite("quick_sort_threshold",
                     &RuntimeConfig::quick_sort_threshold)
      .def_readwrite("fxp_div_goldschmidt_iters",
                     &RuntimeConfig::fxp_div_goldschmidt_iters)
      .def_readwrite("fxp_exp_mode", &RuntimeConfig::fxp_exp_mode)
      .def_readwrite("fxp_exp_iters", &RuntimeConfig::fxp_exp_iters)
      .def_readwrite("fxp_log_mode", &RuntimeConfig::fxp_log_mode)
      .def_readwrite("fxp_log_iters", &RuntimeConfig::fxp_log_iters)
      .def_readwrite("fxp_log_orders", &RuntimeConfig::fxp_log_orders)
      .def_readwrite("sigmoid_mode", &RuntimeConfig::sigmoid_mode)
      .def_readwrite("enable_lower_accuracy_rsqrt",
                     &RuntimeConfig::enable_lower_accuracy_rsqrt)
      .def_readwrite("sine_cosine_iters", &RuntimeConfig::sine_cosine_iters)
      .def_readwrite("beaver_type", &RuntimeConfig::beaver_type)
      .def_readwrite("ttp_beaver_config", &RuntimeConfig::ttp_beaver_config)
      .def_readwrite("cheetah_2pc_config", &RuntimeConfig::cheetah_2pc_config)
      .def_readwrite("trunc_allow_msb_error",
                     &RuntimeConfig::trunc_allow_msb_error)
      .def_readwrite("experimental_disable_mmul_split",
                     &RuntimeConfig::experimental_disable_mmul_split)
      .def_readwrite("experimental_enable_inter_op_par",
                     &RuntimeConfig::experimental_enable_inter_op_par)
      .def_readwrite("experimental_enable_intra_op_par",
                     &RuntimeConfig::experimental_enable_intra_op_par)
      .def_readwrite("experimental_disable_vectorization",
                     &RuntimeConfig::experimental_disable_vectorization)
      .def_readwrite("experimental_inter_op_concurrency",
                     &RuntimeConfig::experimental_inter_op_concurrency)
      .def_readwrite("experimental_enable_colocated_optimization",
                     &RuntimeConfig::experimental_enable_colocated_optimization)
      .def_readwrite("experimental_enable_exp_prime",
                     &RuntimeConfig::experimental_enable_exp_prime)
      .def_readwrite("experimental_exp_prime_offset",
                     &RuntimeConfig::experimental_exp_prime_offset)
      .def_readwrite("experimental_exp_prime_disable_lower_bound",
                     &RuntimeConfig::experimental_exp_prime_disable_lower_bound)
      .def_readwrite("experimental_exp_prime_enable_upper_bound",
                     &RuntimeConfig::experimental_exp_prime_enable_upper_bound)
      .def(py::pickle(
          [](const RuntimeConfig& self) {
            return py::bytes(self.SerializeAsString());
          },
          [](py::bytes data) {
            RuntimeConfig res;
            res.ParseFromString(std::string(data));
            return res;
          }));

  // Compiler
  py::enum_<SourceIRType>(m, "SourceIRType")
      .value("XLA", SourceIRType::XLA)
      .value("STABLEHLO", SourceIRType::STABLEHLO)
      .export_values();

  py::class_<CompilationSource>(m, "CompilationSource")
      .def(py::init<>())
      .def(py::init<SourceIRType, std::string, std::vector<Visibility>>(),
           py::arg("ir_type") = SourceIRType::XLA, py::arg("ir_txt") = "",
           py::arg("input_visibility") = py::list())
      .def("__hash__",
           [](const CompilationSource& self) {
             return std::hash<CompilationSource>{}(self);
           })
      .def("__eq__",
           [](const CompilationSource& self, const CompilationSource& other) {
             return self == other;
           })
      .def_readwrite("ir_type", &CompilationSource::ir_type)
      .def_property(
          "ir_txt",
          [](const CompilationSource& self) { return py::bytes(self.ir_txt); },
          [](CompilationSource& self, const py::bytes& bytes) {
            self.ir_txt = std::string(bytes);
          })
      .def_readwrite("input_visibility", &CompilationSource::input_visibility);

  py::enum_<XLAPrettyPrintKind>(m, "XLAPrettyPrintKind")
      .value("TEXT", XLAPrettyPrintKind::TEXT)
      .value("DOT", XLAPrettyPrintKind::DOT)
      .value("HTML", XLAPrettyPrintKind::HTML)
      .export_values();

  py::class_<CompilerOptions>(m, "CompilerOptions")
      .def(py::init<>())
      .def(py::init<bool, std::string, XLAPrettyPrintKind, bool, bool, bool,
                    bool, bool, bool, bool, bool, bool>(),
           py::arg("enable_pretty_print") = false,
           py::arg("pretty_print_dump_dir") = "",
           py::arg("xla_pp_kind") = XLAPrettyPrintKind::TEXT,
           py::arg("disable_sqrt_plus_epsilon_rewrite") = false,
           py::arg("disable_div_sqrt_rewrite") = false,
           py::arg("disable_reduce_truncation_optimization") = false,
           py::arg("disable_maxpooling_optimization") = false,
           py::arg("disallow_mix_types_opts") = false,
           py::arg("disable_select_optimization") = false,
           py::arg("enable_optimize_denominator_with_broadcast") = false,
           py::arg("disable_deallocation_insertion") = false,
           py::arg("disable_partial_sort_optimization") = false)
      .def("__hash__",
           [](const CompilerOptions& self) {
             return std::hash<spu::CompilerOptions>{}(self);
           })
      .def("__eq__", [](const CompilerOptions& self,
                        const CompilerOptions& other) { return self == other; })
      .def_readwrite("enable_pretty_print",
                     &CompilerOptions::enable_pretty_print)
      .def_readwrite("pretty_print_dump_dir",
                     &CompilerOptions::pretty_print_dump_dir)
      .def_readwrite("xla_pp_kind", &CompilerOptions::xla_pp_kind)
      .def_readwrite("disable_sqrt_plus_epsilon_rewrite",
                     &CompilerOptions::disable_sqrt_plus_epsilon_rewrite)
      .def_readwrite("disable_div_sqrt_rewrite",
                     &CompilerOptions::disable_div_sqrt_rewrite)
      .def_readwrite("disable_reduce_truncation_optimization",
                     &CompilerOptions::disable_reduce_truncation_optimization)
      .def_readwrite("disable_maxpooling_optimization",
                     &CompilerOptions::disable_maxpooling_optimization)
      .def_readwrite("disallow_mix_types_opts",
                     &CompilerOptions::disallow_mix_types_opts)
      .def_readwrite("disable_select_optimization",
                     &CompilerOptions::disable_select_optimization)
      .def_readwrite(
          "enable_optimize_denominator_with_broadcast",
          &CompilerOptions::enable_optimize_denominator_with_broadcast)
      .def_readwrite("disable_deallocation_insertion",
                     &CompilerOptions::disable_deallocation_insertion)
      .def_readwrite("disable_partial_sort_optimization",
                     &CompilerOptions::disable_partial_sort_optimization)
      .def(py::pickle(
          [](const CompilerOptions& self) {
            return py::bytes(self.SerializeAsString());
          },
          [](py::bytes data) {
            CompilerOptions res;
            res.ParseFromString(std::string(data));
            return res;
          }));

  py::class_<ExecutableProto>(m, "Executable")
      .def(py::init<>())
      .def(py::init<std::string, std::vector<std::string>,
                    std::vector<std::string>, std::string>(),
           py::arg("name") = "", py::arg("input_names") = py::list(),
           py::arg("output_names") = py::list(), py::arg("code") = py::bytes())
      .def_readwrite("name", &ExecutableProto::name)
      .def_readwrite("input_names", &ExecutableProto::input_names)
      .def_readwrite("output_names", &ExecutableProto::output_names)
      .def_property(
          "code",
          [](const ExecutableProto& self) { return py::bytes(self.code); },
          [](ExecutableProto& self, const py::bytes& bytes) {
            self.code = std::string(bytes);
          })
      .def("ParseFromString",
           [](ExecutableProto& self, py::bytes data) {
             std::string str_data = data.cast<std::string>();
             return self.ParseFromString(str_data);
           })
      .def("SerializeToString",
           [](const ExecutableProto& self) {
             return py::bytes(self.SerializeAsString());
           })
      .def(py::pickle(
          [](const ExecutableProto& self) {
            return py::bytes(self.SerializeAsString());
          },
          [](py::bytes data) {
            ExecutableProto proto;
            proto.ParseFromString(std::string(data));
            return proto;
          }));

  py::class_<pb::ShapeProto>(m, "Shape")
      .def(py::init<>())
      .def_property(
          "dims",
          [](const pb::ShapeProto& self) {
            return std::vector<int64_t>(self.dims().begin(), self.dims().end());
          },
          [](pb::ShapeProto& self, const std::vector<int64_t>& dims) {
            self.mutable_dims()->Clear();
            for (int64_t dim : dims) {
              self.add_dims(dim);
            }
          });

  py::class_<pb::ValueMetaProto>(m, "ValueMeta")
      .def(py::init<>())
      .def_property(
          "data_type",
          [](const pb::ValueMetaProto& self) {
            return DataType(self.data_type());
          },
          [](pb::ValueMetaProto& self, const DataType& data_type) {
            self.set_data_type(pb::DataType(data_type));
          })
      .def_property("is_complex", &pb::ValueMetaProto::is_complex,
                    &pb::ValueMetaProto::set_is_complex)
      .def_property(
          "visibility",
          [](const pb::ValueMetaProto& self) {
            return Visibility(self.visibility());
          },
          [](pb::ValueMetaProto& self, const Visibility& visibility) {
            self.set_visibility(pb::Visibility(visibility));
          })
      .def_property(
          "shape", &pb::ValueMetaProto::shape,
          [](pb::ValueMetaProto& self, const std::vector<int64_t>& dims) {
            auto shape = self.mutable_shape();
            shape->mutable_dims()->Clear();
            for (int64_t dim : dims) {
              shape->add_dims(dim);
            }
          })
      .def_property("storage_type", &pb::ValueMetaProto::storage_type,
                    [](pb::ValueMetaProto& self, const std::string type) {
                      self.set_storage_type(type);
                    })
      .def("ParseFromString",
           [](pb::ValueMetaProto& self, py::bytes data) {
             std::string str_data = data.cast<std::string>();
             return self.ParseFromString(str_data);
           })
      .def("SerializeToString", [](const pb::ValueMetaProto& self) {
        return py::bytes(self.SerializeAsString());
      });

  py::class_<pb::ValueChunkProto>(m, "ValueChunk")
      .def(py::init<>())
      .def_property("total_bytes", &pb::ValueChunkProto::total_bytes,
                    &pb::ValueChunkProto::set_total_bytes)
      .def_property("chunk_offset", &pb::ValueChunkProto::chunk_offset,
                    &pb::ValueChunkProto::set_chunk_offset)
      .def_property(
          "content",
          [](const pb::ValueChunkProto& self) {
            return py::bytes(self.content());
          },
          [](pb::ValueChunkProto& self, const py::bytes& bytes) {
            self.set_content(std::string(bytes));
          })
      .def("ParseFromString",
           [](pb::ValueChunkProto& self, py::bytes data) {
             std::string str_data = data.cast<std::string>();
             return self.ParseFromString(str_data);
           })
      .def("SerializeToString", [](const pb::ValueChunkProto& self) {
        return py::bytes(self.SerializeAsString());
      });

  py::class_<PyBindShare>(m, "Share", "Share in python runtime")
      .def(py::init<>())
      .def_readwrite("share_chunks", &PyBindShare::share_chunks, "share chunks")
      .def_readwrite("meta", &PyBindShare::meta, "meta of share")
      .def(py::pickle(
          [](const PyBindShare& s) {  // dump
            return py::make_tuple(s.meta, s.share_chunks);
          },
          [](const py::tuple& t) {  // load
            return PyBindShare{t[0].cast<py::bytes>(),
                               t[1].cast<std::vector<py::bytes>>()};
          }));

  // bind spu virtual machine.
  py::class_<RuntimeWrapper>(m, "RuntimeWrapper", "SPU virtual device")
      .def(py::init<std::shared_ptr<yacl::link::Context>,
                    const spu::RuntimeConfig&>(),
           NO_GIL)
      .def("Run", &RuntimeWrapper::Run, NO_GIL)
      .def("SetVar",
           &RuntimeWrapper::
               SetVar)  // https://github.com/pybind/pybind11/issues/1782
                        // SetVar & GetVar are using
                        // py::byte, so they must acquire gil...
      .def("GetVar", &RuntimeWrapper::GetVar)
      .def("GetVarChunksCount", &RuntimeWrapper::GetVarChunksCount)
      .def("GetVarMeta", &RuntimeWrapper::GetVarMeta)
      .def("DelVar", &RuntimeWrapper::DelVar);

  // bind spu io suite.
  py::class_<IoWrapper>(m, "IoWrapper", "SPU VM IO")
      .def(py::init<size_t, const spu::RuntimeConfig&>())
      .def("MakeShares", &IoWrapper::MakeShares, "Create secret shares",
           py::arg("arr"), py::arg("visibility"), py::arg("owner_rank") = -1)
      .def("GetShareChunkCount", &IoWrapper::GetShareChunkCount, py::arg("arr"),
           py::arg("visibility"), py::arg("owner_rank") = -1)
      .def("Reconstruct", &IoWrapper::Reconstruct);

  // bind compiler.
  m.def(
      "compile",
      [](const spu::CompilationSource& source,
         const spu::CompilerOptions& copts) {
        py::scoped_ostream_redirect stream(
            std::cout,                                 // std::ostream&
            py::module_::import("sys").attr("stdout")  // Python output
        );
        return py::bytes(spu::compiler::compile(source, copts));
      },
      "spu compile.", py::arg("source"), py::arg("copts"));
}

void BindLogging(py::module& m) {
  m.doc() = R"pbdoc(
              SPU Logging Library
                  )pbdoc";

  py::enum_<logging::LogLevel>(m, "LogLevel")
      .value("DEBUG", logging::LogLevel::Debug)
      .value("INFO", logging::LogLevel::Info)
      .value("WARN", logging::LogLevel::Warn)
      .value("ERROR", logging::LogLevel::Error);

  py::class_<logging::LogOptions>(m, "LogOptions",
                                  "options for setup spu logger")
      .def(py::init<>())
      .def_readwrite("enable_console_logger",
                     &logging::LogOptions::enable_console_logger)
      .def_readwrite("system_log_path", &logging::LogOptions::system_log_path)
      .def_readwrite("trace_log_path", &logging::LogOptions::trace_log_path)
      .def_readwrite("log_level", &logging::LogOptions::log_level)
      .def_readwrite("max_log_file_size",
                     &logging::LogOptions::max_log_file_size)
      .def_readwrite("max_log_file_count",
                     &logging::LogOptions::max_log_file_count)
      .def_readwrite("trace_content_length",
                     &logging::LogOptions::trace_content_length)
      .def(py::pickle(
          [](const logging::LogOptions& opts) {  // __getstate__
            /* Return a tuple that fully encodes the state of the object */
            return py::make_tuple(
                opts.enable_console_logger, opts.system_log_path,
                opts.trace_log_path, opts.log_level, opts.max_log_file_size,
                opts.max_log_file_count, opts.trace_content_length);
          },
          [](const py::tuple& t) {  // __setstate__
            if (t.size() != 7) {
              throw std::runtime_error("Invalid serialized data!");
            }

            /* Create a new C++ instance */
            logging::LogOptions opts = logging::LogOptions();
            opts.enable_console_logger = t[0].cast<bool>();
            opts.system_log_path = t[1].cast<std::string>();
            opts.trace_log_path = t[2].cast<std::string>();
            opts.log_level = t[3].cast<logging::LogLevel>();
            opts.max_log_file_size = t[4].cast<size_t>();
            opts.max_log_file_count = t[5].cast<size_t>();
            opts.trace_content_length = t[6].cast<size_t>();

            return opts;
          }));
  ;

  m.def(
      "setup_logging",
      [](const logging::LogOptions& options) -> void {
        logging::SetupLogging(options);
      },
      py::arg("options") = logging::LogOptions(), NO_GIL);
}

PYBIND11_MODULE(libspu, m) {
  py::register_exception_translator(
      [](std::exception_ptr p) {  // NOLINT: pybind11
        try {
          if (p) {
            std::rethrow_exception(p);
          }
        } catch (const yacl::Exception& e) {
          // Translate this exception to a standard RuntimeError
          PyErr_SetString(PyExc_RuntimeError,
                          fmt::format("what: \n\t{}\nstacktrace: \n{}\n",
                                      e.what(), e.stack_trace())
                              .c_str());
        }
      });

  BindSPU(m);

  // bind spu libs.
  py::module link_m = m.def_submodule("link");
  BindLink(link_m);

  py::module logging_m = m.def_submodule("logging");
  BindLogging(logging_m);

  // bind check cpu features
  m.def("_check_cpu_features", []() {
#ifdef CHECK_AVX
    static const auto cpu_features = cpu_features::GetX86Info().features;
    if (!cpu_features.avx) {
      throw std::runtime_error(FormatMissingCpuFeatureMsg("AVX"));
    }
    if (!cpu_features.aes) {
      throw std::runtime_error(FormatMissingCpuFeatureMsg("AES"));
    }
#endif
  });

  m.def("_get_version", []() { return spu::getVersionStr(); });

  // Expose buffer conversion functions
  m.def("buffer_to_array", &BufferToArray,
        "Convert yacl::Buffer to numpy array", py::arg("buffer"));
}

}  // namespace spu
