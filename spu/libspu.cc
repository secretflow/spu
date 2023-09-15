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
#include <utility>

#include "fmt/format.h"
#include "pybind11/complex.h"
#include "pybind11/functional.h"
#include "pybind11/iostream.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "yacl/link/link.h"

#include "libspu/compiler/common/compilation_context.h"
#include "libspu/compiler/compile.h"
#include "libspu/core/config.h"
#include "libspu/core/context.h"
#include "libspu/core/logging.h"
#include "libspu/core/type_util.h"
#include "libspu/core/value.h"
#include "libspu/device/api.h"
#include "libspu/device/io.h"
#include "libspu/device/pphlo/pphlo_executor.h"
#include "libspu/mpc/factory.h"
#include "libspu/pir/pir.h"
#include "libspu/psi/bucket_psi.h"
#include "libspu/psi/core/ecdh_psi.h"
#include "libspu/psi/memory_psi.h"

#ifdef CHECK_AVX
#include "cpu_features/cpuinfo_x86.h"
#endif

namespace py = pybind11;

namespace brpc {

DECLARE_uint64(max_body_size);
DECLARE_int64(socket_max_unwritten_bytes);

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

#define NO_GIL py::call_guard<py::gil_scoped_release>()

void BindLink(py::module& m) {
  using yacl::link::CertInfo;
  using yacl::link::Context;
  using yacl::link::ContextDesc;
  using yacl::link::SSLOptions;
  using yacl::link::VerifyOptions;

  // TODO(jint) expose this tag to python?
  constexpr char PY_CALL_TAG[] = "PY_CALL";

  m.doc() = R"pbdoc(
              SPU Link Library
                  )pbdoc";

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
      .def_readwrite("brpc_retry_count", &ContextDesc::brpc_retry_count)
      .def_readwrite("brpc_retry_interval_ms",
                     &ContextDesc::brpc_retry_interval_ms)
      .def_readwrite("brpc_aggressive_retry",
                     &ContextDesc::brpc_aggressive_retry)
      .def_readwrite("link_type", &ContextDesc::link_type)
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
}

struct PyBindShare {
  py::bytes meta;
  std::vector<py::bytes> share_chunks;
};

static spu::Value ValueFromPyBindShare(const PyBindShare& py_share) {
  spu::ValueProto value;
  spu::ValueMetaProto meta;
  SPU_ENFORCE(meta.ParseFromString(py_share.meta));
  value.meta.Swap(&meta);
  for (const auto& s : py_share.share_chunks) {
    spu::ValueChunkProto chunk;
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
                          const std::string& config_pb) {
    spu::RuntimeConfig config;
    SPU_ENFORCE(config.ParseFromString(config_pb));

    // first, fill protobuf default value with implementation defined value.
    populateRuntimeConfig(config);

    sctx_ = std::make_unique<spu::SPUContext>(config, lctx);
    mpc::Factory::RegisterProtocol(sctx_.get(), lctx);
    max_chunk_size_ = config.share_max_chunk_size();
    if (max_chunk_size_ == 0) {
      max_chunk_size_ = 128UL * 1024 * 1024;
    }
  }

  void Run(const py::bytes& exec_pb) {
    spu::ExecutableProto exec;
    SPU_ENFORCE(exec.ParseFromString(exec_pb));

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

  py::bytes GetVarMeta(const std::string& name) const {
    return env_.getVar(name).toMetaProto().SerializeAsString();
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
  FN("bool", PT_BOOL)      \
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
  IoWrapper(size_t world_size, const std::string& config_pb) {
    spu::RuntimeConfig config;
    SPU_ENFORCE(config.ParseFromString(config_pb));

    ptr_ = std::make_unique<spu::device::IoClient>(world_size, config);
    max_chunk_size_ = config.share_max_chunk_size();
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

    auto ndarr = ptr_->combineShares(shares);
    SPU_ENFORCE(ndarr.eltype().isa<PtTy>(), "expect decode to pt_type, got {}",
                ndarr.eltype());

    const auto pt_type = ndarr.eltype().as<PtTy>()->pt_type();
    std::vector<size_t> shape = {ndarr.shape().begin(), ndarr.shape().end()};
    return py::array(py::dtype(PtTypeToPyFormat(pt_type)), shape, ndarr.data());
  }
};

void BindLibs(py::module& m) {
  m.doc() = R"pbdoc(
              SPU Mixed Library
                  )pbdoc";

  py::class_<psi::Progress::Data>(m, "ProgressData", "The progress data")
      .def(py::init<>())
      .def_readonly("total", &psi::Progress::Data::total,
                    "the number of all subjobs")
      .def_readonly("finished", &psi::Progress::Data::finished,
                    "the number of finished subjobs")
      .def_readonly("running", &psi::Progress::Data::running,
                    "the number of running subjobs")
      .def_readonly("percentage", &psi::Progress::Data::percentage,
                    "the percentage of the task progress")
      .def_readonly("description", &psi::Progress::Data::description,
                    "description of the current running subjob");

  m.def(
      "mem_psi",
      [](const std::shared_ptr<yacl::link::Context>& lctx,
         const std::string& config_pb,
         const std::vector<std::string>& items) -> std::vector<std::string> {
        psi::MemoryPsiConfig config;
        SPU_ENFORCE(config.ParseFromString(config_pb));

        psi::MemoryPsi psi(config, lctx);
        return psi.Run(items);
      },
      NO_GIL);

  m.def(
      "bucket_psi",
      [](const std::shared_ptr<yacl::link::Context>& lctx,
         const std::string& config_pb,
         psi::ProgressCallbacks progress_callbacks,
         int64_t callbacks_interval_ms, bool ic_mode) -> py::bytes {
        psi::BucketPsiConfig config;
        SPU_ENFORCE(config.ParseFromString(config_pb));

        psi::BucketPsi psi(config, lctx, ic_mode);
        auto r = psi.Run(progress_callbacks, callbacks_interval_ms);
        return r.SerializeAsString();
      },
      py::arg("link_context"), py::arg("psi_config"),
      py::arg("progress_callbacks") = nullptr,
      py::arg("callbacks_interval_ms") = 5 * 1000, py::arg("ic_mode") = false,
      "Run bucket psi. ic_mode means run in interconnection mode", NO_GIL);

  m.def(
      "pir_setup",
      [](const std::string& config_pb) -> py::bytes {
        pir::PirSetupConfig config;
        SPU_ENFORCE(config.ParseFromString(config_pb));

        config.set_bucket_size(1000000);
        config.set_compressed(false);

        auto r = pir::PirSetup(config);
        return r.SerializeAsString();
      },
      py::arg("pir_config"), "Run pir setup.");

  m.def(
      "pir_server",
      [](const std::shared_ptr<yacl::link::Context>& lctx,
         const std::string& config_pb) -> py::bytes {
        pir::PirServerConfig config;
        SPU_ENFORCE(config.ParseFromString(config_pb));

        auto r = pir::PirServer(lctx, config);
        return r.SerializeAsString();
      },
      py::arg("link_context"), py::arg("pir_config"), "Run pir server");

  m.def(
      "pir_memory_server",
      [](const std::shared_ptr<yacl::link::Context>& lctx,
         const std::string& config_pb) -> py::bytes {
        pir::PirSetupConfig config;
        SPU_ENFORCE(config.ParseFromString(config_pb));
        SPU_ENFORCE(config.setup_path() == "::memory");

        config.set_bucket_size(1000000);
        config.set_compressed(false);

        auto r = pir::PirMemoryServer(lctx, config);
        return r.SerializeAsString();
      },
      py::arg("link_context"), py::arg("pir_config"), "Run pir memory server");

  m.def(
      "pir_client",
      [](const std::shared_ptr<yacl::link::Context>& lctx,
         const std::string& config_pb) -> py::bytes {
        pir::PirClientConfig config;
        SPU_ENFORCE(config.ParseFromString(config_pb));

        auto r = pir::PirClient(lctx, config);
        return r.SerializeAsString();
      },
      py::arg("link_context"), py::arg("pir_config"), "Run pir client");
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

  py::class_<PyBindShare>(m, "Share", "Share in python runtime")
      .def(py::init<>(), NO_GIL)
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
      .def(py::init<std::shared_ptr<yacl::link::Context>, std::string>(),
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
      .def(py::init<size_t, std::string>())
      .def("MakeShares", &IoWrapper::MakeShares)
      .def("GetShareChunkCount", &IoWrapper::GetShareChunkCount)
      .def("Reconstruct", &IoWrapper::Reconstruct);

  // bind compiler.
  m.def(
      "compile",
      [](const py::bytes& source, const std::string& copts) {
        py::scoped_ostream_redirect stream(
            std::cout,                                 // std::ostream&
            py::module_::import("sys").attr("stdout")  // Python output
        );

        spu::compiler::CompilationContext ctx;
        ctx.setCompilerOptions(copts);

        return py::bytes(spu::compiler::compile(&ctx, source));
      },
      "spu compile.", py::arg("source"), py::arg("copts"));

  // bind spu libs.
  py::module link_m = m.def_submodule("link");
  BindLink(link_m);

  py::module libs_m = m.def_submodule("libs");
  BindLibs(libs_m);

  py::module logging_m = m.def_submodule("logging");
  BindLogging(logging_m);

  // bind check cpu features
  m.def(
      "check_cpu_features",
      []() {
#ifdef CHECK_AVX
        static const auto cpu_features = cpu_features::GetX86Info().features;
        if (!cpu_features.avx) {
          throw std::runtime_error(FormatMissingCpuFeatureMsg("AVX"));
        }
        if (!cpu_features.aes) {
          throw std::runtime_error(FormatMissingCpuFeatureMsg("AES"));
        }
#endif
      },
      "check cpu features");
}

}  // namespace spu
