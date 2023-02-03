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

#include "fmt/format.h"
#include "pybind11/iostream.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "yacl/link/link.h"

#include "libspu/compiler/common/compilation_context.h"
#include "libspu/compiler/compile.h"
#include "libspu/core/logging.h"
#include "libspu/core/type_util.h"
#include "libspu/device/api.h"
#include "libspu/device/io.h"
#include "libspu/device/pphlo/pphlo_executor.h"
#include "libspu/kernel/context.h"
#include "libspu/kernel/value.h"
#include "libspu/psi/bucket_psi.h"
#include "libspu/psi/core/ecdh_psi.h"
#include "libspu/psi/memory_psi.h"

namespace py = pybind11;

namespace brpc {

DECLARE_uint64(max_body_size);
DECLARE_int64(socket_max_unwritten_bytes);

}  // namespace brpc

namespace spu {

#define NO_GIL py::call_guard<py::gil_scoped_release>()

void BindLink(py::module& m) {
  using yacl::link::Context;
  using yacl::link::ContextDesc;

  // TODO(jint) expose this tag to python?
  constexpr char PY_CALL_TAG[] = "PY_CALL";

  m.doc() = R"pbdoc(
              SPU Link Library
                  )pbdoc";

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
      .def_readwrite("connect_retry_times", &ContextDesc::connect_retry_times)
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
      .def(
          "add_party",
          [](ContextDesc& desc, std::string id, std::string host) {
            desc.parties.push_back({id, host});
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
          "all parties, aka MPI_Allgather")
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
          "Sends data to dst_rank")
      .def(
          "send_async",
          [&PY_CALL_TAG](const std::shared_ptr<Context>& self, size_t dst_rank,
                         const std::string& in) {
            self->SendAsync(dst_rank, yacl::Buffer(in), PY_CALL_TAG);
          },
          "Sends data to dst_rank asynchronously")
      .def(
          "recv",
          [&PY_CALL_TAG](const std::shared_ptr<Context>& self,
                         size_t src_rank) -> py::bytes {
            auto buf = self->Recv(src_rank, PY_CALL_TAG);
            return py::bytes{buf.data<char>(), static_cast<size_t>(buf.size())};
          },  // Since it uses py bytes, we cannot release GIL here
          "Receives data from src_rank")
      .def(
          "next_rank",
          [](const std::shared_ptr<Context>& self, size_t strides = 1) {
            return self->NextRank(strides);
          },
          NO_GIL, "Gets next party rank", py::arg("strides") = 1);

  m.def("create_brpc",
        [](const ContextDesc& desc,
           size_t self_rank) -> std::shared_ptr<Context> {
          py::gil_scoped_release release;
          brpc::FLAGS_max_body_size = std::numeric_limits<uint64_t>::max();
          brpc::FLAGS_socket_max_unwritten_bytes =
              std::numeric_limits<int64_t>::max() / 2;
          auto ctx = yacl::link::FactoryBrpc().CreateContext(desc, self_rank);
          ctx->ConnectToMesh();
          return ctx;
        });

  m.def("create_mem",
        [](const ContextDesc& desc,
           size_t self_rank) -> std::shared_ptr<Context> {
          py::gil_scoped_release release;

          auto ctx = yacl::link::FactoryMem().CreateContext(desc, self_rank);
          ctx->ConnectToMesh();
          return ctx;
        });
}

// Wrap Runtime, it's workaround for protobuf pybind11/protoc conflict.
class RuntimeWrapper {
  std::unique_ptr<spu::HalContext> hctx_;

  // the golbals, could be used to cross session stuffs.
  spu::device::SymbolTable env_;

 public:
  explicit RuntimeWrapper(std::shared_ptr<yacl::link::Context> lctx,
                          const std::string& config_pb) {
    spu::RuntimeConfig config;
    YACL_ENFORCE(config.ParseFromString(config_pb));

    hctx_ = std::make_unique<spu::HalContext>(config, lctx);
  }

  void Run(const py::bytes& exec_pb) {
    spu::ExecutableProto exec;
    YACL_ENFORCE(exec.ParseFromString(exec_pb));

    spu::device::pphlo::PPHloExecutor executor;
    spu::device::execute(&executor, hctx_.get(), exec, &env_);
  }

  void SetVar(const std::string& name, const py::bytes& value) {
    ValueProto proto;
    YACL_ENFORCE(proto.ParseFromString(value));

    env_.setVar(name, spu::Value::fromProto(proto));
  }

  py::bytes GetVar(const std::string& name) const {
    return env_.getVar(name).toProto().SerializeAsString();
  }

  py::bytes GetVarMeta(const std::string& name) const {
    return env_.getVar(name).toMetaProto().SerializeAsString();
  }

  void DelVar(const std::string& name) { env_.delVar(name); }

  void Clear() { env_.clear(); }
};

#define FOR_PY_FORMATS(FN) \
  FN("b", PT_I8)           \
  FN("h", PT_I16)          \
  FN("i", PT_I32)          \
  FN("l", PT_I64)          \
  FN("q", PT_I64)          \
  FN("B", PT_U8)           \
  FN("H", PT_U16)          \
  FN("I", PT_U32)          \
  FN("L", PT_U64)          \
  FN("Q", PT_U64)          \
  FN("f", PT_F32)          \
  FN("d", PT_F64)          \
  FN("?", PT_BOOL)

// https://docs.python.org/3/library/struct.html#format-characters
// https://numpy.org/doc/stable/reference/arrays.scalars.html#built-in-scalar-types
// Note: python and numpy has different type string, here pybind11 uses numpy's
// definition
spu::PtType PyFormatToPtType(const std::string& format) {
#define CASE(FORMAT, PT_TYPE) \
  if (format == FORMAT) return PT_TYPE;

  if (false) {
  }
  FOR_PY_FORMATS(CASE)

#undef CASE
  YACL_THROW("unknown py format={}", format);
}

std::string PtTypeToPyFormat(spu::PtType pt_type) {
#define CASE(FORMAT, PT_TYPE) \
  if (pt_type == PT_TYPE) return FORMAT;

  if (false) {
  }
  FOR_PY_FORMATS(CASE)

#undef CASE
  YACL_THROW("unknown pt_type={}", pt_type);
}

template <typename Iter>
std::vector<int64_t> ByteToElementStrides(const Iter& begin, const Iter& end,
                                          size_t elsize) {
  std::vector<int64_t> ret(std::distance(begin, end));
  std::transform(begin, end, ret.begin(), [&](int64_t c) -> int64_t {
    YACL_ENFORCE(c % elsize == 0);
    return c / elsize;
  });
  return ret;
}

constexpr void SizeCheck() {
  static_assert(sizeof(intptr_t) == 8, "SPU only supports 64-bit system");
  static_assert(sizeof(long long) == 8, "SPU assumes size of longlong == 8");
  static_assert(sizeof(unsigned long long) == 8,
                "SPU assumes size of ulonglong == 8");
}

class IoWrapper {
  std::unique_ptr<spu::device::IoClient> ptr_;

 public:
  IoWrapper(size_t world_size, const std::string& config_pb) {
    spu::RuntimeConfig config;
    YACL_ENFORCE(config.ParseFromString(config_pb));

    ptr_ = std::make_unique<spu::device::IoClient>(world_size, config);
  }

  std::vector<py::bytes> MakeShares(const py::array& arr, int visibility,
                                    int owner_rank = -1) {
    // When working with Python, do a static size check, this has no runtime
    // cost
    SizeCheck();

    const py::buffer_info& binfo = arr.request();
    const PtType pt_type = PyFormatToPtType(binfo.format);

    spu::PtBufferView view(
        binfo.ptr, pt_type,
        std::vector<int64_t>(binfo.shape.begin(), binfo.shape.end()),
        ByteToElementStrides(binfo.strides.begin(), binfo.strides.end(),
                             binfo.itemsize));

    auto shares =
        ptr_->makeShares(view, spu::Visibility(visibility), owner_rank);
    std::vector<py::bytes> serialized(shares.size());
    for (size_t idx = 0; idx < shares.size(); ++idx) {
      std::string s;
      YACL_ENFORCE(shares[idx].toProto().SerializeToString(&s));
      serialized[idx] = py::bytes(s);
    }

    return serialized;
  }

  py::array reconstruct(const std::vector<std::string>& vals) {
    std::vector<spu::Value> shares;
    YACL_ENFORCE(vals.size() > 0);
    for (const auto& val_str : vals) {
      spu::ValueProto vp;
      YACL_ENFORCE(vp.ParseFromString(val_str));
      shares.push_back(spu::Value::fromProto(vp));
    }

    // sanity
    for (size_t idx = 1; idx < vals.size(); ++idx) {
      const auto& cur = shares[idx];
      const auto& prev = shares[idx - 1];
      YACL_ENFORCE(cur.storage_type() == prev.storage_type(),
                   "storage type mismatch, {} {}", cur.storage_type(),
                   prev.storage_type());
      YACL_ENFORCE(cur.dtype() == prev.dtype(), "data type mismatch, {} {}",
                   cur.dtype(), prev.dtype());
    }

    auto ndarr = ptr_->combineShares(shares);
    YACL_ENFORCE(ndarr.eltype().isa<PtTy>(), "expect decode to pt_type, got {}",
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

  m.def(
      "mem_psi",
      [](const std::shared_ptr<yacl::link::Context>& lctx,
         const std::string& config_pb,
         const std::vector<std::string>& items) -> std::vector<std::string> {
        psi::MemoryPsiConfig config;
        YACL_ENFORCE(config.ParseFromString(config_pb));

        psi::MemoryPsi psi(config, lctx);
        return psi.Run(items);
      },
      NO_GIL);

  m.def(
      "bucket_psi",
      [](const std::shared_ptr<yacl::link::Context>& lctx,
         const std::string& config_pb, bool ic_mode) -> py::bytes {
        psi::BucketPsiConfig config;
        YACL_ENFORCE(config.ParseFromString(config_pb));

        psi::BucketPsi psi(config, lctx, ic_mode);
        auto r = psi.Run();
        return r.SerializeAsString();
      },
      py::arg("link_context"), py::arg("psi_config"),
      py::arg("ic_mode") = false,
      "Run bucket psi. ic_mode means run in interconnection mode");
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
      .def_readwrite("log_level", &logging::LogOptions::log_level)
      .def_readwrite("max_log_file_size",
                     &logging::LogOptions::max_log_file_size)
      .def_readwrite("max_log_file_count",
                     &logging::LogOptions::max_log_file_count)
      .def(py::pickle(
          [](const logging::LogOptions& opts) {  // __getstate__
            /* Return a tuple that fully encodes the state of the object */
            return py::make_tuple(opts.enable_console_logger,
                                  opts.system_log_path, opts.log_level,
                                  opts.max_log_file_size,
                                  opts.max_log_file_count);
          },
          [](py::tuple t) {  // __setstate__
            if (t.size() != 5) {
              throw std::runtime_error("Invalid serialized data!");
            }

            /* Create a new C++ instance */
            logging::LogOptions opts = logging::LogOptions();
            opts.enable_console_logger = t[0].cast<bool>();
            opts.system_log_path = t[1].cast<std::string>();
            opts.log_level = t[2].cast<logging::LogLevel>();
            opts.max_log_file_size = t[3].cast<size_t>();
            opts.max_log_file_count = t[4].cast<size_t>();

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
  py::register_exception_translator([](std::exception_ptr p) {
    try {
      if (p) {
        std::rethrow_exception(p);
      }
    } catch (const yacl::Exception& e) {
      // Translate this exception to a standard RuntimeError
      PyErr_SetString(PyExc_RuntimeError,
                      fmt::format("what: \n\t{}\nstacktrace: \n{}\n", e.what(),
                                  e.stack_trace())
                          .c_str());
    }
  });

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
      .def("GetVarMeta", &RuntimeWrapper::GetVarMeta)
      .def("DelVar", &RuntimeWrapper::DelVar);

  // bind spu io suite.
  py::class_<IoWrapper>(m, "IoWrapper", "SPU VM IO")
      .def(py::init<size_t, std::string>())
      .def("MakeShares", &IoWrapper::MakeShares)
      .def("Reconstruct", &IoWrapper::reconstruct);

  // bind compiler.
  m.def(
      "compile",
      [](const py::bytes& ir_text, const std::string& ir_type,
         const std::string& input_visbility_map, const std::string& dump_path) {
        py::scoped_ostream_redirect stream(
            std::cout,                                 // std::ostream&
            py::module_::import("sys").attr("stdout")  // Python output
        );

        spu::compiler::CompilationContext ctx;
        ctx.setInputVisibilityString(input_visbility_map);

        if (!dump_path.empty()) {
          ctx.enablePrettyPrintWithDir(dump_path);
        }

        return py::bytes(spu::compiler::compile(&ctx, ir_text, ir_type));
      },
      "spu compile.", py::arg("ir_text"), py::arg("ir_type"),
      py::arg("vis_map"), py::arg("dump_path"));

  // bind spu libs.
  py::module link_m = m.def_submodule("link");
  BindLink(link_m);

  py::module libs_m = m.def_submodule("libs");
  BindLibs(libs_m);

  py::module logging_m = m.def_submodule("logging");
  BindLogging(logging_m);
}

}  // namespace spu
