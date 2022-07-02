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

#include "fmt/format.h"
#include "pybind11/iostream.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "yasl/link/link.h"

#include "spu/compiler/common/compilation_context.h"
#include "spu/compiler/compile.h"
#include "spu/core/type_util.h"
#include "spu/device/io.h"
#include "spu/device/pphlo_executor.h"
#include "spu/hal/context.h"
#include "spu/hal/value.h"
#include "spu/psi/core/ecdh_psi.h"
#include "spu/psi/psi.h"

namespace py = pybind11;

namespace spu {

#define NO_GIL py::call_guard<py::gil_scoped_release>()

void BindLink(py::module& m) {
  using yasl::link::Context;
  using yasl::link::ContextDesc;

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
            return yasl::link::Barrier(self, PY_CALL_TAG);
          },
          NO_GIL,
          "Blocks until all parties have reached this routine, aka MPI_Barrier")
      .def(
          "all_gather",
          [&PY_CALL_TAG](const std::shared_ptr<Context>& self,
                         const std::string& in) -> std::vector<std::string> {
            auto bufs = yasl::link::AllGather(self, in, PY_CALL_TAG);
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
            auto bufs = yasl::link::Gather(self, in, root, PY_CALL_TAG);
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
            auto buf = yasl::link::Broadcast(self, in, root, PY_CALL_TAG);
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
            auto buf = yasl::link::Scatter(self, {in.begin(), in.end()}, root,
                                           PY_CALL_TAG);
            return {buf.data<char>(), static_cast<size_t>(buf.size())};
          },
          NO_GIL,
          "Sends data from one party to all other parties, aka MPI_Scatter");

  m.def("create_brpc",
        [](const ContextDesc& desc,
           size_t self_rank) -> std::shared_ptr<Context> {
          py::gil_scoped_release release;
          auto ctx = yasl::link::FactoryBrpc().CreateContext(desc, self_rank);
          ctx->ConnectToMesh();
          return ctx;
        });

  m.def("create_mem",
        [](const ContextDesc& desc,
           size_t self_rank) -> std::shared_ptr<Context> {
          py::gil_scoped_release release;
          auto ctx = yasl::link::FactoryMem().CreateContext(desc, self_rank);
          ctx->ConnectToMesh();
          return ctx;
        });
}

// Wrap Processor, it's workaround for protobuf pybind11/protoc conflict.
class RuntimeWrapper {
  std::unique_ptr<spu::HalContext> hctx_;

  // the golbals, could be used to cross session stuffs.
  spu::device::SymbolTable env_;

 public:
  explicit RuntimeWrapper(std::shared_ptr<yasl::link::Context> lctx,
                          const std::string& config_pb) {
    spu::RuntimeConfig config;
    YASL_ENFORCE(config.ParseFromString(config_pb));

    hctx_ = std::make_unique<spu::HalContext>(config, lctx);
  }

  void Run(const py::bytes& exec_pb) {
    spu::ExecutableProto exec;
    YASL_ENFORCE(exec.ParseFromString(exec_pb));

    spu::device::PPHloExecutor executor(hctx_.get());
    executor.runWithEnv(exec, &env_);
  }

  void SetVar(const std::string& name, const py::bytes& value) {
    ValueProto proto;
    YASL_ENFORCE(proto.ParseFromString(value));
    env_.setVar(name, hal::Value::fromProto(proto));
  }

  py::bytes GetVar(const std::string& name) const {
    return env_.getVar(name).toProto().SerializeAsString();
  }

  void DelVar(const std::string& name) { env_.delVar(name); }
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
  YASL_THROW("unknown py format={}", format);
}

std::string PtTypeToPyFormat(spu::PtType pt_type) {
#define CASE(FORMAT, PT_TYPE) \
  if (pt_type == PT_TYPE) return FORMAT;

  if (false) {
  }
  FOR_PY_FORMATS(CASE)

#undef CASE
  YASL_THROW("unknown pt_type={}", pt_type);
}

template <typename Iter>
std::vector<int64_t> ByteToElementStrides(const Iter& begin, const Iter& end,
                                          size_t elsize) {
  std::vector<int64_t> ret(std::distance(begin, end));
  std::transform(begin, end, ret.begin(), [&](int64_t c) -> int64_t {
    YASL_ENFORCE(c % elsize == 0);
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
    YASL_ENFORCE(config.ParseFromString(config_pb));

    ptr_ = std::make_unique<spu::device::IoClient>(world_size, config);
  }

  std::vector<py::bytes> MakeShares(const py::array& arr, int visibility) {
    // When working with Python, do a sataic size check, this has no runtime
    // cost
    SizeCheck();

    const py::buffer_info& binfo = arr.request();
    const PtType pt_type = PyFormatToPtType(binfo.format);

    spu::PtBufferView view(
        binfo.ptr, pt_type,
        std::vector<int64_t>(binfo.shape.begin(), binfo.shape.end()),
        ByteToElementStrides(binfo.strides.begin(), binfo.strides.end(),
                             binfo.itemsize));

    auto shares = ptr_->makeShares(view, spu::Visibility(visibility));
    std::vector<py::bytes> serialized(shares.size());
    for (size_t idx = 0; idx < shares.size(); ++idx) {
      std::string s;
      YASL_ENFORCE(shares[idx].toProto().SerializeToString(&s));
      serialized[idx] = py::bytes(s);
    }

    return serialized;
  }

  py::array reconstruct(const std::vector<std::string>& vals) {
    std::vector<spu::hal::Value> shares;
    YASL_ENFORCE(vals.size() > 0);
    for (const auto& val_str : vals) {
      spu::ValueProto vp;
      YASL_ENFORCE(vp.ParseFromString(val_str));
      shares.push_back(spu::hal::Value::fromProto(vp));
    }

    // sanity
    for (size_t idx = 1; idx < vals.size(); ++idx) {
      const auto& cur = shares[idx];
      const auto& prev = shares[idx - 1];
      YASL_ENFORCE(cur.storage_type() == prev.storage_type(),
                   "storage type mismatch, {} {}", cur.storage_type(),
                   prev.storage_type());
      YASL_ENFORCE(cur.dtype() == prev.dtype(), "data type mismatch, {} {}",
                   cur.dtype(), prev.dtype());
    }

    auto ndarr = ptr_->combineShares(shares);
    YASL_ENFORCE(ndarr.eltype().isa<PtTy>(), "expect decode to pt_type, got {}",
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
      "ecdh_psi",
      [](const std::shared_ptr<yasl::link::Context>& lctx,
         const std::vector<std::string>& items,
         int64_t rank) -> std::vector<std::string> {
        // Sanity rank
        size_t target_rank = rank;
        if (rank == -1) {
          target_rank = yasl::link::kAllRank;
        } else if (rank < -1) {
          YASL_THROW("rank should be >= -1, got {}", rank);
        }
        return psi::RunEcdhPsi(lctx, items, target_rank);
      },
      NO_GIL);

  m.def(
      "ecdh_3pc_psi",
      [](const std::shared_ptr<yasl::link::Context>& lctx,
         const std::vector<std::string>& selected_fields,
         const std::string& input_path, const std::string& output_path,
         bool should_sort, psi::PsiReport* report) -> void {
        psi::LegacyPsiOptions psi_opts;
        psi_opts.base_options.link_ctx = lctx;
        psi_opts.base_options.field_names = selected_fields;
        psi_opts.base_options.in_path = input_path;
        psi_opts.base_options.out_path = output_path;
        psi_opts.base_options.should_sort = should_sort;
        psi_opts.psi_protocol = psi::kPsiProtocolEcdh;

        auto executor = psi::BuildPsiExecutor(psi_opts);
        executor->Init();
        executor->Run(report);
      },
      NO_GIL);

  m.def(
      "kkrt_2pc_psi",
      [](const std::shared_ptr<yasl::link::Context>& lctx,
         const std::vector<std::string>& selected_fields,
         const std::string& input_path, const std::string& output_path,
         bool should_sort, psi::PsiReport* report,
         bool broadcast_result) -> void {
        psi::LegacyPsiOptions psi_opts;
        psi_opts.base_options.link_ctx = lctx;
        psi_opts.base_options.field_names = selected_fields;
        psi_opts.base_options.in_path = input_path;
        psi_opts.base_options.out_path = output_path;
        psi_opts.base_options.should_sort = should_sort;
        psi_opts.psi_protocol = psi::kPsiProtocolKkrt;
        psi_opts.broadcast_result = broadcast_result;

        auto executor = psi::BuildPsiExecutor(psi_opts);
        executor->Init();
        executor->Run(report);
      },
      NO_GIL);

  m.def(
      "ecdh_2pc_psi",
      [](const std::shared_ptr<yasl::link::Context>& lctx,
         const std::vector<std::string>& selected_fields,
         const std::string& input_path, const std::string& output_path,
         size_t num_bins, bool should_sort, psi::PsiReport* report) -> void {
        psi::LegacyPsiOptions psi_opts;
        psi_opts.base_options.link_ctx = lctx;
        psi_opts.base_options.field_names = selected_fields;
        psi_opts.base_options.in_path = input_path;
        psi_opts.base_options.out_path = output_path;
        psi_opts.base_options.should_sort = should_sort;
        psi_opts.num_bins = num_bins;
        psi_opts.psi_protocol = psi::kPsiProtocolEcdh2PC;

        auto executor = psi::BuildPsiExecutor(psi_opts);
        executor->Init();
        executor->Run(report);
      },
      NO_GIL);
}

PYBIND11_MODULE(_lib, m) {
  py::register_exception_translator([](std::exception_ptr p) {
    try {
      if (p) {
        std::rethrow_exception(p);
      }
    } catch (const yasl::Exception& e) {
      // Translate this exception to a standard RuntimeError
      PyErr_SetString(PyExc_RuntimeError,
                      fmt::format("what: \n\t{}\nstacktrace: \n{}\n", e.what(),
                                  e.stack_trace())
                          .c_str());
    }
  });

  // bind spu virtual machine.
  py::class_<RuntimeWrapper>(m, "RuntimeWrapper", "SPU virtual device")
      .def(py::init<std::shared_ptr<yasl::link::Context>, std::string>(),
           NO_GIL)
      .def("Run", &RuntimeWrapper::Run, NO_GIL)
      .def("SetVar",
           &RuntimeWrapper::
               SetVar)  // https://github.com/pybind/pybind11/issues/1782
                        // SetVar & GetVar are using
                        // py::byte, so they must acquire gil...
      .def("GetVar", &RuntimeWrapper::GetVar)
      .def("DelVar", &RuntimeWrapper::DelVar);

  // bind spu io suite.
  py::class_<IoWrapper>(m, "IoWrapper", "SPU VM IO")
      .def(py::init<size_t, std::string>())
      .def("MakeShares", &IoWrapper::MakeShares)
      .def("Reconstruct", &IoWrapper::reconstruct);

  // bind compiler.
  // TODO: use type compile :: IrProto -> IrProto
  m.def(
      "compile",
      [](const py::bytes& hlo_text, const std::string& input_visbility_map,
         const std::string& dump_path) {
        py::scoped_ostream_redirect stream(
            std::cout,                                 // std::ostream&
            py::module_::import("sys").attr("stdout")  // Python output
        );

        spu::compiler::CompilationContext ctx;
        ctx.setInputVisibilityString(input_visbility_map);

        if (!dump_path.empty()) {
          ctx.enablePrettyPrintWithDir(dump_path);
        }

        return py::bytes(spu::compiler::compile(&ctx, hlo_text));
      },
      "spu compile.", py::arg("hlo_text"), py::arg("vis_map"),
      py::arg("dump_path"));

  // bind spu libs.
  py::module link_m = m.def_submodule("link");
  BindLink(link_m);

  py::module libs_m = m.def_submodule("libs");
  BindLibs(libs_m);

  py::class_<psi::PsiReport>(libs_m, "PsiReport")
      .def(py::init())
      .def_readwrite("intersection_count", &psi::PsiReport::intersection_count)
      .def_readwrite("original_count", &psi::PsiReport::original_count);
}

}  // namespace spu
