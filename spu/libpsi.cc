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

#include "pybind11/functional.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "yacl/base/exception.h"
#include "yacl/link/context.h"

#include "psi/launch.h"
#include "psi/legacy/memory_psi.h"
#include "psi/utils/progress.h"

#include "psi/proto/pir.pb.h"
#include "psi/proto/psi.pb.h"
#include "psi/proto/psi_v2.pb.h"

namespace py = pybind11;

#define NO_GIL py::call_guard<py::gil_scoped_release>()

namespace psi {

void BindLibs(py::module& m) {
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
        YACL_ENFORCE(config.ParseFromString(config_pb));

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
        YACL_ENFORCE(config.ParseFromString(config_pb));

        psi::BucketPsi psi(config, lctx, ic_mode);
        auto r = psi.Run(std::move(progress_callbacks), callbacks_interval_ms);
        return r.SerializeAsString();
      },
      py::arg("link_context"), py::arg("psi_config"),
      py::arg("progress_callbacks") = nullptr,
      py::arg("callbacks_interval_ms") = 5 * 1000, py::arg("ic_mode") = false,
      "Run bucket psi. ic_mode means run in interconnection mode", NO_GIL);

  m.def(
      "psi",
      [](const std::string& config_pb,
         const std::shared_ptr<yacl::link::Context>& lctx) -> py::bytes {
        psi::v2::PsiConfig psi_config;
        YACL_ENFORCE(psi_config.ParseFromString(config_pb));

        auto report = psi::RunPsi(psi_config, lctx);
        return report.SerializeAsString();
      },
      py::arg("psi_config"), py::arg("link_context"), "Run PSI with v2 API.",
      NO_GIL);

  m.def(
      "ub_psi",
      [](const std::string& config_pb,
         const std::shared_ptr<yacl::link::Context>& lctx) -> py::bytes {
        psi::v2::UbPsiConfig ub_psi_config;
        YACL_ENFORCE(ub_psi_config.ParseFromString(config_pb));

        auto report = psi::RunUbPsi(ub_psi_config, lctx);
        return report.SerializeAsString();
      },
      py::arg("ub_psi_config"), py::arg("link_context") = nullptr,
      "Run UB PSI with v2 API.", NO_GIL);

  m.def(
      "pir",
      [](const std::string& config_pb,
         const std::shared_ptr<yacl::link::Context>& lctx) -> py::bytes {
        psi::PirConfig config;
        YACL_ENFORCE(config.ParseFromString(config_pb));

        auto r = psi::RunPir(config, lctx);
        return r.SerializeAsString();
      },
      py::arg("pir_config"), py::arg("link_context") = nullptr, "Run PIR.");
}

PYBIND11_MODULE(libpsi, m) {
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

  py::module libs_m = m.def_submodule("libs");
  BindLibs(libs_m);
}

}  // namespace psi
