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

// The fxp accuracy module.

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>

#include "pybind11/iostream.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "spdlog/spdlog.h"
#include "yasl/base/exception.h"

#include "spu/hal/simulation/fxp_sim.h"

namespace py = pybind11;

namespace spu::hal::simulation {

#define NO_GIL py::call_guard<py::gil_scoped_release>()

PYBIND11_MODULE(fxp_acc, m) {
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

  m.def(
      "exp",
      [](const std::vector<float>& x) {
        py::scoped_ostream_redirect stream(
            std::cout,                                 // std::ostream&
            py::module_::import("sys").attr("stdout")  // Python output
        );

        // suppress all link logs.
        spdlog::set_level(spdlog::level::off);

        return exp_sim(x, VIS_SECRET);
      },
      "exp.", py::arg("x"));

  m.def(
      "reciprocal",
      [](const std::vector<float>& x) {
        py::scoped_ostream_redirect stream(
            std::cout,                                 // std::ostream&
            py::module_::import("sys").attr("stdout")  // Python output
        );

        // suppress all link logs.
        spdlog::set_level(spdlog::level::off);

        return reciprocal_sim(x, VIS_SECRET);
      },
      "reciprocal.", py::arg("x"));

  m.def(
      "log",
      [](const std::vector<float>& x) {
        py::scoped_ostream_redirect stream(
            std::cout,                                 // std::ostream&
            py::module_::import("sys").attr("stdout")  // Python output
        );

        // suppress all link logs.
        spdlog::set_level(spdlog::level::off);

        return log_sim(x, VIS_SECRET);
      },
      "log.", py::arg("x"));

  m.def(
      "logistic",
      [](const std::vector<float>& x) {
        py::scoped_ostream_redirect stream(
            std::cout,                                 // std::ostream&
            py::module_::import("sys").attr("stdout")  // Python output
        );

        // suppress all link logs.
        spdlog::set_level(spdlog::level::off);

        return logistic_sim(x, VIS_SECRET);
      },
      "logistic.", py::arg("x"));

  m.def(
      "div",
      [](const std::vector<float>& x, const std::vector<float>& y) {
        py::scoped_ostream_redirect stream(
            std::cout,                                 // std::ostream&
            py::module_::import("sys").attr("stdout")  // Python output
        );

        // suppress all link logs.
        spdlog::set_level(spdlog::level::off);

        return div_sim(x, VIS_SECRET, y, VIS_SECRET);
      },
      "div.", py::arg("x"), py::arg("y"));
}

}  // namespace spu::hal::simulation
