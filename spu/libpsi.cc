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

#include "psi/apps/psi_launcher/launch.h"

namespace py = pybind11;

#define NO_GIL py::call_guard<py::gil_scoped_release>()

namespace psi::api {

void BindPsi(py::module& m) {
  py::class_<psi::Progress::Data>(m, "ProgressData", "The progress data")
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

  py::class_<ProgressParams>(m, "ProgressParams", "progress params")
      .def(py::init<psi::ProgressCallbacks, uint32_t>(),
           py::arg("hook") = nullptr, py::arg("interval_ms") = 5 * 1000)
      .def_readwrite("hook", &ProgressParams::hook)
      .def_readwrite("interval_ms", &ProgressParams::interval_ms);

  py::enum_<PsiProtocol>(m, "PsiProtocol")
      .value("PROTOCOL_UNSPECIFIED", PsiProtocol::PROTOCOL_UNSPECIFIED)
      .value("PROTOCOL_ECDH", PsiProtocol::PROTOCOL_ECDH)
      .value("PROTOCOL_KKRT", PsiProtocol::PROTOCOL_KKRT)
      .value("PROTOCOL_RR22", PsiProtocol::PROTOCOL_RR22)
      .value("PROTOCOL_ECDH_3PC", PsiProtocol::PROTOCOL_ECDH_3PC)
      .value("PROTOCOL_ECDH_NPC", PsiProtocol::PROTOCOL_ECDH_NPC)
      .value("PROTOCOL_KKRT_NPC", PsiProtocol::PROTOCOL_KKRT_NPC)
      .value("PROTOCOL_DP", PsiProtocol::PROTOCOL_DP)
      .export_values();

  py::enum_<psi::api::EllipticCurveType>(m, "EllipticCurveType")
      .value("CURVE_INVALID_TYPE",
             psi::api::EllipticCurveType::CURVE_INVALID_TYPE)
      .value("CURVE_25519", psi::api::EllipticCurveType::CURVE_25519)
      .value("CURVE_FOURQ", psi::api::EllipticCurveType::CURVE_FOURQ)
      .value("CURVE_SM2", psi::api::EllipticCurveType::CURVE_SM2)
      .value("CURVE_SECP256K1", psi::api::EllipticCurveType::CURVE_SECP256K1)
      .value("CURVE_25519_ELLIGATOR2",
             psi::api::EllipticCurveType::CURVE_25519_ELLIGATOR2)
      .export_values();

  py::class_<EcdhParams>(m, "EcdhParams", "ecdh params")
      .def(py::init<EllipticCurveType, uint32_t>(),
           py::arg("curve") = EllipticCurveType::CURVE_INVALID_TYPE,
           py::arg("batch_size") = 4096)
      .def_readwrite("curve", &EcdhParams::curve)
      .def_readwrite("batch_size", &EcdhParams::batch_size);

  py::class_<Rr22Rarams>(m, "Rr22Rarams", "rr22 rarams")
      .def(py::init<bool>(), py::arg("low_comm_mode") = false)
      .def_readwrite("low_comm_mode", &Rr22Rarams::low_comm_mode);

  py::class_<DpParams>(m, "DpParams", "dp params")
      .def(py::init<double, double>(), py::arg("bob_sub_sampling") = 0.9,
           py::arg("epsilon") = 3.0)
      .def_readwrite("bob_sub_sampling", &DpParams::bob_sub_sampling)
      .def_readwrite("epsilon", &DpParams::epsilon);

  py::class_<PsiProtocolConfig>(m, "PsiProtocolConfig", "psi protocol config")
      .def(py::init<PsiProtocol, uint32_t, uint64_t, bool, EcdhParams,
                    Rr22Rarams, DpParams>(),
           py::arg("protocol") = PsiProtocol::PROTOCOL_UNSPECIFIED,
           py::arg("receiver_rank") = 0, py::arg("bucket_size") = 1 << 20,
           py::arg("broadcast_result") = false,
           py::arg("ecdh_params") = EcdhParams(),
           py::arg("rr22_params") = Rr22Rarams(),
           py::arg("dp_params") = DpParams())
      .def_readwrite("protocol", &PsiProtocolConfig::protocol)
      .def_readwrite("receiver_rank", &PsiProtocolConfig::receiver_rank)
      .def_readwrite("bucket_size", &PsiProtocolConfig::bucket_size)
      .def_readwrite("broadcast_result", &PsiProtocolConfig::broadcast_result)
      .def_readwrite("ecdh_params", &PsiProtocolConfig::ecdh_params)
      .def_readwrite("rr22_params", &PsiProtocolConfig::rr22_params)
      .def_readwrite("dp_params", &PsiProtocolConfig::dp_params);

  py::enum_<SourceType>(m, "SourceType")
      .value("SOURCE_TYPE_UNSPECIFIED", SourceType::SOURCE_TYPE_UNSPECIFIED)
      .value("SOURCE_TYPE_FILE_CSV", SourceType::SOURCE_TYPE_FILE_CSV)
      .export_values();

  py::class_<InputParams>(m, "InputParams", "input params")
      .def(py::init<SourceType, std::string, std::vector<std::string>, bool>(),
           py::arg("type") = SourceType::SOURCE_TYPE_FILE_CSV,
           py::arg("path") = "", py::arg("selected_keys") = py::list(),
           py::arg("keys_unique") = false)
      .def_readwrite("type", &InputParams::type)
      .def_readwrite("path", &InputParams::path)
      .def_readwrite("selected_keys", &InputParams::selected_keys)
      .def_readwrite("keys_unique", &InputParams::keys_unique);

  py::class_<OutputParams>(m, "OutputParams", "output params")
      .def(py::init<SourceType, std::string, bool, std::string>(),
           py::arg("type") = SourceType::SOURCE_TYPE_FILE_CSV,
           py::arg("path") = "", py::arg("disable_alignment") = false,
           py::arg("csv_null_rep") = "NULL")
      .def_readwrite("type", &OutputParams::type)
      .def_readwrite("path", &OutputParams::path)
      .def_readwrite("disable_alignment", &OutputParams::disable_alignment)
      .def_readwrite("csv_null_rep", &OutputParams::csv_null_rep);

  py::class_<CheckpointConfig>(m, "CheckpointConfig", "checkpoint config")
      .def(py::init<bool, std::string>(), py::arg("enable") = false,
           py::arg("path") = "")
      .def_readwrite("enable", &CheckpointConfig::enable)
      .def_readwrite("path", &CheckpointConfig::path);

  py::enum_<ResultJoinType>(m, "ResultJoinType")
      .value("JOIN_TYPE_UNSPECIFIED", ResultJoinType::JOIN_TYPE_UNSPECIFIED)
      .value("JOIN_TYPE_INNER_JOIN", ResultJoinType::JOIN_TYPE_INNER_JOIN)
      .value("JOIN_TYPE_LEFT_JOIN", ResultJoinType::JOIN_TYPE_LEFT_JOIN)
      .value("JOIN_TYPE_RIGHT_JOIN", ResultJoinType::JOIN_TYPE_RIGHT_JOIN)
      .value("JOIN_TYPE_FULL_JOIN", ResultJoinType::JOIN_TYPE_FULL_JOIN)
      .value("JOIN_TYPE_DIFFERENCE", ResultJoinType::JOIN_TYPE_DIFFERENCE)
      .export_values();

  py::class_<ResultJoinConfig>(m, "ResultJoinConfig", "result join config")
      .def(py::init<ResultJoinType, uint32_t>(),
           py::arg("type") = ResultJoinType::JOIN_TYPE_INNER_JOIN,
           py::arg("left_side_rank") = 0)
      .def_readwrite("type", &ResultJoinConfig::type)
      .def_readwrite("left_side_rank", &ResultJoinConfig::left_side_rank);

  py::class_<PsiExecuteConfig>(m, "PsiExecuteConfig", "psi execute config")
      .def(py::init<PsiProtocolConfig, InputParams, OutputParams,
                    ResultJoinConfig, CheckpointConfig>(),
           py::arg("protocol_conf") = PsiProtocolConfig(),
           py::arg("input_params") = InputParams(),
           py::arg("output_params") = OutputParams(),
           py::arg("join_conf") = ResultJoinConfig(),
           py::arg("checkpoint_conf") = CheckpointConfig())
      .def_readwrite("protocol_conf", &PsiExecuteConfig::protocol_conf)
      .def_readwrite("input_params", &PsiExecuteConfig::input_params)
      .def_readwrite("output_params", &PsiExecuteConfig::output_params)
      .def_readwrite("join_conf", &PsiExecuteConfig::join_conf)
      .def_readwrite("checkpoint_conf", &PsiExecuteConfig::checkpoint_conf);

  // ub psi
  py::enum_<ub::UbPsiExecuteConfig::Mode>(m, "UbPsiMode")
      .value("MODE_UNSPECIFIED", ub::UbPsiExecuteConfig::Mode::MODE_UNSPECIFIED)
      .value("MODE_OFFLINE_GEN_CACHE",
             ub::UbPsiExecuteConfig::Mode::MODE_OFFLINE_GEN_CACHE)
      .value("MODE_OFFLINE_TRANSFER_CACHE",
             ub::UbPsiExecuteConfig::Mode::MODE_OFFLINE_TRANSFER_CACHE)
      .value("MODE_OFFLINE", ub::UbPsiExecuteConfig::Mode::MODE_OFFLINE)
      .value("MODE_ONLINE", ub::UbPsiExecuteConfig::Mode::MODE_ONLINE)
      .value("MODE_FULL", ub::UbPsiExecuteConfig::Mode::MODE_FULL)
      .export_values();

  py::enum_<ub::UbPsiRole>(m, "UbPsiRole")
      .value("ROLE_UNSPECIFIED", ub::UbPsiRole::ROLE_UNSPECIFIED)
      .value("ROLE_SERVER", ub::UbPsiRole::ROLE_SERVER)
      .value("ROLE_CLIENT", ub::UbPsiRole::ROLE_CLIENT)
      .export_values();

  py::class_<ub::UbPsiServerParams>(m, "UbPsiServerParams",
                                    "ub psi server params")
      .def(py::init<std::string>(), py::arg("secret_key_path") = "")
      .def_readwrite("secret_key_path",
                     &ub::UbPsiServerParams::secret_key_path);

  py::class_<ub::UbPsiExecuteConfig>(m, "UbPsiExecuteConfig",
                                     "ub psi execute config")
      // .def(py::init<>())
      .def(py::init<ub::UbPsiExecuteConfig::Mode, ub::UbPsiRole, bool, bool,
                    std::string, InputParams, OutputParams,
                    ub::UbPsiServerParams, ResultJoinConfig>(),
           py::arg("mode") = ub::UbPsiExecuteConfig::Mode::MODE_UNSPECIFIED,
           py::arg("role") = ub::UbPsiRole::ROLE_UNSPECIFIED,
           py::arg("server_receive_result") = false,
           py::arg("client_receive_result") = false, py::arg("cache_path") = "",
           py::arg("input_params") = InputParams(),
           py::arg("output_params") = OutputParams(),
           py::arg("server_params") = ub::UbPsiServerParams(),
           py::arg("join_conf") = ResultJoinConfig())
      .def_readwrite("mode", &ub::UbPsiExecuteConfig::mode)
      .def_readwrite("role", &ub::UbPsiExecuteConfig::role)
      .def_readwrite("server_receive_result",
                     &ub::UbPsiExecuteConfig::server_receive_result)
      .def_readwrite("client_receive_result",
                     &ub::UbPsiExecuteConfig::client_receive_result)
      .def_readwrite("cache_path", &ub::UbPsiExecuteConfig::cache_path)
      .def_readwrite("input_params", &ub::UbPsiExecuteConfig::input_params)
      .def_readwrite("output_params", &ub::UbPsiExecuteConfig::output_params)
      .def_readwrite("server_params", &ub::UbPsiExecuteConfig::server_params)
      .def_readwrite("join_conf", &ub::UbPsiExecuteConfig::join_conf);

  py::class_<PsiExecuteReport>(m, "PsiExecuteReport", "psi execute report")
      .def(py::init<>())
      .def_readonly("original_count", &PsiExecuteReport::original_count)
      .def_readonly("intersection_count", &PsiExecuteReport::intersection_count)
      .def_readonly("original_unique_count",
                    &PsiExecuteReport::original_unique_count)
      .def_readonly("intersection_unique_count",
                    &PsiExecuteReport::intersection_unique_count);

  // psi function
  m.def("psi_execute", PsiExecute, "psi execute", py::arg("config"),
        py::arg("lctx"), py::arg("progress_params") = ProgressParams(), NO_GIL);
  m.def("ub_psi_execute", UbPsiExecute, "ub psi execute", py::arg("config"),
        py::arg("lctx"), NO_GIL);
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

  BindPsi(m);
}

}  // namespace psi::api
