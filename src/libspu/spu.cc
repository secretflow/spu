// Copyright 2024 Ant Group Co., Ltd.
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

#include "libspu/spu.h"

#include <functional>

#include "google/protobuf/json/json.h"
#include "magic_enum.hpp"

#include "libspu/spu.pb.h"
namespace spu {

std::string_view GetDataTypeName(DataType dtype) {
  return magic_enum::enum_name(dtype);
}

std::string_view GetPtTypeName(PtType pt_type) {
  return magic_enum::enum_name(pt_type);
}

std::string_view GetVisibilityName(Visibility vis) {
  return magic_enum::enum_name(vis);
}

std::string_view GetFieldTypeName(FieldType field) {
  return magic_enum::enum_name(field);
}

std::string_view GetProtocolKindName(ProtocolKind kind) {
  return magic_enum::enum_name(kind);
}

std::string_view GetSortMethodName(RuntimeConfig::SortMethod method) {
  return magic_enum::enum_name(method);
}

std::string_view GetExpModeName(RuntimeConfig::ExpMode mode) {
  return magic_enum::enum_name(mode);
}

std::string_view GetLogModeName(RuntimeConfig::LogMode mode) {
  return magic_enum::enum_name(mode);
}

std::string_view GetSigmoidModeName(RuntimeConfig::SigmoidMode mode) {
  return magic_enum::enum_name(mode);
}

std::string_view GetBeaverTypeName(RuntimeConfig::BeaverType type) {
  return magic_enum::enum_name(type);
}

std::string_view GetSourceIRTypeName(SourceIRType type) {
  return magic_enum::enum_name(type);
}

std::string_view GetXLAPrettyPrintKindName(XLAPrettyPrintKind kind) {
  return magic_enum::enum_name(kind);
}

bool ParseProtocolKind(std::string_view str, ProtocolKind* protocol) {
  auto result = magic_enum::enum_cast<ProtocolKind>(str);
  if (result.has_value()) {
    *protocol = result.value();
    return true;
  }
  return false;
}

void convertFromPB(const pb::RuntimeConfig& src, RuntimeConfig& dst) {
  dst.protocol = ProtocolKind(src.protocol());
  dst.field = FieldType(src.field());
  dst.fxp_fraction_bits = src.fxp_fraction_bits();
  dst.max_concurrency = src.max_concurrency();
  dst.enable_action_trace = src.enable_action_trace();
  dst.enable_type_checker = src.enable_type_checker();
  dst.enable_pphlo_trace = src.enable_pphlo_trace();
  dst.enable_runtime_snapshot = src.enable_runtime_snapshot();
  dst.snapshot_dump_dir = src.snapshot_dump_dir();
  dst.enable_pphlo_profile = src.enable_pphlo_profile();
  dst.enable_hal_profile = src.enable_hal_profile();
  dst.public_random_seed = src.public_random_seed();
  dst.share_max_chunk_size = src.share_max_chunk_size();
  dst.sort_method = RuntimeConfig::SortMethod(src.sort_method());
  dst.quick_sort_threshold = src.quick_sort_threshold();
  dst.fxp_div_goldschmidt_iters = src.fxp_div_goldschmidt_iters();
  dst.fxp_exp_mode = RuntimeConfig::ExpMode(src.fxp_exp_mode());
  dst.fxp_exp_iters = src.fxp_exp_iters();
  dst.fxp_log_mode = RuntimeConfig::LogMode(src.fxp_log_mode());
  dst.fxp_log_iters = src.fxp_log_iters();
  dst.fxp_log_orders = src.fxp_log_orders();
  dst.sigmoid_mode = RuntimeConfig::SigmoidMode(src.sigmoid_mode());
  dst.enable_lower_accuracy_rsqrt = src.enable_lower_accuracy_rsqrt();
  dst.sine_cosine_iters = src.sine_cosine_iters();
  dst.beaver_type = RuntimeConfig::BeaverType(src.beaver_type());
  dst.trunc_allow_msb_error = src.trunc_allow_msb_error();
  dst.experimental_disable_mmul_split = src.experimental_disable_mmul_split();
  dst.experimental_enable_inter_op_par = src.experimental_enable_inter_op_par();
  dst.experimental_enable_intra_op_par = src.experimental_enable_intra_op_par();
  dst.experimental_disable_vectorization =
      src.experimental_disable_vectorization();
  dst.experimental_inter_op_concurrency =
      src.experimental_inter_op_concurrency();
  dst.experimental_enable_colocated_optimization =
      src.experimental_enable_colocated_optimization();
  dst.experimental_enable_exp_prime = src.experimental_enable_exp_prime();
  dst.experimental_exp_prime_offset = src.experimental_exp_prime_offset();
  dst.experimental_exp_prime_disable_lower_bound =
      src.experimental_exp_prime_disable_lower_bound();
  dst.experimental_exp_prime_enable_upper_bound =
      src.experimental_exp_prime_enable_upper_bound();

  if (src.has_ttp_beaver_config()) {
    auto ttp_conf = src.ttp_beaver_config();
    std::unique_ptr<ClientSSLConfig> ssl_config;
    if (ttp_conf.has_ssl_config()) {
      ssl_config = std::make_unique<ClientSSLConfig>(
          ttp_conf.ssl_config().certificate(),
          ttp_conf.ssl_config().private_key(),
          ttp_conf.ssl_config().ca_file_path(),
          ttp_conf.ssl_config().verify_depth());
    }
    dst.ttp_beaver_config = std::make_unique<TTPBeaverConfig>(
        ttp_conf.server_host(), ttp_conf.adjust_rank(),
        ttp_conf.asym_crypto_schema(), ttp_conf.server_public_key(),
        ttp_conf.transport_protocol(), std::move(ssl_config));
  }

  if (src.has_cheetah_2pc_config()) {
    dst.cheetah_2pc_config =
        CheetahConfig(src.cheetah_2pc_config().disable_matmul_pack(),
                      src.cheetah_2pc_config().enable_mul_lsb_error(),
                      CheetahOtKind(src.cheetah_2pc_config().ot_kind()));
  }
}

void convertToPB(const RuntimeConfig& src, pb::RuntimeConfig& dst) {
  dst.set_protocol(pb::ProtocolKind(src.protocol));
  dst.set_field(pb::FieldType(src.field));
  dst.set_fxp_fraction_bits(src.fxp_fraction_bits);
  dst.set_max_concurrency(src.max_concurrency);
  dst.set_enable_action_trace(src.enable_action_trace);
  dst.set_enable_type_checker(src.enable_type_checker);
  dst.set_enable_pphlo_trace(src.enable_pphlo_trace);
  dst.set_enable_runtime_snapshot(src.enable_runtime_snapshot);
  dst.set_snapshot_dump_dir(src.snapshot_dump_dir);
  dst.set_enable_pphlo_profile(src.enable_pphlo_profile);
  dst.set_enable_hal_profile(src.enable_hal_profile);
  dst.set_public_random_seed(src.public_random_seed);
  dst.set_share_max_chunk_size(src.share_max_chunk_size);
  dst.set_sort_method(pb::RuntimeConfig::SortMethod(src.sort_method));
  dst.set_quick_sort_threshold(src.quick_sort_threshold);
  dst.set_fxp_div_goldschmidt_iters(src.fxp_div_goldschmidt_iters);
  dst.set_fxp_exp_mode(pb::RuntimeConfig::ExpMode(src.fxp_exp_mode));
  dst.set_fxp_exp_iters(src.fxp_exp_iters);
  dst.set_fxp_log_mode(pb::RuntimeConfig::LogMode(src.fxp_log_mode));
  dst.set_fxp_log_iters(src.fxp_log_iters);
  dst.set_fxp_log_orders(src.fxp_log_orders);
  dst.set_sigmoid_mode(pb::RuntimeConfig::SigmoidMode(src.sigmoid_mode));
  dst.set_enable_lower_accuracy_rsqrt(src.enable_lower_accuracy_rsqrt);
  dst.set_sine_cosine_iters(src.sine_cosine_iters);
  dst.set_beaver_type(pb::RuntimeConfig::BeaverType(src.beaver_type));
  if (src.ttp_beaver_config) {
    auto ttp_conf = dst.mutable_ttp_beaver_config();
    ttp_conf->set_server_host(src.ttp_beaver_config->server_host);
    ttp_conf->set_adjust_rank(src.ttp_beaver_config->adjust_rank);
    ttp_conf->set_asym_crypto_schema(src.ttp_beaver_config->asym_crypto_schema);
    ttp_conf->set_server_public_key(src.ttp_beaver_config->server_public_key);
    ttp_conf->set_transport_protocol(src.ttp_beaver_config->transport_protocol);
    if (src.ttp_beaver_config->ssl_config) {
      auto ssl_config = ttp_conf->mutable_ssl_config();
      ssl_config->set_certificate(
          src.ttp_beaver_config->ssl_config->certificate);
      ssl_config->set_private_key(
          src.ttp_beaver_config->ssl_config->private_key);
      ssl_config->set_ca_file_path(
          src.ttp_beaver_config->ssl_config->ca_file_path);
      ssl_config->set_verify_depth(
          src.ttp_beaver_config->ssl_config->verify_depth);
    }
  }
  if (src.protocol == ProtocolKind::CHEETAH) {
    auto cheetah_conf = dst.mutable_cheetah_2pc_config();
    cheetah_conf->set_disable_matmul_pack(
        src.cheetah_2pc_config.disable_matmul_pack);
    cheetah_conf->set_enable_mul_lsb_error(
        src.cheetah_2pc_config.enable_mul_lsb_error);
    cheetah_conf->set_ot_kind(
        pb::CheetahOtKind(src.cheetah_2pc_config.ot_kind));
  }
  dst.set_trunc_allow_msb_error(src.trunc_allow_msb_error);
  dst.set_experimental_disable_mmul_split(src.experimental_disable_mmul_split);
  dst.set_experimental_enable_inter_op_par(
      src.experimental_enable_inter_op_par);
  dst.set_experimental_enable_intra_op_par(
      src.experimental_enable_intra_op_par);
  dst.set_experimental_disable_vectorization(
      src.experimental_disable_vectorization);
  dst.set_experimental_inter_op_concurrency(
      src.experimental_inter_op_concurrency);
  dst.set_experimental_enable_colocated_optimization(
      src.experimental_enable_colocated_optimization);
  dst.set_experimental_enable_exp_prime(src.experimental_enable_exp_prime);
  dst.set_experimental_exp_prime_offset(src.experimental_exp_prime_offset);
  dst.set_experimental_exp_prime_disable_lower_bound(
      src.experimental_exp_prime_disable_lower_bound);
  dst.set_experimental_exp_prime_enable_upper_bound(
      src.experimental_exp_prime_enable_upper_bound);
}

RuntimeConfig::RuntimeConfig(const spu::pb::RuntimeConfig& pb_conf) {
  convertFromPB(pb_conf, *this);
}

std::string RuntimeConfig::SerializeAsString() const {
  pb::RuntimeConfig pb_conf;
  convertToPB(*this, pb_conf);
  return pb_conf.SerializeAsString();
}

std::string RuntimeConfig::DebugString() const {
  pb::RuntimeConfig pb_conf;
  convertToPB(*this, pb_conf);
  return pb_conf.DebugString();
}

pb::RuntimeConfig RuntimeConfig::ToProto() const {
  pb::RuntimeConfig pb_conf;
  convertToPB(*this, pb_conf);
  return pb_conf;
}

bool RuntimeConfig::ParseFromJsonString(std::string_view data) {
  pb::RuntimeConfig pb_conf;
  auto status = google::protobuf::json::JsonStringToMessage(data, &pb_conf);
  if (!status.ok()) return false;
  convertFromPB(pb_conf, *this);
  return true;
}

bool RuntimeConfig::ParseFromString(std::string_view data) {
  pb::RuntimeConfig pb_conf;
  if (!pb_conf.ParseFromString(data)) return false;
  convertFromPB(pb_conf, *this);
  return true;
}

bool ExecutableProto::ParseFromString(std::string_view data) {
  pb::ExecutableProto pb_exec;
  if (!pb_exec.ParseFromString(data)) return false;
  name = pb_exec.name();
  input_names = {pb_exec.input_names().begin(), pb_exec.input_names().end()};
  output_names = {pb_exec.output_names().begin(), pb_exec.output_names().end()};
  code = pb_exec.code();
  return true;
}

std::string ExecutableProto::SerializeAsString() const {
  pb::ExecutableProto pb_exec;
  pb_exec.set_name(name);
  for (const auto& in : input_names) {
    pb_exec.add_input_names(in);
  }
  for (const auto& out : output_names) {
    pb_exec.add_output_names(out);
  }
  pb_exec.set_code(code);
  return pb_exec.SerializeAsString();
}

#if __cplusplus < 202002L
bool CompilationSource::operator==(const CompilationSource& other) const {
  return ir_type == other.ir_type && ir_txt == other.ir_txt &&
         input_visibility == other.input_visibility;
}

bool CompilerOptions::operator==(const CompilerOptions& other) const {
  return enable_pretty_print == other.enable_pretty_print &&
         pretty_print_dump_dir == other.pretty_print_dump_dir &&
         xla_pp_kind == other.xla_pp_kind &&
         disable_sqrt_plus_epsilon_rewrite ==
             other.disable_sqrt_plus_epsilon_rewrite &&
         disable_div_sqrt_rewrite == other.disable_div_sqrt_rewrite &&
         disable_reduce_truncation_optimization ==
             other.disable_reduce_truncation_optimization &&
         disable_maxpooling_optimization ==
             other.disable_maxpooling_optimization &&
         disallow_mix_types_opts == other.disallow_mix_types_opts &&
         disable_select_optimization == other.disable_select_optimization &&
         enable_optimize_denominator_with_broadcast ==
             other.enable_optimize_denominator_with_broadcast &&
         disable_deallocation_insertion ==
             other.disable_deallocation_insertion &&
         disable_partial_sort_optimization ==
             other.disable_partial_sort_optimization;
}
#endif
};  // namespace spu

namespace std {
template <typename T, typename... Rest>
inline void hash_combine(std::size_t& seed, const T& v, const Rest&... rest) {
  seed ^= std::hash<T>{}(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  (hash_combine(seed, rest), ...);
}

std::size_t hash<spu::CompilationSource>::operator()(
    const spu::CompilationSource& cs) const {
  std::size_t seed = 0;
  hash_combine(seed, cs.ir_type, cs.ir_txt);
  for (const auto& v : cs.input_visibility) {
    hash_combine(seed, v);
  }

  return seed;
}

std::size_t hash<spu::CompilerOptions>::operator()(
    const spu::CompilerOptions& co) const {
  std::size_t seed = 0;
  hash_combine(
      seed, co.enable_pretty_print, co.pretty_print_dump_dir, co.xla_pp_kind,
      co.disable_sqrt_plus_epsilon_rewrite, co.disable_div_sqrt_rewrite,
      co.disable_reduce_truncation_optimization,
      co.disable_maxpooling_optimization, co.disallow_mix_types_opts,
      co.disable_select_optimization,
      co.enable_optimize_denominator_with_broadcast,
      co.disable_deallocation_insertion, co.disable_partial_sort_optimization);
  return seed;
}
};  // namespace std
