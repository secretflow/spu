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

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace spu {

namespace pb {
class RuntimeConfig;
}  // namespace pb

// The SPU datatype
enum DataType {
  DT_INVALID = 0,

  DT_I1 = 1,    // 1bit integer (bool).
  DT_I8 = 2,    // int8
  DT_U8 = 3,    // uint8
  DT_I16 = 4,   // int16
  DT_U16 = 5,   // uint16
  DT_I32 = 6,   // int32
  DT_U32 = 7,   // uint32
  DT_I64 = 8,   // int64
  DT_U64 = 9,   // uint64
  DT_F16 = 10,  // half
  DT_F32 = 11,  // float
  DT_F64 = 12,  // double
};

// The visibility type.
//
// SPU is a secure evaluation runtime, but not all data are secret, some of them
// are publicly known to all parties, marking them as public will improve
// performance significantly.
enum Visibility {
  VIS_INVALID = 0,
  VIS_SECRET = 1,   // Invisible(unknown) for all or some of the parties.
  VIS_PUBLIC = 2,   // Visible(public) for all parties.
  VIS_PRIVATE = 3,  // Visible for only one party
};

// Plaintext type
//
// SPU runtime does not process with plaintext directly, plaintext type is
// mainly used for IO purposes, when converting a plaintext buffer to an SPU
// buffer, we have to let spu know which type the plaintext buffer is.
enum PtType {
  PT_INVALID = 0,  //
  PT_I8 = 1,       // int8_t
  PT_U8 = 2,       // uint8_t
  PT_I16 = 3,      // int16_t
  PT_U16 = 4,      // uint16_t
  PT_I32 = 5,      // int32_t
  PT_U32 = 6,      // uint32_t
  PT_I64 = 7,      // int64_t
  PT_U64 = 8,      // uint64_t
  PT_I128 = 9,     // int128_t
  PT_U128 = 10,    // uint128_t
  PT_I1 = 11,      // bool

  PT_F16 = 30,  // half
  PT_F32 = 31,  // float
  PT_F64 = 32,  // double

  PT_CF32 = 50,  // complex float
  PT_CF64 = 51,  // complex double
};

// A security parameter type.
//
// The secure evaluation is based on some algebraic structure (ring or field),
enum FieldType {
  FT_INVALID = 0,

  FM32 = 1,   // Ring 2^32
  FM64 = 2,   // Ring 2^64
  FM128 = 3,  // Ring 2^128
};

// The protocol kind.
enum ProtocolKind {
  // Invalid protocol.
  PROT_INVALID = 0,

  // The reference implementation in `ring^2k`, note: this 'protocol' only
  // behave-like a fixed point secure protocol without any security guarantee.
  // Hence, it should only be selected for debugging purposes.
  REF2K = 1,

  // A semi-honest multi-party protocol. This protocol requires a trusted third
  // party to generate the offline correlated randoms. Currently, SecretFlow by
  // default ships this protocol with a trusted first party. Hence, it should
  // only be used for debugging purposes.
  SEMI2K = 2,

  // A honest majority 3PC-protocol. SecretFlow provides the semi-honest
  // implementation without Yao.
  ABY3 = 3,

  // The famous [Cheetah](https://eprint.iacr.org/2022/207) protocol, a very
  // fast 2PC protocol.
  CHEETAH = 4,

  // A semi-honest 3PC-protocol for Neural Network, P2 as the helper,
  // (https://eprint.iacr.org/2018/442)
  SECURENN = 5,
};

//////////////////////////////////////////////////////////////////////////
// Runtime configuration
//////////////////////////////////////////////////////////////////////////
struct ClientSSLConfig {
  // Certificate in PEM format, supported both file path and raw string
  std::string certificate;
  // Private key in PEM format, supported both file path and raw string based on
  // prefix
  std::string private_key;
  // The trusted CA file to verify the peer's certificate
  // If empty, use the system default CA files
  std::string ca_file_path;
  // Maximum depth of the certificate chain for verification
  // If 0, turn off the verification
  int32_t verify_depth;

  ClientSSLConfig() = default;
  ClientSSLConfig(std::string certificate, std::string private_key,
                  std::string ca_file_path, int32_t verify_depth)
      : certificate(std::move(certificate)),
        private_key(std::move(private_key)),
        ca_file_path(std::move(ca_file_path)),
        verify_depth(verify_depth) {}
};

struct TTPBeaverConfig {
  // TrustedThirdParty beaver server's remote ip:port or load-balance uri.
  std::string server_host;
  // which rank do adjust rpc call, usually choose the rank closer to the
  // server.
  int32_t adjust_rank = 0;

  // asym_crypto_schema: support ["SM2"]
  // Will support 25519 in the future, after yacl supported it.
  std::string asym_crypto_schema;
  // Server's public key in PEM format
  std::string server_public_key;

  // Transport protocol, support ["http", "h2"]
  std::string transport_protocol;

  // Configurations related to SSL
  std::shared_ptr<ClientSSLConfig> ssl_config;

  bool has_ssl_config() const { return ssl_config != nullptr; }

  TTPBeaverConfig() = default;
  TTPBeaverConfig(std::string server_host, int32_t adjust_rank,
                  std::string asym_crypto_schema, std::string server_public_key,
                  std::string transport_protocol,
                  std::shared_ptr<ClientSSLConfig> ssl_config = nullptr)
      : server_host(std::move(server_host)),
        adjust_rank(adjust_rank),
        asym_crypto_schema(std::move(asym_crypto_schema)),
        server_public_key(std::move(server_public_key)),
        transport_protocol(std::move(transport_protocol)),
        ssl_config(std::move(ssl_config)) {}
};

enum CheetahOtKind { YACL_Ferret = 0, YACL_Softspoken = 1, EMP_Ferret = 2 };

struct CheetahConfig {
  // disable the ciphertext packing for matmul
  bool disable_matmul_pack;
  // allow least significant bits error for point-wise mul
  bool enable_mul_lsb_error;
  // Setup for cheetah ot
  CheetahOtKind ot_kind;

  CheetahConfig() = default;
  CheetahConfig(bool disable_matmul_pack, bool enable_mul_lsb_error,
                CheetahOtKind ot_kind)
      : disable_matmul_pack(disable_matmul_pack),
        enable_mul_lsb_error(enable_mul_lsb_error),
        ot_kind(ot_kind) {}
};

// The SPU runtime configuration.
struct RuntimeConfig {
  static const uint64_t kDefaultShareMaxChunkSize = 128 * 1024 * 1024;
  static const int64_t kDefaultQuickSortThreshold = 32;
  static const int64_t kDefaultFxpDivGoldschmidtIters = 2;
  static const int64_t kDefaultFxpExpIters = 8;
  static const int64_t kDefaultFxpLogIters = 3;
  static const int64_t kDefaultFxpLogOrders = 8;
  static const int64_t kDefaultSineCosineIters = 10;
  static const uint64_t kDefaultExperimentalInterOpConcurrency = 8;
  ///////////////////////////////////////
  // Basic
  ///////////////////////////////////////

  // The protocol kind.
  ProtocolKind protocol = PROT_INVALID;

  // The field type.
  FieldType field = FT_INVALID;

  // Number of fraction bits of fixed-point number.
  // 0(default) indicates implementation defined.
  int64_t fxp_fraction_bits = 0;

  // Max number of cores
  int32_t max_concurrency = 0;

  ///////////////////////////////////////
  // Advanced
  ///////////////////////////////////////

  // @exclude
  // Runtime related, reserved for [10, 50)

  // When enabled, runtime prints verbose info of the call stack, debug purpose
  // only.
  bool enable_action_trace = false;

  // When enabled, runtime checks runtime type infos against the
  // compile-time ones, exceptions are raised if mismatches happen. Note:
  // Runtime outputs prefer runtime type infos even when flag is on.
  bool enable_type_checker = false;

  // When enabled, runtime prints executed pphlo list, debug purpose only.
  bool enable_pphlo_trace = false;

  // When enabled, runtime dumps executed executables in the dump_dir, debug
  // purpose only.
  bool enable_runtime_snapshot = false;
  std::string snapshot_dump_dir;

  // When enabled, runtime records detailed pphlo timing data, debug purpose
  // only.
  // WARNING: the `send bytes` information is only accurate when
  // `experimental_enable_inter_op_par` and `experimental_enable_intra_op_par`
  // options are disabled.
  bool enable_pphlo_profile = false;

  // When enabled, runtime records detailed hal timing data, debug purpose only.
  // WARNING: the `send bytes` information is only accurate when
  // `experimental_enable_inter_op_par` and `experimental_enable_intra_op_par`
  // options are disabled.
  bool enable_hal_profile = false;

  // The public random variable generated by the runtime, the concrete prg
  // function is implementation defined.
  // Note: this seed only applies to `public variable` only, it has nothing
  // to do with security.
  uint64_t public_random_seed = 0;

  // max chunk size for Value::toProto
  // default: 128 * 1024 * 1024
  uint64_t share_max_chunk_size = kDefaultShareMaxChunkSize;

  enum SortMethod {
    SORT_DEFAULT = 0,  // Implementation defined.
    SORT_RADIX = 1,    // The radix sort (stable sort, need efficient shuffle).
    SORT_QUICK = 2,    // The quick sort (unstable, need efficient shuffle).
    SORT_NETWORK = 3,  // The odd-even sorting network (unstable, most general).
  };

  // SPU supports multiple sorting algorithms.
  //  -for 2pc, only sorting network is supported.
  //  -for 2.5pc or 3pc, all these algorithms are supported.
  //  -for stable sort, only radix sort is supported.
  SortMethod sort_method = SORT_DEFAULT;

  // threshold for quick sort, when the size of the array is less than this
  // value, use merge sort instead
  int64_t quick_sort_threshold = kDefaultQuickSortThreshold;

  // @exclude
  // Fixed-point arithmetic related, reserved for [50, 100)

  // The iterations use in f_div with Goldschmidt method.
  // 0(default) indicates implementation defined.
  int64_t fxp_div_goldschmidt_iters = kDefaultFxpDivGoldschmidtIters;

  // The exponential approximation method.
  enum ExpMode {
    EXP_DEFAULT = 0,  // Implementation defined.
    EXP_PADE = 1,     // The pade approximation.
    EXP_TAYLOR = 2,   // Taylor series approximation.
    EXP_PRIME = 3,    // exp prime only available for some implementations
  };

  // The exponent approximation method.
  ExpMode fxp_exp_mode = EXP_DEFAULT;

  // Number of iterations of `exp` approximation, 0(default) indicates impl
  // defined.
  int64_t fxp_exp_iters = kDefaultFxpExpIters;

  // The logarithm approximation method.
  enum LogMode {
    LOG_DEFAULT = 0,  // Implementation defined.
    LOG_PADE = 1,     // The pade approximation.
    LOG_NEWTON = 2,   // The newton approximation.
    LOG_MINMAX = 3,   // The minmax approximation.
  };

  // The logarithm approximation method.
  LogMode fxp_log_mode = LOG_DEFAULT;

  // Number of iterations of `log` approximation, 0(default) indicates
  // impl-defined.
  int64_t fxp_log_iters = kDefaultFxpLogIters;

  // Number of orders of `log` approximation, 0(default) indicates impl defined.
  int64_t fxp_log_orders = kDefaultFxpLogOrders;

  // The sigmoid approximation method.
  enum SigmoidMode {
    // Implementation defined.
    SIGMOID_DEFAULT = 0,
    // Minmax approximation one order.
    // f(x) = 0.5 + 0.125 * x
    SIGMOID_MM1 = 1,
    // Piece-wise simulation.
    // f(x) = 0.5 + 0.125x if -4 <= x <= 4
    //        1            if       x > 4
    //        0            if  -4 > x
    SIGMOID_SEG3 = 2,
    // The real definition, which depends on exp's accuracy.
    // f(x) = 1 / (1 + exp(-x))
    SIGMOID_REAL = 3,
  };

  // The sigmoid function approximation model.
  SigmoidMode sigmoid_mode = SIGMOID_DEFAULT;

  // Enable a simpler rsqrt approximation
  bool enable_lower_accuracy_rsqrt = false;

  // Sine/Cosine approximation iterations
  int64_t sine_cosine_iters = kDefaultSineCosineIters;

  /// - MPC protocol related definitions.

  enum BeaverType {
    // Assume first party (rank0) as trusted party to generate beaver triple.
    // WARNING: It is NOT SAFE and SHOULD NOT BE used in production.
    TrustedFirstParty = 0,
    // Generate beaver triple through an additional trusted third party.
    TrustedThirdParty = 1,
    // Generate beaver triple through multi-party.
    MultiParty = 2,
  };
  // beaver config, works for semi2k and spdz2k for now.
  BeaverType beaver_type = TrustedFirstParty;

  // TrustedThirdParty configs.
  std::shared_ptr<TTPBeaverConfig> ttp_beaver_config;

  // Cheetah 2PC configs.
  CheetahConfig cheetah_2pc_config;

  // For protocol like SecureML, the most significant bit may have error with
  // low probability, which lead to huge calculation error.
  bool trunc_allow_msb_error = false;

  /// System related configurations start.

  // Experimental: DO NOT USE
  bool experimental_disable_mmul_split = false;
  // Inter op parallel, aka, DAG level parallel.
  bool experimental_enable_inter_op_par = false;
  // Intra op parallel, aka, hal/mpc level parallel.
  bool experimental_enable_intra_op_par = false;
  // Disable kernel level vectorization.
  bool experimental_disable_vectorization = false;
  // Inter op concurrency.
  uint64_t experimental_inter_op_concurrency =
      kDefaultExperimentalInterOpConcurrency;
  // Enable use of private type
  bool experimental_enable_colocated_optimization = false;

  // enable experimental exp prime method
  bool experimental_enable_exp_prime = false;

  // The offset parameter for exp prime methods.
  // control the valid range of exp prime method.
  // valid range is:
  // ((47 - offset - 2fxp)/log_2(e), (125 - 2fxp - offset)/log_2(e))
  // clamp to value would be
  //                lower bound: (48 - offset - 2fxp)/log_2(e)
  //                higher bound: (124 - 2fxp - offset)/log_2(e)
  // default offset is 13, 0 offset is not supported.
  uint32_t experimental_exp_prime_offset = 0;
  // whether to apply the clamping lower bound
  // default to enable it
  bool experimental_exp_prime_disable_lower_bound = false;
  // whether to apply the clamping upper bound
  // default to disable it
  bool experimental_exp_prime_enable_upper_bound = false;

  // static RuntimeConfig makeFromJson(const std::string& json_str);

  RuntimeConfig() = default;
  RuntimeConfig(ProtocolKind protocol, FieldType field,
                int64_t fxp_fraction_bits = 0)
      : protocol(protocol),
        field(field),
        fxp_fraction_bits(fxp_fraction_bits) {};
  RuntimeConfig(const RuntimeConfig& other) = default;
  explicit RuntimeConfig(const pb::RuntimeConfig& pb_conf);

  bool has_ttp_beaver_config() const { return ttp_beaver_config != nullptr; }

  bool ParseFromJsonString(std::string_view data);
  bool ParseFromString(std::string_view data);
  std::string SerializeAsString() const;
  std::string DebugString() const;
};

//////////////////////////////////////////////////////////////////////////
// Compiler relate definition
//////////////////////////////////////////////////////////////////////////
enum SourceIRType { XLA = 0, STABLEHLO = 1 };

struct CompilationSource {
  // Input IR type
  SourceIRType ir_type;

  // IR
  std::string ir_txt;

  // Input visibilities
  std::vector<Visibility> input_visibility;

  CompilationSource() = default;
  CompilationSource(SourceIRType ir_type, std::string ir_txt,
                    std::vector<Visibility> input_visibility)
      : ir_type(ir_type),
        ir_txt(std::move(ir_txt)),
        input_visibility(std::move(input_visibility)) {}

#if __cplusplus >= 202002L
  bool operator==(const CompilationSource& other) const = default;
#else
  bool operator==(const CompilationSource& other) const;
#endif
};

enum XLAPrettyPrintKind { TEXT = 0, DOT = 1, HTML = 2 };

struct CompilerOptions {
  // Pretty print
  bool enable_pretty_print = false;
  std::string pretty_print_dump_dir;
  XLAPrettyPrintKind xla_pp_kind = XLAPrettyPrintKind::TEXT;

  // Disable sqrt(x) + eps to sqrt(x+eps) rewrite
  bool disable_sqrt_plus_epsilon_rewrite = false;

  // Disable x/sqrt(y) to x*rsqrt(y) rewrite
  bool disable_div_sqrt_rewrite = false;

  // Disable reduce truncation optimization
  bool disable_reduce_truncation_optimization = false;

  // Disable maxpooling optimization
  bool disable_maxpooling_optimization = false;

  // Disallow mix type operations
  bool disallow_mix_types_opts = false;

  // Disable SelectOp optimization
  bool disable_select_optimization = false;

  // Enable optimize x/bcast(y) -> x * bcast(1/y)
  bool enable_optimize_denominator_with_broadcast = false;

  // Disable deallocation insertion pass
  bool disable_deallocation_insertion = false;

  // Disable sort->topk rewrite when only partial sort is required
  bool disable_partial_sort_optimization = false;

#if __cplusplus >= 202002L
  bool operator==(const CompilerOptions& other) const = default;
#else
  bool operator==(const CompilerOptions& other) const;
#endif
};

// The executable format accepted by SPU runtime.
//
// - Inputs should be prepared before running executable.
// - Output is maintained after execution, and can be fetched by output name.
//
// ```python
//   rt = spu.Runtime(...)            # create an spu runtime.
//   rt.set_var('x', ...)             # set variable to the runtime.
//   exe = spu.ExecutableProto(       # prepare the executable.
//           name = 'balabala',
//           input_names = ['x'],
//           output_names = ['y'],
//           code = ...)
//   rt.run(exe)                      # run the executable.
//   y = rt.get_var('y')              # get the executable from spu runtime.
// ```
struct ExecutableProto {
  // The name of the executable.
  std::string name;

  // The input names.
  std::vector<std::string> input_names;

  // The output names.
  std::vector<std::string> output_names;

  // The bytecode of the program, with format IR_MLIR_SPU.
  std::string code;

  ExecutableProto() = default;
  ExecutableProto(std::string name, std::vector<std::string> input_names,
                  std::vector<std::string> output_names, std::string code)
      : name(std::move(name)),
        input_names(std::move(input_names)),
        output_names(std::move(output_names)),
        code(std::move(code)) {}

  bool ParseFromString(std::string_view data);
  std::string SerializeAsString() const;
};
};  // namespace spu

namespace std {
template <>
struct hash<spu::CompilationSource> {
  std::size_t operator()(const spu::CompilationSource& cs) const;
};
template <>
struct hash<spu::CompilerOptions> {
  std::size_t operator()(const spu::CompilerOptions& co) const;
};
};  // namespace std
