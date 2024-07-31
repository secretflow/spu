# SPU Runtime Configuration

## Table of Contents



- Messages
    - [CheetahConfig](#cheetahconfig)
    - [CompilationSource](#compilationsource)
    - [CompilerOptions](#compileroptions)
    - [ExecutableProto](#executableproto)
    - [RuntimeConfig](#runtimeconfig)
    - [ShapeProto](#shapeproto)
    - [TTPBeaverConfig](#ttpbeaverconfig)
    - [ValueChunkProto](#valuechunkproto)
    - [ValueMetaProto](#valuemetaproto)



- Enums
    - [CheetahOtKind](#cheetahotkind)
    - [DataType](#datatype)
    - [FieldType](#fieldtype)
    - [ProtocolKind](#protocolkind)
    - [PtType](#pttype)
    - [RuntimeConfig.BeaverType](#runtimeconfigbeavertype)
    - [RuntimeConfig.ExpMode](#runtimeconfigexpmode)
    - [RuntimeConfig.LogMode](#runtimeconfiglogmode)
    - [RuntimeConfig.SigmoidMode](#runtimeconfigsigmoidmode)
    - [SourceIRType](#sourceirtype)
    - [Visibility](#visibility)
    - [XLAPrettyPrintKind](#xlaprettyprintkind)



- [Scalar Value Types](#scalar-value-types)



 <!-- end services -->

## Messages


### CheetahConfig



| Field | Type | Description |
| ----- | ---- | ----------- |
| disable_matmul_pack | [ bool](#bool) | disable the ciphertext packing for matmul |
| enable_mul_lsb_error | [ bool](#bool) | allow least significant bits error for point-wise mul |
| ot_kind | [ CheetahOtKind](#cheetahotkind) | Setup for cheetah ot |
 <!-- end Fields -->
 <!-- end HasFields -->


### CompilationSource



| Field | Type | Description |
| ----- | ---- | ----------- |
| ir_type | [ SourceIRType](#sourceirtype) | Input IR type |
| ir_txt | [ bytes](#bytes) | IR |
| input_visibility | [repeated Visibility](#visibility) | Input visibilities |
 <!-- end Fields -->
 <!-- end HasFields -->


### CompilerOptions



| Field | Type | Description |
| ----- | ---- | ----------- |
| enable_pretty_print | [ bool](#bool) | Pretty print |
| pretty_print_dump_dir | [ string](#string) | none |
| xla_pp_kind | [ XLAPrettyPrintKind](#xlaprettyprintkind) | none |
| disable_sqrt_plus_epsilon_rewrite | [ bool](#bool) | Disable sqrt(x) + eps to sqrt(x+eps) rewrite |
| disable_div_sqrt_rewrite | [ bool](#bool) | Disable x/sqrt(y) to x*rsqrt(y) rewrite |
| disable_reduce_truncation_optimization | [ bool](#bool) | Disable reduce truncation optimization |
| disable_maxpooling_optimization | [ bool](#bool) | Disable maxpooling optimization |
| disallow_mix_types_opts | [ bool](#bool) | Disallow mix type operations |
| disable_select_optimization | [ bool](#bool) | Disable SelectOp optimization |
| enable_optimize_denominator_with_broadcast | [ bool](#bool) | Enable optimize x/bcast(y) -> x * bcast(1/y) |
| disable_deallocation_insertion | [ bool](#bool) | Disable deallocation insertion pass |
| disable_partial_sort_optimization | [ bool](#bool) | Disable sort->topk rewrite when only partial sort is required |
 <!-- end Fields -->
 <!-- end HasFields -->


### ExecutableProto
The executable format accepted by SPU runtime.

- Inputs should be prepared before running executable.
- Output is maintained after execution, and can be fetched by output name.

```python
  rt = spu.Runtime(...)            # create an spu runtime.
  rt.set_var('x', ...)             # set variable to the runtime.
  exe = spu.ExecutableProto(       # prepare the executable.
          name = 'balabala',
          input_names = ['x'],
          output_names = ['y'],
          code = ...)
  rt.run(exe)                      # run the executable.
  y = rt.get_var('y')              # get the executable from spu runtime.
```


| Field | Type | Description |
| ----- | ---- | ----------- |
| name | [ string](#string) | The name of the executable. |
| input_names | [repeated string](#string) | The input names. |
| output_names | [repeated string](#string) | The output names. |
| code | [ bytes](#bytes) | The bytecode of the program, with format IR_MLIR_SPU. |
 <!-- end Fields -->
 <!-- end HasFields -->


### RuntimeConfig
The SPU runtime configuration.


| Field | Type | Description |
| ----- | ---- | ----------- |
| protocol | [ ProtocolKind](#protocolkind) | The protocol kind. |
| field | [ FieldType](#fieldtype) | The field type. |
| fxp_fraction_bits | [ int64](#int64) | Number of fraction bits of fixed-point number. 0(default) indicates implementation defined. |
| max_concurrency | [ int32](#int32) | Max number of cores |
| enable_action_trace | [ bool](#bool) | When enabled, runtime prints verbose info of the call stack, debug purpose only. |
| enable_type_checker | [ bool](#bool) | When enabled, runtime checks runtime type infos against the compile-time ones, exceptions are raised if mismatches happen. Note: Runtime outputs prefer runtime type infos even when flag is on. |
| enable_pphlo_trace | [ bool](#bool) | When enabled, runtime prints executed pphlo list, debug purpose only. |
| enable_runtime_snapshot | [ bool](#bool) | When enabled, runtime dumps executed executables in the dump_dir, debug purpose only. |
| snapshot_dump_dir | [ string](#string) | none |
| enable_pphlo_profile | [ bool](#bool) | When enabled, runtime records detailed pphlo timing data, debug purpose only. WARNING: the `send bytes` information is only accurate when `experimental_enable_inter_op_par` and `experimental_enable_intra_op_par` options are disabled. |
| enable_hal_profile | [ bool](#bool) | When enabled, runtime records detailed hal timing data, debug purpose only. WARNING: the `send bytes` information is only accurate when `experimental_enable_inter_op_par` and `experimental_enable_intra_op_par` options are disabled. |
| public_random_seed | [ uint64](#uint64) | The public random variable generated by the runtime, the concrete prg function is implementation defined. Note: this seed only applies to `public variable` only, it has nothing to do with security. |
| share_max_chunk_size | [ uint64](#uint64) | max chunk size for Value::toProto default: 128 * 1024 * 1024 |
| fxp_div_goldschmidt_iters | [ int64](#int64) | The iterations use in f_div with Goldschmidt method. 0(default) indicates implementation defined. |
| fxp_exp_mode | [ RuntimeConfig.ExpMode](#runtimeconfigexpmode) | The exponent approximation method. |
| fxp_exp_iters | [ int64](#int64) | Number of iterations of `exp` approximation, 0(default) indicates impl defined. |
| fxp_log_mode | [ RuntimeConfig.LogMode](#runtimeconfiglogmode) | The logarithm approximation method. |
| fxp_log_iters | [ int64](#int64) | Number of iterations of `log` approximation, 0(default) indicates impl-defined. |
| fxp_log_orders | [ int64](#int64) | Number of orders of `log` approximation, 0(default) indicates impl defined. |
| sigmoid_mode | [ RuntimeConfig.SigmoidMode](#runtimeconfigsigmoidmode) | The sigmoid function approximation model. |
| enable_lower_accuracy_rsqrt | [ bool](#bool) | Enable a simpler rsqrt approximation |
| sine_cosine_iters | [ int64](#int64) | Sine/Cosine approximation iterations |
| beaver_type | [ RuntimeConfig.BeaverType](#runtimeconfigbeavertype) | beaver config, works for semi2k and spdz2k for now. |
| ttp_beaver_config | [ TTPBeaverConfig](#ttpbeaverconfig) | TrustedThirdParty configs. |
| cheetah_2pc_config | [ CheetahConfig](#cheetahconfig) | Cheetah 2PC configs. |
| trunc_allow_msb_error | [ bool](#bool) | For protocol like SecureML, the most significant bit may have error with low probability, which lead to huge calculation error. |
| experimental_disable_mmul_split | [ bool](#bool) | Experimental: DO NOT USE |
| experimental_enable_inter_op_par | [ bool](#bool) | Inter op parallel, aka, DAG level parallel. |
| experimental_enable_intra_op_par | [ bool](#bool) | Intra op parallel, aka, hal/mpc level parallel. |
| experimental_disable_vectorization | [ bool](#bool) | Disable kernel level vectorization. |
| experimental_inter_op_concurrency | [ uint64](#uint64) | Inter op concurrency. |
| experimental_enable_colocated_optimization | [ bool](#bool) | Enable use of private type |
 <!-- end Fields -->
 <!-- end HasFields -->


### ShapeProto



| Field | Type | Description |
| ----- | ---- | ----------- |
| dims | [repeated int64](#int64) | none |
 <!-- end Fields -->
 <!-- end HasFields -->


### TTPBeaverConfig



| Field | Type | Description |
| ----- | ---- | ----------- |
| server_host | [ string](#string) | TrustedThirdParty beaver server's remote ip:port or load-balance uri. |
| adjust_rank | [ int32](#int32) | which rank do adjust rpc call, usually choose the rank closer to the server. |
| asym_crypto_schema | [ string](#string) | asym_crypto_schema: support ["SM2"] Will support 25519 in the future, after yacl supported it. |
| server_public_key | [ bytes](#bytes) | server's public key |
 <!-- end Fields -->
 <!-- end HasFields -->


### ValueChunkProto
The spu Value proto, used for spu value serialization.


| Field | Type | Description |
| ----- | ---- | ----------- |
| total_bytes | [ uint64](#uint64) | chunk info |
| chunk_offset | [ uint64](#uint64) | none |
| content | [ bytes](#bytes) | chunk bytes |
 <!-- end Fields -->
 <!-- end HasFields -->


### ValueMetaProto



| Field | Type | Description |
| ----- | ---- | ----------- |
| data_type | [ DataType](#datatype) | The data type. |
| is_complex | [ bool](#bool) | none |
| visibility | [ Visibility](#visibility) | The data visibility. |
| shape | [ ShapeProto](#shapeproto) | The shape of the value. |
| storage_type | [ string](#string) | The storage type, defined by the underline evaluation engine. i.e. `aby3.AShr<FM64>` means an aby3 arithmetic share in FM64. usually, the application does not care about this attribute. |
 <!-- end Fields -->
 <!-- end HasFields -->
 <!-- end messages -->

## Enums


### CheetahOtKind


| Name | Number | Description |
| ---- | ------ | ----------- |
| YACL_Ferret | 0 | none |
| YACL_Softspoken | 1 | none |
| EMP_Ferret | 2 | none |




### DataType
The SPU datatype

| Name | Number | Description |
| ---- | ------ | ----------- |
| DT_INVALID | 0 | none |
| DT_I1 | 1 | 1bit integer (bool). |
| DT_I8 | 2 | int8 |
| DT_U8 | 3 | uint8 |
| DT_I16 | 4 | int16 |
| DT_U16 | 5 | uint16 |
| DT_I32 | 6 | int32 |
| DT_U32 | 7 | uint32 |
| DT_I64 | 8 | int64 |
| DT_U64 | 9 | uint64 |
| DT_F16 | 10 | half |
| DT_F32 | 11 | float |
| DT_F64 | 12 | double |




### FieldType
A security parameter type.

The secure evaluation is based on some algebraic structure (ring or field),

| Name | Number | Description |
| ---- | ------ | ----------- |
| FT_INVALID | 0 | none |
| FM32 | 1 | Ring 2^32 |
| FM64 | 2 | Ring 2^64 |
| FM128 | 3 | Ring 2^128 |




### ProtocolKind
The protocol kind.

| Name | Number | Description |
| ---- | ------ | ----------- |
| PROT_INVALID | 0 | Invalid protocol. |
| REF2K | 1 | The reference implementation in `ring^2k`, note: this 'protocol' only behave-like a fixed point secure protocol without any security guarantee. Hence, it should only be selected for debugging purposes. |
| SEMI2K | 2 | A semi-honest multi-party protocol. This protocol requires a trusted third party to generate the offline correlated randoms. Currently, SecretFlow by default ships this protocol with a trusted first party. Hence, it should only be used for debugging purposes. |
| ABY3 | 3 | A honest majority 3PC-protocol. SecretFlow provides the semi-honest implementation without Yao. |
| CHEETAH | 4 | The famous [Cheetah](https://eprint.iacr.org/2022/207) protocol, a very fast 2PC protocol. |
| SECURENN | 5 | A semi-honest 3PC-protocol for Neural Network, P2 as the helper, (https://eprint.iacr.org/2018/442) |




### PtType
Plaintext type

SPU runtime does not process with plaintext directly, plaintext type is
mainly used for IO purposes, when converting a plaintext buffer to an SPU
buffer, we have to let spu know which type the plaintext buffer is.

| Name | Number | Description |
| ---- | ------ | ----------- |
| PT_INVALID | 0 | none |
| PT_I8 | 1 | int8_t |
| PT_U8 | 2 | uint8_t |
| PT_I16 | 3 | int16_t |
| PT_U16 | 4 | uint16_t |
| PT_I32 | 5 | int32_t |
| PT_U32 | 6 | uint32_t |
| PT_I64 | 7 | int64_t |
| PT_U64 | 8 | uint64_t |
| PT_I128 | 9 | int128_t |
| PT_U128 | 10 | uint128_t |
| PT_I1 | 11 | bool |
| PT_F16 | 30 | half |
| PT_F32 | 31 | float |
| PT_F64 | 32 | double |
| PT_CF32 | 50 | complex float |
| PT_CF64 | 51 | complex double |




### RuntimeConfig.BeaverType


| Name | Number | Description |
| ---- | ------ | ----------- |
| TrustedFirstParty | 0 | Assume first party (rank0) as trusted party to generate beaver triple. WARNING: It is NOT SAFE and SHOULD NOT BE used in production. |
| TrustedThirdParty | 1 | Generate beaver triple through an additional trusted third party. |
| MultiParty | 2 | Generate beaver triple through multi-party. |




### RuntimeConfig.ExpMode
The exponential approximation method.

| Name | Number | Description |
| ---- | ------ | ----------- |
| EXP_DEFAULT | 0 | Implementation defined. |
| EXP_PADE | 1 | The pade approximation. |
| EXP_TAYLOR | 2 | Taylor series approximation. |




### RuntimeConfig.LogMode
The logarithm approximation method.

| Name | Number | Description |
| ---- | ------ | ----------- |
| LOG_DEFAULT | 0 | Implementation defined. |
| LOG_PADE | 1 | The pade approximation. |
| LOG_NEWTON | 2 | The newton approximation. |
| LOG_MINMAX | 3 | The minmax approximation. |




### RuntimeConfig.SigmoidMode
The sigmoid approximation method.

| Name | Number | Description |
| ---- | ------ | ----------- |
| SIGMOID_DEFAULT | 0 | Implementation defined. |
| SIGMOID_MM1 | 1 | Minmax approximation one order. f(x) = 0.5 + 0.125 * x |
| SIGMOID_SEG3 | 2 | Piece-wise simulation. f(x) = 0.5 + 0.125x if -4 <= x <= 4 1 if x > 4 0 if -4 > x |
| SIGMOID_REAL | 3 | The real definition, which depends on exp's accuracy. f(x) = 1 / (1 + exp(-x)) |




### SourceIRType
Compiler relate definition
////////////////////////////////////////////////////////////////////////

| Name | Number | Description |
| ---- | ------ | ----------- |
| XLA | 0 | none |
| STABLEHLO | 1 | none |




### Visibility
The visibility type.

SPU is a secure evaluation runtime, but not all data are secret, some of them
are publicly known to all parties, marking them as public will improve
performance significantly.

| Name | Number | Description |
| ---- | ------ | ----------- |
| VIS_INVALID | 0 | none |
| VIS_SECRET | 1 | Invisible(unknown) for all or some of the parties. |
| VIS_PUBLIC | 2 | Visible(public) for all parties. |
| VIS_PRIVATE | 3 | Visible for only one party |




### XLAPrettyPrintKind


| Name | Number | Description |
| ---- | ------ | ----------- |
| TEXT | 0 | none |
| DOT | 1 | none |
| HTML | 2 | none |


 <!-- end Enums -->
 <!-- end Files -->

## Scalar Value Types

| .proto Type | Notes | C++ Type | Java Type | Python Type |
| ----------- | ----- | -------- | --------- | ----------- |
| <div><h4 id="double" /></div><a name="double" /> double |  | double | double | float |
| <div><h4 id="float" /></div><a name="float" /> float |  | float | float | float |
| <div><h4 id="int32" /></div><a name="int32" /> int32 | Uses variable-length encoding. Inefficient for encoding negative numbers – if your field is likely to have negative values, use sint32 instead. | int32 | int | int |
| <div><h4 id="int64" /></div><a name="int64" /> int64 | Uses variable-length encoding. Inefficient for encoding negative numbers – if your field is likely to have negative values, use sint64 instead. | int64 | long | int/long |
| <div><h4 id="uint32" /></div><a name="uint32" /> uint32 | Uses variable-length encoding. | uint32 | int | int/long |
| <div><h4 id="uint64" /></div><a name="uint64" /> uint64 | Uses variable-length encoding. | uint64 | long | int/long |
| <div><h4 id="sint32" /></div><a name="sint32" /> sint32 | Uses variable-length encoding. Signed int value. These more efficiently encode negative numbers than regular int32s. | int32 | int | int |
| <div><h4 id="sint64" /></div><a name="sint64" /> sint64 | Uses variable-length encoding. Signed int value. These more efficiently encode negative numbers than regular int64s. | int64 | long | int/long |
| <div><h4 id="fixed32" /></div><a name="fixed32" /> fixed32 | Always four bytes. More efficient than uint32 if values are often greater than 2^28. | uint32 | int | int |
| <div><h4 id="fixed64" /></div><a name="fixed64" /> fixed64 | Always eight bytes. More efficient than uint64 if values are often greater than 2^56. | uint64 | long | int/long |
| <div><h4 id="sfixed32" /></div><a name="sfixed32" /> sfixed32 | Always four bytes. | int32 | int | int |
| <div><h4 id="sfixed64" /></div><a name="sfixed64" /> sfixed64 | Always eight bytes. | int64 | long | int/long |
| <div><h4 id="bool" /></div><a name="bool" /> bool |  | bool | boolean | boolean |
| <div><h4 id="string" /></div><a name="string" /> string | A string must always contain UTF-8 encoded or 7-bit ASCII text. | string | String | str/unicode |
| <div><h4 id="bytes" /></div><a name="bytes" /> bytes | May contain any arbitrary sequence of bytes. | string | ByteString | str |
