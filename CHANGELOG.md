# CHANGELOGS

> Instrument:
>
> - Add `[Feature]` prefix for new features
> - Add `[Bugfix]` prefix for bug fixes
> - Add `[API]` prefix for API changes

## staging
>
> please add your unreleased change here.

## 20230614

- [SPU] 0.4.0 release
- [Feature] Improve secret dynamic slicing performance
- [Feature] Improve accuracy of tanh approximation
- [Example] Add flax GPT2
- [Example] Add ResNet
- [Bugfix] Fix gen cache in unbalanced psi
- [Bugfix] Fix possible compilation cache collision
- [Improvement] Improve accuracy of mean with large size input
- [Improvement] Improve support of Jax 64bit mode
- [API] Add compiler options

## 20230406

- [SPU] 0.3.2 release
- [Feature] Add TrustedThirdParty beaver provider for semi2k
- [Feature] Expose ssl/tls options
- [Feature] Add EpsilonOp
- [Feature] Support CaseOp
- [Feature] Improve sort performance
- [Feature] Improve shift performance
- [Feature] Support shift by secret number of bits
- [Feature] Support secret indexing
- [Feature] Add PIR python binding
- [bugfix] Fix boolean ConstantOp
- [bugfix] Fix jnp.median
- [bugfix] Fix jnp.sort on floating point inputs
- [bugfix] Fix secret sort with public payloads
- [3p] Relax TensorFlow version in requirements.txt
- [3p] Move to OpenXLA
- [API] Move C++ API from spu to libspu
- [PSI] Add ecdh-oprf 2-party two stage mode.

## 20230104

- [SPU] 0.3.1 release
- [API] Add get_var_meta
- [API] Change ValueProto to bytes in Runtime/IO binding API
- [Feature] Add SPU logging
- [Feature] ECDH-PSI supports white box interconnection mode
- [Feature] Various performance improvements
- [Feature] Lift IfOp condition limit
- [bugfix] Fix various crashes
- [3p] Build with Tensorflow 2.11.0
- [3p] Update JAX to 0.4.1

## 20221116

- [SPU] 0.3.0 release
- [API] Add IR type for frontend compilation
- [Feature] Support SignOp, Expm1Op, DotGeneralOp
- [Feature] Improve convolution performance
- [Feature] Link Send/Recv python binding
- [bugfix] Relax iota type constraint
- [bugfix] Fix type inference for whileOp/dynamicUpdateSliceOp
- [bugfix] Relax clamp type constraint
- [bugfix] Fix NotOp with non-pred integer types

## 20220905

- [SPU] 0.2.4 release
- [bugfix] Fix Tensorflow example
- [miscs] Performance improvements

## 20220822

- [SPU] 0.2.1 release
- [Feature] Add compiler cache
- [Feature] Support bc22-pcg psi
- [API] Refactor psi api
- [miscs] Various performance improvements

## 20220727

- [SPU] 0.1.1 release
- [API] Added rsqrt support
- [bugfix] Fixed some crashes and wrong answers

## 20220527

- [SPU] 0.1.0 release
- [API] Add exp/log approximation configurations.
- [API] Change SigmoidMode enum value.

## 20220513

- [SPU] 0.0.10
- [Infra] Set minimum macOS requirements to 11.0
- [Feature] Performance improvements
- [Feature] Support SelectAndScatter.

## 20220511

- [SPU] 0.0.9 release
- [Feature] Add spu.binding.distributed.
- [API] Expose visibility to ValueProto.
- [bugfix] Fixed several vtype and dtype mismatches.
- [bugfix] Improve multiply accuracy under certain scenarios.
- [bugfix] Ensure dynamic-slice & dynamic-update-slice OOB behavior is aligned with XLA semantics.

## 20220427

- [SPU] 0.0.8.1 release
- [bugfix] Fix iota type mismatch.

## 20220425

- [SPU] 0.0.8 release
- [Feature] Add tanh support.
- [bugfix] Fix secret shift wrong answer.

## 20220424

- [SPU] 0.0.7.1 release
- [bugfix] Fix secret shift.

## 20220422

- [SPU] 0.0.7 release
- [Feature] Add python simulation module.
- [Feature] Support general sort comparator.
- [Feature] Support remainder operation.
- [Feature] Support arithmetic right shift operation.
- [API] Change DataType definition, support more accurate encoding.

## 20220412.1

- [SPU] 0.0.6.1 release
- [bugfix] Fixed several macOS build issues.

## 20220412

- [SPU] 0.0.6 release
- [API] Add storage_type/data_type, remove type_data.
- [Experimental] Log(natural logarithm) is calculated with Pade approximation
                 instead of Householder's approximation.
- [API] Fix typo, goldschmdit.
- [Improvement] fxp div, improvement accuracy with normal distributed inputs.

## 20220325

- [SPU] 0.0.5.1 release
- [Bugfix] Fix SEMI2K divivsion wrong answer

## 20220324

- [SPU] 0.0.5 release
- [Improvement] Lift Convolution limitations.
- [Improvement] Optimize maximum/minimum speed, with MPC B2A single bit conversion.
- [Experimental] Div is calculated with direct goldschmdit method instead of
                 multiplication of dividend and divisor reciprocal.
- [API] change RuntimeConfig.enable_op_time_profile to enable_pphlo_profile.
- [API] change RuntimeConfig.fxp_reciprocal_goldschmdit_iters to fxp_div_goldschmdit_iters.

## 20220308

- [SPU] 0.0.4 release
- [Feature] add silent ot support for various ot scenarios (chosen/correlated/random
            messages, chosen/correlated/random choices, 1o2/1oN)
- [Feature] add non-linear computation protocols based on silent ot (comparison,
            truncation, b2a, triple, randbit, etc)
- [Feature] add a 2PC protocol: Cheetah
- [Improvement] concatenate is a lot faster
- [API] add RuntimeConfig.enable_op_time_profile to ask SPU to collect timing profiling data
- [Bugfix] fixed pphlo.ConstOp may lose signed bit after encoding

## 20220303

- [spu] 0.0.3 release
- [API] merge (config.proto, executable.proto, types.proto) into single spu.proto.
- [API] change RuntimeConfig.enable_protocol_trace to enable_action_trace.
- [API] change RuntimeConfig.fxp_recirptocal_goldschmdit_iters to fxp_reciprocal_goldschmdit_iters.
- [API] add RuntimConfig.reveal_secret_condition to allow reveal secret control flow condition.
- [Bugfix] Fixed SEGV when reconstruct from an ABY3 scalar secret
- [Feature] Left/right shift now properly supports non-scalar inputs

## 20220217

- [spu] 0.0.2.3 release

## 20220211

- [spu] 0.0.2.2 release
- [Bugfix] Fix exception when kkrt psi input is too small

## 20220210

- [spu] 0.0.2.1 release
- [Bugfix] Fix exception when streaming psi output directory already exists

## 20220209

- [spu] 0.0.2 release.
- [Feature] Support multi-parties psi

## 20210930

- [spu] Init release.
