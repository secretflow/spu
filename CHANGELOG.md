# CHANGELOGS

> Instrument:
>
> - Add `[Feature]` prefix for new features
> - Add `[Bugfix]` prefix for bug fixes
> - Add `[API]` prefix for API changes

## staging
> please add your unrelease change here.

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

## 20220412
- [SPU] 0.0.6.1 release
- [bugfix] Fixed several macOS build issues.

## 20220412

- [SPU] 0.0.6 release
- [API] Add storage_type/data_type, remove type_data.
- [Experimental] Log(natural logarithm) is calculated with Pade approximation instead of Householder's approximation.
- [API] Fix typo, goldschmdit.
- [Improvement] fxp div, improvement accuracy with normal distributed inputs.

## 20220325

- [SPU] 0.0.5.1 release
- [Bugfix] Fix SEMI2K divivsion wrong answer

## 20220324

- [SPU] 0.0.5 release
- [Improvement] Lift Convolution limitations.
- [Improvement] Optimize maximum/minimum speed, with MPC B2A single bit conversion.
- [Experimental] Div is calculated with direct goldschmdit method instead of multiplication of dividend and divisor reciprocal.
- [API] change RuntimeConfig.enable_op_time_profile to enable_pphlo_profile.
- [API] change RuntimeConfig.fxp_reciprocal_goldschmdit_iters to fxp_div_goldschmdit_iters.

## 20220308

- [SPU] 0.0.4 release
- [Feature] add silent ot support for various ot scenarios (chosen/correlated/random messages, chosen/correlated/random choices, 1o2/1oN)
- [Feature] add non-linear computation protocols based on silent ot (comparison, truncation, b2a, triple, randbit, etc)
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
