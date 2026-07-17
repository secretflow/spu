# SPU: Secure Processing Unit

[![CircleCI](https://dl.circleci.com/status-badge/img/gh/secretflow/spu/tree/main.svg?style=shield)](https://dl.circleci.com/status-badge/redirect/gh/secretflow/spu/tree/main)
[![Python](https://img.shields.io/pypi/pyversions/spu.svg)](https://pypi.org/project/spu/)
[![PyPI version](https://img.shields.io/pypi/v/spu)](https://pypi.org/project/spu/)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/secretflow/spu/badge)](https://securityscorecards.dev/viewer/?uri=github.com/secretflow/spu)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/8311/badge)](https://www.bestpractices.dev/projects/8311)

SPU (Secure Processing Unit) aims to be a `provable`, `measurable` secure computation device,
which provides computation ability while keeping your private data protected.

SPU could be treated as a programmable device, it's not designed to be used directly.
Normally we use SecretFlow framework, which use SPU as the underline secure computing device.

Currently, we mainly focus on `provable` security. It contains a secure runtime that evaluates
[XLA](https://www.tensorflow.org/xla/operation_semantics)-like tensor operations,
which use [MPC](https://en.wikipedia.org/wiki/Secure_multi-party_computation) as the underline
evaluation engine to protect privacy information.

SPU python package also contains a simple distributed module to demo SPU usage,
but it's **NOT designed for production** due to system security and performance concerns,
please **DO NOT** use it directly in production.

## Contribution Guidelines

If you would like to contribute to SPU, please check [Contribution guidelines](CONTRIBUTING.md).

If you would like to use SPU for research purposes, please check [research development guidelines](docs/SPU_gudience.pdf) from [@fionser](https://github.com/fionser).

This documentation also contains instructions for [build and testing](CONTRIBUTING.md#build).

## Installation Guidelines

### Supported platforms

|            | Linux x86_64 | Linux aarch64 | macOS x64      | macOS Apple Silicon | Windows x64    | Windows WSL2    x64 |
|------------|--------------|---------------|----------------|---------------------|----------------|---------------------|
| CPU        | yes          | yes           | yes<sup>1</sup>| yes                 | no             | yes                 |
| NVIDIA GPU | experimental | no            | no             | n/a                 | no             | experimental        |

1. Due to CI resource limitation, macOS x64 prebuild binary is no longer available.

### Instructions

Please follow [Installation Guidelines](INSTALLATION.md) to install SPU.

### Hardware Requirements

| General Features | FourQ based PSI | GPU |
| ---------------- | --------------- | --- |
| AVX/ARMv8        | AVX2/ARMv8      | CUDA 11.8+ |

## Citing SPU

If you think SPU is helpful for your research or development, please consider citing our papers:

[USENIX ATC'23](https://www.usenix.org/conference/atc23/presentation/ma)

```text
@inproceedings {spu,
    author = {Junming Ma and Yancheng Zheng and Jun Feng and Derun Zhao and Haoqi Wu and Wenjing Fang and Jin Tan and Chaofan Yu and Benyu Zhang and Lei Wang},
    title = {{SecretFlow-SPU}: A Performant and {User-Friendly} Framework for {Privacy-Preserving} Machine Learning},
    booktitle = {2023 USENIX Annual Technical Conference (USENIX ATC 23)},
    year = {2023},
    isbn = {978-1-939133-35-9},
    address = {Boston, MA},
    pages = {17--33},
    url = {https://www.usenix.org/conference/atc23/presentation/ma},
    publisher = {USENIX Association},
    month = jul,
}
```

[ICML'24](https://proceedings.mlr.press/v235/wu24d.html)

```text
@inproceedings{ditto,
  title = {Ditto: Quantization-aware Secure Inference of Transformers upon {MPC}},
  author = {Wu, Haoqi and Fang, Wenjing and Zheng, Yancheng and Ma, Junming and Tan, Jin and Wang, Lei},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning},
  pages = {53346--53365},
  year = {2024},
  editor = {Salakhutdinov, Ruslan and Kolter, Zico and Heller, Katherine and Weller, Adrian and Oliver, Nuria and Scarlett, Jonathan and Berkenkamp, Felix},
  volume = {235},
  series = {Proceedings of Machine Learning Research},
  month = {21--27 Jul},
  publisher = {PMLR},
  pdf = {https://raw.githubusercontent.com/mlresearch/v235/main/assets/wu24d/wu24d.pdf},
  url = {https://proceedings.mlr.press/v235/wu24d.html},
  abstract = {Due to the rising privacy concerns on sensitive client data and trained models like Transformers, secure multi-party computation (MPC) techniques are employed to enable secure inference despite attendant overhead. Existing works attempt to reduce the overhead using more MPC-friendly non-linear function approximations. However, the integration of quantization widely used in plaintext inference into the MPC domain remains unclear. To bridge this gap, we propose the framework named Ditto to enable more efficient quantization-aware secure Transformer inference. Concretely, we first incorporate an MPC-friendly quantization into Transformer inference and employ a quantization-aware distillation procedure to maintain the model utility. Then, we propose novel MPC primitives to support the type conversions that are essential in quantization and implement the quantization-aware MPC execution of secure quantized inference. This approach significantly decreases both computation and communication overhead, leading to improvements in overall efficiency. We conduct extensive experiments on Bert and GPT2 models to evaluate the performance of Ditto. The results demonstrate that Ditto is about $3.14\sim 4.40\times$ faster than MPCFormer (ICLR 2023) and $1.44\sim 2.35\times$ faster than the state-of-the-art work PUMA with negligible utility degradation.}
}
```

## Acknowledgement

We thank the significant contributions made by [Alibaba Gemini Lab](https://alibaba-gemini-lab.github.io) and security advisories made by [VUL337@NISL@THU](https://netsec.ccert.edu.cn/vul337).

## HOPTA Setup

### Prerequisite

Setup with docker is recommended.
#### Docker

```sh
## start container
docker run -d -it --name spu-dev-$(whoami) \
         --mount type=bind,source="$(pwd)",target=/home/admin/dev/ \
         -w /home/admin/dev \
         --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
         --cap-add=NET_ADMIN \
         --privileged=true \
         --entrypoint="bash" \
         secretflow/ubuntu-base-ci:latest

# attach to build container
docker exec -it spu-dev-$(whoami) bash

pip install -r requirements.txt
pip install -r requirements-dev.txt
```

#### Linux

```sh
Install gcc>=11.2, cmake>=3.26, ninja, nasm>=2.15, python>=3.9, bazelisk, xxd, lld
```

About the commands used to install the above dependencies, you can follow [Ubuntu docker file](https://github.com/secretflow/devtools/blob/main/dockerfiles/ubuntu-base-ci.DockerFile).

```sh
python3 -m pip install -r requirements.txt
python3 -m pip install -r requirements-dev.txt
```

#### macOS

```sh
# macOS >= 13.0, Xcode >= 15.0

# Install Xcode
https://apps.apple.com/us/app/xcode/id497799835?mt=12

# Select Xcode toolchain version
sudo xcode-select -s /Applications/Xcode.app/Contents/Developer

# Install homebrew
https://brew.sh/

# Install dependencies
# Be aware, brew may install a newer version of bazel, when that happens bazel will give an error message during build.
# Please follow instructions in the error message to install the required version
brew install bazelisk cmake ninja libomp wget

# For Intel mac only
brew install nasm

# Install python dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### Build & UnitTest

``` sh
# build as debug
bazel build //... -c dbg

# build as release
bazel build //... -c opt

# test
bazel test //...
```

### Bazel build options

- `--define gperf=on` enable gperf
- `--define tracelog=on` enable link trace log.

### Install StableHLO

```sh
pip install stablehlo -f https://github.com/openxla/stablehlo/releases/expanded_assets/v1.0.0
```

### Fix a GLIBCXX import error during `bazel test //...`

If the following error occurs:

```text
ImportError: /root/miniconda3/bin/../lib/libstdc++.so.6: version `GLIBCXX_3.4.30' not found
```

Go to the `lib` directory of the Python environment. For example, the Docker
image uses `/root/miniconda3/lib`:

```sh
cd /root/miniconda3/lib
mv libstdc++.so.6 libstdc++.so.6.old
ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6 libstdc++.so.6
```

### Tips

If a build runs out of resources, limit the number of concurrent jobs with
`--jobs=4`.

## Workflow

The complete workflow has three stages:

1. **Preparation**: discover XLA passes, expose them as controllable SPU
   compiler options, and prepare parallel test copies.
2. **PassTester**: run the baseline and individual pass configurations in
   batches, then extract communication costs, execution time, and IR data.
3. **PatternExtractor**: select a valid performance difference, capture its XLA
   HLO, and use delta debugging to reduce it to a minimal triggering pattern.

The commands below assume that they start in the repository root. Unless noted
otherwise, run the Python scripts from `pass-test`, because they use relative
paths.

### Stage 1: Preparation

The preparation stage is normally required only when setting up the environment
for the first time, changing the XLA revision, or regenerating test copies. Skip
a step if its generated artifacts are already present and up to date.

#### 1. Discover XLA pass information

Clone XLA into the repository root and check out the revision used by the
current experiment:

```sh
git clone https://github.com/openxla/xla.git
cd xla
git checkout 64bdcc53a1b24abf19b1fe598e6f9b0fe6454470
cd ../pass-test
python _0_get_xla_pass_inf.py
```

`_0_get_xla_pass_inf.py` scans `../xla/xla/service` and primarily generates:

- `pass_inf/extract_XLAPass.json`: discovered XLA pass information.
- `pass_inf/extract_AlgebraOption.json`: algebraic simplification option
  information.

#### 2. Add pass compiler options to SPU

```sh
cd pass-test
python _0_pass_adder_modifier.py
cd ..
bazel build //... -c opt
```

`_0_pass_adder_modifier.py` adds pass switches to the SPU compiler options and
HLO importer. It also generates pass lists for batch testing, such as
`pass_inf/pass_options_HLO_delete.txt`. Because the script modifies SPU source
and BUILD files, rebuild SPU after running it.

#### 3. Generate parallel test copies

Skip this step when using a single test process. For parallel testing, the
current configuration recommends seven additional SML copies:

```sh
cd pass-test
python _0_extend_testcase.py
cd ..
bazel build //... -c opt
```

Based on `extend_number`, `_0_extend_testcase.py` copies `sml` to `sml1`
through `sml7` and updates their imports and Bazel targets. The script
overwrites existing copies with the same names, and a rebuild is required
afterward. The original `sml` plus seven copies correspond to
`thread_num = 8` in PassTester.

After preparation, verify that at least the following artifacts exist:

```text
bazel-bin/sml/...
pass-test/file-to-be-modified/sml/...
pass-test/pass_inf/pass_options_HLO_delete.txt
```

For parallel execution, `bazel-bin/sml1` through `bazel-bin/sml7` must also
exist.

### Stage 2: PassTester

PassTester consists of `_1_auto_test_pass.py` and `_2_extract_inf.py`. The
first script runs experiments and saves raw logs; the second converts those
logs into JSON files suitable for analysis.

#### 1. Configure and run `_1_auto_test_pass.py`

Review the configuration near the start of the script:

```python
test_mod = mod.sml
repeat_time = 1
output_folder = "test-log/SEMI2K-all"
thread_num = 8
protocal = "SEMI2K"
passoptions_path = "pass_inf/pass_options_HLO_delete.txt"
hlo_log = False
refer_to_complete_test = False
```

The options have the following meanings:

| Option | Description |
| --- | --- |
| `test_mod` | Selects the test collection, normally `mod.sml`. |
| `repeat_time` | Number of repetitions per configuration. Set `thread_num = 1` when this is greater than 1. |
| `output_folder` | Dedicated output directory for this run. Use a new directory to avoid overwriting existing results. |
| `thread_num` | Number of parallel workers. It must not exceed the number of prepared SML copies. |
| `protocal` | MPC protocol: `SEMI2K`, `ABY3`, or `CHEETAH`, depending on the experiment. The spelling matches the variable in the script. |
| `passoptions_path` | Pass configuration list used by this run. |
| `hlo_log` | Whether to print XLA HLO. Leave it disabled for ordinary pass testing and enable it when capturing a PatternExtractor case. |
| `refer_to_complete_test` | Whether to filter tasks using an existing complete test result. |

To test only selected files, also narrow `test_file_list` in the script. Then
run:

```sh
cd pass-test
python _1_auto_test_pass.py
```

The script performs the following operations:

1. Reads test templates containing placeholders from
   `file-to-be-modified/sml`.
2. Replaces `{partynum}`, `{protocalchosen}`, `{pphlo_dict}`, and
   `{skip_control}`, then inserts the current pass compiler option.
3. Writes each instantiated file to the current worker's `sml`, `sml1`, or
   another test-copy directory.
4. Runs the baseline first, followed by every pass configuration.
5. Records PPHLO and skips redundant results whose IR is identical to the
   baseline.

A typical output tree is:

```text
pass-test/test-log/SEMI2K-all/
├── test-log.txt
└── <test_name>/
    ├── baseline/
    │   ├── test_*.txt
    │   └── extract_result.json
    ├── IR_record/
    │   └── pphlo_<function_name>.json
    └── <pass_option>/
        └── test_*.txt
```

`test-log.txt` is the scheduler log, `test_*.txt` contains the raw output
from individual experiments, and `IR_record` identifies which passes actually
changed each function's PPHLO.

#### 2. Extract experiment data

Review the configuration in the main section of `_2_extract_inf.py`:

```python
party_nums = 2
log_directory = "test-log/SEMI2K-all"
```

`SEMI2K` and `CHEETAH` normally use `party_nums = 2`; `ABY3` uses
`party_nums = 3`. After configuring the script, run:

```sh
cd pass-test
python _2_extract_inf.py
```

The script splits raw logs by party and extracts per-function execution time,
bytes sent, send actions, PPHLO, and related data. It writes a merged result
under each test directory:

```text
test-log/SEMI2K-all/<test_name>/extract_result.json
```

This file is indexed by function and pass configuration and is the primary
input for selecting performance differences.

If the terminal reports:

```text
Function <function_name> in <pass_directory> does not have log file
```

the pass changed the IR but did not produce a complete function profile. Common
causes include a failed test, compilation failure, process crash, or truncated
log. Such a configuration is not merged into the final JSON. A small number of
these messages can occur when disabling passes in bulk, but the message alone
does not establish that the overall test run is healthy.

The valid outputs of the PassTester stage are baseline and pass logs whose
correctness checks passed, together with the merged `extract_result.json` in
each test directory.

### Stage 3: PatternExtractor

PatternExtractor uses `_3_code_reduce_case_gen.py` to generate a reduction
case. `_4_code_reducer.py` then repeatedly removes HLO instructions to find
the smallest pattern that preserves the target communication difference.

#### 1. Select a reduction candidate

Select a function and pass from the PassTester `extract_result.json` that meet
all of the following requirements:

- Both the baseline and target-pass tests completed successfully.
- The target pass actually changed the PPHLO.
- `send_bytes` or `send_actions` has a stable, reproducible difference.
- The original configuration `ori`, mutated configuration `mut`, protocol,
  and target metric are recorded explicitly.

Do not treat crashes, compilation failures, or incorrect computation results as
performance patterns.

#### 2. Obtain the XLA HLO for reduction

The reducer requires **XLA HLO text** beginning with `HloModule`, not
PPHLO/MLIR beginning with `module @jit_...`. Pass tests record only PPHLO by
default. If an existing `test-log/SEMI2K-all` run contains none of
`HLO_IR|`, `Start printing HLO IR`, or `HloModule`, it cannot directly
provide the reducer input.

Capture HLO separately for the candidate test so that the formal experiment
results are not overwritten. Temporarily configure `_1_auto_test_pass.py` as
follows:

```python
hlo_log = True
output_folder = "test-log/HLO-capture"
test_file_list = ["sml/cluster/tests/kmeans_test.py"]
```

If only the baseline HLO is needed, point `passoptions_path` to an empty file.
XLA HLO is generated before the SPU pass pipeline, so the same baseline HLO can
normally be used to compare the later `ori` and `mut` configurations. Run:

```sh
cd pass-test
python _1_auto_test_pass.py
```

The target section in the raw log has this form:

```text
HLO_IR|Start printing HLO IR for test_kmeans_random
HLO_IR|HloModule jit_test_kmeans_random, ...
HLO_IR|...
HLO_IR|End of printing HLO IR for test_kmeans_random
```

Copy from `HLO_IR|HloModule ...` through the function's final `HLO_IR|}`.
Do not include the Start or End marker. HLO can also be extracted automatically
from a baseline log that contains it:

```sh
cd pass-test
python - <<'PY'
from _2_extract_inf import split_log_party, extract_log

path = "test-log/HLO-capture/kmeans_test/baseline"
split_log_party(path, party_nums=2, pphlo_log=True, hlo_log=True)
extract_log(path, pphlo_log=True, hlo_log=True)
PY
```

The extracted HLO is written to:

```text
test-log/HLO-capture/kmeans_test/baseline/extracted_inf/hlo.txt
```

The corresponding `extract_result.json` also contains an `hlo` field. Both
forms have already had the `HLO_IR|` prefix removed.

#### 3. Generate a reduction case

Paste the complete HLO for the target function into
`_3_code_reduce_case_gen.py` and configure:

```python
case_name = "your-case-name"
pass_option_mut = {
    "ori": [],
    "mut": ["disable_or_enable_some_pass"],
}
protocalchosen = "SEMI2K"  # ABY3 and CHEETAH are also supported
matrix = "send_bytes"      # send_actions is also supported
```

When pasting prefixed text directly from a raw log, use the existing
`test_case_full` format:

```python
test_case_full = """HLO_IR|HloModule ...
HLO_IR|...
HLO_IR|}"""
```

The script removes `HLO_IR|` from every line. When using content already
cleaned in `hlo.txt`, construct `input_test_case` directly instead of
splitting the text on `HLO_IR|`. Then run:

```sh
cd pass-test
python _3_code_reduce_case_gen.py
```

The generated case is:

```text
pass-test/reduce_case/<case_name>.json
```

#### 4. Run HLO reduction

At the end of `_4_code_reducer.py`, set the case name to the value used in the
previous step and select one or more reduction methods:

```python
case_name_list = ["your-case-name"]
ddmethod_list = ["ddmin"]
```

`ddmethod_list` supports only `ddmin`, `onlycomplement`, `CDD`, and
`ProfDD`.

Also verify the extraction-module import at the beginning of the file. When
only `_2_extract_inf.py` is present, use:

```python
from _2_extract_inf import split_log_party, extract_log
```

Confirm that `bazel-bin/spu/tests/hlo_debug` has been built, then run:

```sh
cd pass-test
python _4_code_reducer.py
```

The reducer writes each candidate HLO into the test template associated with
`spu/tests/hlo_debug.py`, runs the `ori` and `mut` configurations, and
uses `matrix` to determine whether the communication difference remains.
Its primary output is:

```text
pass-test/reduce_log/<case_name>/<case_name>-<ddmethod>/
```

This directory contains the reduction trace, triggering records, and final
candidate pattern. `_4_code_reducer.py` deletes an existing reduction-log
directory with the same name. Back up any results that must be preserved and
recheck `case_name_list` before rerunning it.

#### 5. Validate the final pattern

After reduction, rerun both `ori` and `mut` at least once with the final HLO
and verify that:

- Both configurations compile and execute successfully.
- The output is correct.
- The target `send_bytes` or `send_actions` difference remains stable.
- The final HLO is sufficiently small, and removing any essential component
  eliminates the target difference.

After these checks pass, `reduce_case/<case_name>.json`,
`reduce_log/<case_name>/`, and the corresponding raw PassTester logs together
form a reproducible pattern case.
