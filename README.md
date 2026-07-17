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

### 安装stablehlo
pip install stablehlo -f https://github.com/openxla/stablehlo/releases/expanded_assets/v1.0.0

### 在bazel test //...的时候如果出现ImportError: /root/miniconda3/bin/../lib/libstdc++.so.6: version `GLIBCXX_3.4.30' not found
到python环境的lib目录下，比如用镜像的话到/root/miniconda3/lib
cd /root/miniconda3/lib
mv libstdc++.so.6 libstdc++.so.6.old
ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6 libstdc++.so.6

### tips
如果编译阶段出错，可以加上--jobs=4这个参数。

## 正式流程

完整流程分为三个阶段：

1. **准备阶段**：提取 XLA Pass、为 SPU 增加可控制的编译选项，并准备并行测试副本。
2. **PassTester**：批量执行 baseline 和不同 Pass 配置，随后提取通信量、运行时间和 IR 等信息。
3. **PatternExtractor**：从有效的性能差异中选定案例，获取 XLA HLO，并通过差分约简得到最小触发 Pattern。

以下命令默认从仓库主目录开始执行。除特别说明外，Python 脚本都应在
`pass-test` 目录运行，因为脚本中使用了相对路径。

### 阶段一：准备阶段

准备阶段通常只需在首次搭建环境、XLA 版本发生变化或重新生成测试副本时执行。
仓库中已有相应生成结果时，可以跳过对应步骤。

#### 1. 获取 XLA Pass 信息

将 XLA 克隆到仓库主目录，并切换到与当前实验匹配的版本：

```sh
git clone https://github.com/openxla/xla.git
cd xla
git checkout 64bdcc53a1b24abf19b1fe598e6f9b0fe6454470
cd ../pass-test
python _0_get_xla_pass_inf.py
```

`_0_get_xla_pass_inf.py` 扫描 `../xla/xla/service`，主要生成：

- `pass_inf/extract_XLAPass.json`：发现的 XLA Pass 信息；
- `pass_inf/extract_AlgebraOption.json`：代数化简相关选项信息。

#### 2. 为 SPU 增加 Pass 编译选项

```sh
cd pass-test
python _0_pass_adder_modifier.py
cd ..
bazel build //... -c opt
```

`_0_pass_adder_modifier.py` 将 Pass 开关加入 SPU 的编译选项和 HLO importer，
并生成后续批量测试使用的 Pass 列表，例如
`pass_inf/pass_options_HLO_delete.txt`。脚本会修改 SPU 源码和 BUILD 文件，
因此执行后必须重新编译。

#### 3. 生成并行测试副本

如果只使用一个测试进程，可以跳过此步。如果需要并行测试，当前推荐额外生成
7 份 SML 副本：

```sh
cd pass-test
python _0_extend_testcase.py
cd ..
bazel build //... -c opt
```

`_0_extend_testcase.py` 根据 `extend_number` 将 `sml` 复制为 `sml1` 至
`sml7`，并同步修改 import 和 Bazel target。脚本会覆盖同名的既有副本，生成后
也必须重新编译。默认的 `sml` 加 7 份副本对应 PassTester 中的
`thread_num = 8`。

准备完成后，至少应确认以下内容存在：

```text
bazel-bin/sml/...
pass-test/file-to-be-modified/sml/...
pass-test/pass_inf/pass_options_HLO_delete.txt
```

并行运行时还应存在 `bazel-bin/sml1` 至 `bazel-bin/sml7`。

### 阶段二：PassTester

PassTester 由 `_1_auto_test_pass.py` 和 `_2_extract_inf.py` 两步组成。第一步
运行实验并保存原始日志，第二步把日志整理成可分析的 JSON。

#### 1. 配置并运行 `_1_auto_test_pass.py`

首先检查脚本主函数中的配置：

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

各选项含义如下：

| 选项 | 作用 |
| --- | --- |
| `test_mod` | 选择测试集合，通常为 `mod.sml`。 |
| `repeat_time` | 每个配置的重复次数；大于 1 时应设置 `thread_num = 1`。 |
| `output_folder` | 本轮实验的独立输出目录，避免覆盖已有结果。 |
| `thread_num` | 并行 worker 数，不能超过已准备的 SML 副本数。 |
| `protocal` | MPC 协议，可按测试需求设为 `SEMI2K`、`ABY3` 或 `CHEETAH`。变量名沿用脚本中的拼写。 |
| `passoptions_path` | 本轮测试使用的 Pass 配置列表。 |
| `hlo_log` | 是否额外打印 XLA HLO；普通 Pass 测试可关闭，PatternExtractor 捕获案例时需开启。 |
| `refer_to_complete_test` | 是否参考已有完整测试结果筛选任务。 |

如只测试部分文件，可同时缩小脚本中的 `test_file_list`。随后运行：

```sh
cd pass-test
python _1_auto_test_pass.py
```

脚本会完成以下工作：

1. 从 `file-to-be-modified/sml` 读取带占位符的测试模板；
2. 替换 `{partynum}`、`{protocalchosen}`、`{pphlo_dict}` 和
   `{skip_control}`，并插入当前 Pass 编译选项；
3. 将实例化后的文件写入当前 worker 对应的 `sml`、`sml1` 等目录；
4. 先运行 baseline，再逐个运行 Pass 配置；
5. 记录 PPHLO，并跳过与 baseline IR 相同的冗余结果。

典型输出结构为：

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

其中 `test-log.txt` 是整体调度日志，`test_*.txt` 是各实验的原始输出，
`IR_record` 记录哪些 Pass 确实改变了函数的 PPHLO。

#### 2. 提取实验信息

检查 `_2_extract_inf.py` 主函数中的配置：

```python
party_nums = 2
log_directory = "test-log/SEMI2K-all"
```

`SEMI2K` 和 `CHEETAH` 通常使用 `party_nums = 2`，`ABY3` 使用
`party_nums = 3`。配置完成后运行：

```sh
cd pass-test
python _2_extract_inf.py
```

脚本按参与方拆分原始日志，提取每个函数的运行时间、发送字节数、发送次数、
PPHLO 等信息，并在每个测试目录生成合并结果：

```text
test-log/SEMI2K-all/<test_name>/extract_result.json
```

该文件以函数和 Pass 配置为索引，是后续筛选性能差异的主要输入。

如果终端出现：

```text
Function <function_name> in <pass_directory> does not have log file
```

表示该配置虽然改变了 IR，但没有产出完整的函数 profile。常见原因是测试失败、
编译失败、进程崩溃或日志未完整写入；这类配置不会被合并到最终 JSON。少量此类
提示在批量禁用 Pass 的实验中是可能出现的，但不能仅凭提示认定测试正常。

PassTester 阶段的最终有效产物是：正确性通过的 baseline/Pass 原始日志，以及
对应测试目录中的合并 `extract_result.json`。

### 阶段三：PatternExtractor

PatternExtractor 使用 `_3_code_reduce_case_gen.py` 生成约简案例，再由
`_4_code_reducer.py` 反复删除 HLO 指令，寻找仍能保留目标通信差异的最小 Pattern。

#### 1. 选择待约简案例

从 PassTester 的 `extract_result.json` 中选择满足以下条件的函数和 Pass：

- baseline 与目标 Pass 的测试均正确完成；
- 目标 Pass 确实改变了 PPHLO；
- `send_bytes` 或 `send_actions` 存在稳定、可复现的差异；
- 明确记录原始配置 `ori`、变异配置 `mut`、协议和目标指标。

不要把仅有崩溃、编译失败或错误计算结果的配置作为性能 Pattern。

#### 2. 获取用于约简的 XLA HLO

约简器需要的是以 `HloModule` 开头的 **XLA HLO 文本**，不是以
`module @jit_...` 开头的 PPHLO/MLIR。默认 Pass 测试只记录 PPHLO；如果现有
`test-log/SEMI2K-all` 中没有 `HLO_IR|`、`Start printing HLO IR` 或
`HloModule`，就不能直接从该批日志获取约简输入。

建议针对候选测试单独捕获 HLO，避免覆盖正式实验结果。在
`_1_auto_test_pass.py` 中临时设置：

```python
hlo_log = True
output_folder = "test-log/HLO-capture"
test_file_list = ["sml/cluster/tests/kmeans_test.py"]
```

如果只需要 baseline HLO，可让 `passoptions_path` 指向一个空文件。XLA HLO
在 SPU Pass 流水线之前产生，通常同一份 baseline HLO 就可用于比较后续的
`ori` 与 `mut` 配置。运行：

```sh
cd pass-test
python _1_auto_test_pass.py
```

原始日志中的目标内容形如：

```text
HLO_IR|Start printing HLO IR for test_kmeans_random
HLO_IR|HloModule jit_test_kmeans_random, ...
HLO_IR|...
HLO_IR|End of printing HLO IR for test_kmeans_random
```

复制 `HLO_IR|HloModule ...` 到该函数最后一个 `HLO_IR|}`，不要包含 Start 和
End 标记。也可以从含有 HLO 的 baseline 日志自动提取：

```sh
cd pass-test
python - <<'PY'
from _2_extract_inf import split_log_party, extract_log

path = "test-log/HLO-capture/kmeans_test/baseline"
split_log_party(path, party_nums=2, pphlo_log=True, hlo_log=True)
extract_log(path, pphlo_log=True, hlo_log=True)
PY
```

提取结果位于：

```text
test-log/HLO-capture/kmeans_test/baseline/extracted_inf/hlo.txt
```

相应 `extract_result.json` 中也会包含 `hlo` 字段。提取后的内容已经去掉
`HLO_IR|` 前缀。

#### 3. 生成约简案例

在 `_3_code_reduce_case_gen.py` 中放入目标函数的完整 HLO，并配置：

```python
case_name = "your-case-name"
pass_option_mut = {
    "ori": [],
    "mut": ["disable_or_enable_some_pass"],
}
protocalchosen = "SEMI2K"  # 也可以是 ABY3 或 CHEETAH
matrix = "send_bytes"      # 也可以是 send_actions
```

如果粘贴的是带前缀的原始日志，可沿用脚本中的 `test_case_full` 格式：

```python
test_case_full = """HLO_IR|HloModule ...
HLO_IR|...
HLO_IR|}"""
```

脚本会去掉每行的 `HLO_IR|`。如果使用 `hlo.txt` 中已经清理过的内容，则应直接
构造 `input_test_case`，不要再次按 `HLO_IR|` 分割。随后运行：

```sh
cd pass-test
python _3_code_reduce_case_gen.py
```

生成结果为：

```text
pass-test/reduce_case/<case_name>.json
```

#### 4. 运行 HLO 约简

运行前检查 `_4_code_reducer.py` 文件末尾的配置，确保案例名与上一步一致：

```python
case_name_list = ["your-case-name"]
ddmethod_list = ["ddmin"]
```

`ddmethod_list` 只支持 `ddmin`、`onlycomplement`、`CDD` 和 `ProfDD`，
列表中可以填写其中一种或多种方法。

同样检查文件开头的提取模块导入。当前只保留 `_2_extract_inf.py` 时，应使用：

```python
from _2_extract_inf import split_log_party, extract_log
```

同时确认 `bazel-bin/spu/tests/hlo_debug` 已经编译。然后执行：

```sh
cd pass-test
python _4_code_reducer.py
```

约简器会将候选 HLO 写入 `spu/tests/hlo_debug.py` 对应的测试模板，分别运行
`ori` 和 `mut` 配置，并依据 `matrix` 判断通信差异是否仍然存在。主要输出位于：

```text
pass-test/reduce_log/<case_name>/<case_name>-<ddmethod>/
```

其中包含约简过程、触发记录和最终候选 Pattern。`_4_code_reducer.py` 会删除同名的
旧约简日志目录，因此重新运行前应先备份需要保留的结果，并再次核对
`case_name_list`。

#### 5. 验证最终 Pattern

约简结束后，应使用最终 HLO 至少重新运行一次 `ori` 和 `mut`，确认：

- 两种配置都能成功编译和执行；
- 输出结果正确；
- 目标 `send_bytes` 或 `send_actions` 差异仍然稳定存在；
- 最终 HLO 足够小，并且删除任一关键部分会使目标差异消失。

通过以上检查后，`reduce_case/<case_name>.json`、`reduce_log/<case_name>/`
以及对应的 PassTester 原始日志共同构成一个可复现的 Pattern 案例。
