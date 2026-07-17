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
### 准备阶段
### step 0-1: 获取xla pass (已经有跑完的结果，可以直接跳过)

git clone https://github.com/openxla/xla.git
cd xla
git checkout 64bdcc53a1b24abf19b1fe598e6f9b0fe6454470
cd ../pass-test
python _0_get_xla_pass_inf.py

### step 0-2: 将获取的xla pass和在spu中已使用的xla pass转化为编译选项 (已经有跑完的结果，可以直接跳过)
cd pass-test
python _0_pass_adder_modifier.py
# 生成完成后，要重新编译
cd ..
bazel build //... -c opt

### step 0-3: 复制多份sml，用于并行测试 (如果不需要并行测试，可以直接跳过；推荐再生成七个)
cd pass-test
python _0_extend_testcase.py
# 生成完成后，要重新编译
cd ..
bazel build //... -c opt

### PassTester

### PatternExtractor

#### 获取用于 HLO 约简的 HLO IR
`pass-test/code_reduce_case_gen.py` 和 `pass-test/code_reducer.py` 使用的是
XLA HLO 文本，格式类似：

```text
HloModule jit_test_kmeans_random, ...

region_0.1 {
  Arg_0.2 = ...
  ROOT add.3 = ...
}

ENTRY main.4 {
  ...
}
```

它不是 `module @jit_...` 开头的 PPHLO/MLIR。默认的 Pass 测试只打印
PPHLO，因此现有 `pass-test/test-log/SEMI2K-all` 中如果没有
`HLO_IR|`、`Start printing HLO IR` 或 `HloModule`，就不能直接从该批日志
取得约简器所需的 HLO。

#### 1. 开启 HLO 日志

在 `pass-test/_1_auto_test_pass.py` 中将：

```python
hlo_log = False
```

改为：

```python
hlo_log = True
```

测试模板中的 `{skip_control}` 随后会被实例化为 `hlo_log=True`。SPU 前端会在
原始测试日志中输出：

```text
HLO_IR|Start printing HLO IR for test_kmeans_random
HLO_IR|HloModule jit_test_kmeans_random, ...
HLO_IR|...
HLO_IR|End of printing HLO IR for test_kmeans_random
```

建议只运行需要约简的测试文件，并使用独立输出目录，避免覆盖已有的
`test-log/SEMI2K-all`。例如在 `_1_auto_test_pass.py` 中临时设置：

```python
output_folder = "test-log/HLO-capture"
test_file_list = ["sml/cluster/tests/kmeans_test.py"]
```

如果只需要 HLO，可以让 `passoptions_path` 指向一个空文件，使脚本只运行
baseline。HLO 在 SPU Pass 流水线执行前由 JAX/XLA 前端生成，因此通常使用
baseline HLO 即可；之后在 `code_reduce_case_gen.py` 中分别设置 `ori` 和
`mut` Pass 配置。

脚本应从 `pass-test` 目录运行：

```sh
cd pass-test
python _1_auto_test_pass.py
```

#### 2. 从原始日志复制 HLO

在目标 baseline 日志中找到：

```text
HLO_IR|Start printing HLO IR for <function_name>
```

复制它后面的 `HLO_IR|HloModule ...` 至该函数 HLO 的最后一个
`HLO_IR|}`。不要复制 `Start printing` 和 `End of printing` 两行。

将复制结果放入 `pass-test/code_reduce_case_gen.py`：

```python
test_case_full = """HLO_IR|HloModule ...
HLO_IR|...
HLO_IR|}"""
```

脚本中的下列代码会删除每行的 `HLO_IR|` 前缀：

```python
input_test_case = [
    line.split("HLO_IR|")[1]
    for line in test_case_full.split("\n")
]
```

#### 3. 使用 `_2_extract_inf.py` 提取 HLO

对于已经包含 HLO 的 baseline 原始日志，也可以在 `pass-test` 目录执行：

```sh
python - <<'PY'
from _2_extract_inf import split_log_party, extract_log

path = "test-log/HLO-capture/kmeans_test/baseline"

split_log_party(
    path,
    party_nums=2,
    pphlo_log=True,
    hlo_log=True,
)
extract_log(
    path,
    pphlo_log=True,
    hlo_log=True,
)
PY
```

提取后会生成：

```text
test-log/HLO-capture/kmeans_test/baseline/extracted_inf/hlo.txt
```

相应函数在 `extract_result.json` 中也会包含 `hlo` 字段。`hlo.txt` 和 JSON
中的内容已经去掉 `HLO_IR|` 前缀；如果从这里复制，可以直接构造
`input_test_case`，或者在粘贴到现有 `test_case_full` 格式前重新加上前缀。

#### 4. 配置并生成约简案例

在 `pass-test/code_reduce_case_gen.py` 中设置：

```python
case_name = "your-case-name"
pass_option_mut = {
    "ori": [],
    "mut": ["disable_or_enable_some_pass"],
}
protocalchosen = "SEMI2K"  # 也可以是 ABY3 或 CHEETAH
matrix = "send_bytes"      # 也可以是 send_actions
```

然后运行：

```sh
cd pass-test
python code_reduce_case_gen.py
python code_reducer.py
```

前者生成 `reduce_case/<case_name>.json`，后者反复缩减 HLO，并将过程和结果
写入 `reduce_log/<case_name>/`。运行 `code_reducer.py` 前应检查文件末尾的
`case_name_list` 和 `ddmethod_list`；它还会删除同名的旧约简日志目录。

### Todo

