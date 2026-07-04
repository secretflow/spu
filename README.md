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
