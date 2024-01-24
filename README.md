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

This documentation also contains instructions for [build and testing](CONTRIBUTING.md#build).

## Installation Guidelines

### Supported platforms

|            | Linux x86_64 | Linux aarch64 | macOS x64      | macOS Apple Silicon | Windows x64    | Windows WSL2    x64 |
|------------|--------------|---------------|----------------|---------------------|----------------|---------------------|
| CPU        | yes          | yes           | yes<sup>1</sup>| yes                 | no             | yes                 |
| NVIDIA GPU | experimental | no            | no             | n/a                 | no             | no                  |

1. macOS x64 support will be deprecated soon

### Instructions

Please follow [Installation Guidelines](INSTALLATION.md) to install SPU.

### Hardware Requirements

| General Features | FourQ based PSI | GPU |
| ---------------- | --------------- | --- |
| AVX/ARMv8        | AVX2/ARMv8      | CUDA 11.8+ |

## Citing SPU

If you think SPU helpful for your research or development, please consider citing our [paper](https://www.usenix.org/conference/atc23/presentation/ma):

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

## Acknowledgement

We thank the significant contributions made by [Alibaba Gemini Lab](https://alibaba-gemini-lab.github.io).
