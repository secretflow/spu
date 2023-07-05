# SPU: Secure Processing Unit

[![CircleCI](https://dl.circleci.com/status-badge/img/gh/secretflow/spu/tree/main.svg?style=svg)](https://dl.circleci.com/status-badge/redirect/gh/secretflow/spu/tree/main)

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

Please follow [Installation Guidelines](INSTALLATION.md) to install SPU.

## Citing SPU

If you think SPU helpful for your research or development, please consider citing our [paper](https://www.usenix.org/conference/atc23/presentation/ma):

```text
@inproceedings {288747,
    author = {Junming Ma and Yancheng Zheng and Jun Feng and Derun Zhao and Haoqi Wu and Wenjing Fang and Jin Tan and Chaofan Yu and Benyu Zhang and Lei Wang},
    title = {{SecretFlow-SPU}: A Performant and {User-Friendly} Framework for {Privacy-Preserving} Machine Learning},
    booktitle = {2023 USENIX Annual Technical Conference (USENIX ATC 23)},
    year = {2023},
    address = {Boston, MA},
    url = {https://www.usenix.org/conference/atc23/presentation/ma},
    publisher = {USENIX Association},
    month = jul,
}
```

## Acknowledgement

We thank the significant contributions made by [Alibaba Gemini Lab](https://alibaba-gemini-lab.github.io).
