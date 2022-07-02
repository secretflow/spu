# Repository Layout

![SPU Architect](docs/imgs/spu_arch.png)

This is a high level overview of how the repository is laid out. Some major folders are listed below:

* [bazel/](bazel/): Configuration for SPU's use of [Bazel](https://bazel.build/).
* [docs/](docs/): Documents of SPU.
* [examples/](examples/): Contains both cpp and python examples of SPU.
* [spu/](spu/): Core implementations of SPU.
    - [binding/](spu/binding/): Python binding of the SPU device.
    - [compiler/](spu/compiler/): SPU's compiler stack. It accepts the standard XLA IR along with inputs mpc metadata and lowering the XLA IR to a mpc specific IR.
    - [core/](spu/core/): Basic data structures used in SPU.
    - [crypto/](spu/crypto/): Crytographic primitives used in mpc protocols, say oblivious transfer.
    - [device/](spu/device/): SPU device `Runtime`.It consists of IO(infeed, outfeed), symbol tables(storage) and the IR executor.
    - [dialect/](spu/dialect/): Internal MPC specific IR used by SPU.
    - [hal/](spu/hal/): Hardware adapt layer implements crypto independent core logics, say the fixed point related abstractions and some non-linear APIs. It could be viewed as a builtin library in addition to the SPU VM.
    - [mpc/](spu/mpc/): Various mpc protocols. This folder defines the [standard interface](spu/mpc/interfaces.h) different mpc protocols need to conform.
        * [aby3/](spu/mpc/aby3/): The semi-honest variant of ABY3 protocol. Currently only `Arithmetic` and `Boolean` are implemented.
        * [cheetah/](spu/mpc/cheetah/): An excellent semi-honest 2PC protocol implemented by [Alibaba-Gemini-Lab](https://alibaba-gemini-lab.github.io/).
        * [semi2k/](spu/mpc/semi2k/): The semi-honest variant of SPDZ protocol. It could be shipped with different correlated random generators.
        * [ref2k/](spu/mpc/ref2k/): A plaintext protocol. It is aimed to serve as a reference implementation of how a new protocol could be added in SPU.
        * [util/](spu/mpc/util/): Common utilities for different mpc protocols.
