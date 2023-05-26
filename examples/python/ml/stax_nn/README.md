# Stax NN Example

This example demonstrates how to use SPU to train neural network models privately for image classification.

These models are widely used for evaluation benchmarks in MPC-enabled literature such as [deep-mpc](https://arxiv.org/abs/2107.00501) and [SeureNN](https://eprint.iacr.org/2018/442.pdf).

1. Launch SPU backend runtime

    ```sh
    bazel run -c opt //examples/python/utils:nodectl -- up
    ```

2. Run `stax_nn` example

    ```sh
    bazel run -c opt //examples/python/ml/stax_nn
    ```
