# Flax MLP Example

This example demonstrates how to use SPU to train an MLP model privately.

1. Launch SPU backend runtime

    ```sh
    bazel run -c opt //examples/python/utils:nodectl -- up
    ```

2. Run `flax_mlp` example

    ```sh
    bazel run -c opt //examples/python/ml/flax_mlp
    ```
