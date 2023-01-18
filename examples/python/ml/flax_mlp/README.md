This example demonstrates how to use SPU to train an MLP model privately.

1. Launch SPU backend runtime
    ```
    bazel run -c opt //examples/python/utils:nodectl -- up
    ```

2. Run `flax_mlp` example
    ```
    bazel run -c opt //examples/python/ml/flax_mlp
    ```
