# Jax LR Example

This example demonstrates how to use SPU to train a logistic regression model privately.

1. Launch SPU backend runtime

    ```sh
    bazel run -c opt //examples/python/utils:nodectl -- up
    ```

2. Run `jax_lr` example

    ```sh
    bazel run -c opt //examples/python/ml/jax_lr
    ```
