This example demonstrates how to use SPU to train a logistic regression model privately.

1. Launch SPU backend runtime
    ```
    bazel run -c opt //examples/python/utils:nodectl -- up
    ```

2. Run `jax_lr` example
    ```
    bazel run -c opt //examples/python/ml/jax_lr
    ```
