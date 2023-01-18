This example demonstrates how to use SPU to train linear/logistic regression models privately.

1. Launch SPU backend runtime
    ```
    bazel run -c opt //examples/python/utils:nodectl -- up
    ```

2. Run `ss_lr` example
    ```
    bazel run -c opt //examples/python/ml/ss_lr
    ```
