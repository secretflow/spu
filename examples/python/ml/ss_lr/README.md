# SS-LR Example

This example demonstrates how to use SPU to train linear/logistic regression models privately.

1. Launch SPU backend runtime

    ```sh
    bazel run -c opt //examples/python/utils:nodectl -- up
    ```

2. Run `ss_lr` example

    ```sh
    bazel run -c opt //examples/python/ml/ss_lr
    ```
