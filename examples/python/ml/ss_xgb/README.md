# SS-XGB Example

This example demonstrates how to use SPU to train XGB models privately.

1. Launch SPU backend runtime

    ```sh
    bazel run -c opt //examples/python/utils:nodectl -- up
    ```

2. Run `ss_xgb` example

    ```sh
    bazel run -c opt //examples/python/ml/ss_xgb
    ```
