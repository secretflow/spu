This example demonstrates how to use SPU to train XGB models privately.

1. Launch SPU backend runtime
    ```
    bazel run -c opt //examples/python/utils:nodectl -- up
    ```

2. Run `ss_xgb` example
    ```
    bazel run -c opt //examples/python/ml/ss_xgb
    ```
