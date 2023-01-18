This example demonstrates how to use SPU to train an SVM model privately.

1. Launch SPU backend runtime
    ```
    bazel run -c opt //examples/python/utils:nodectl -- up
    ```

2. Run `jax_svm` example
    ```
    bazel run -c opt //examples/python/ml/jax_svm
    ```
