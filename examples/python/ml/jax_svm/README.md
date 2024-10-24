# Jax SVM Example

This example demonstrates how to use SPU to train an SVM model privately.

1. Launch SPU backend runtime

    ```sh
    bazel run -c opt //examples/python/utils:nodectl -- up
    ```

2. Run `jax_svm` example

    ```sh
    bazel run -c opt //examples/python/ml/jax_svm
    ```
