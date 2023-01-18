This example demonstrates how to use SPU to train K-Means clustering privately.

1. Launch SPU backend runtime
    ```
    bazel run -c opt //examples/python/utils:nodectl -- up
    ```

2. Run `jax_kmeans` example
    ```
    bazel run -c opt //examples/python/ml/jax_kmeans
    ```
