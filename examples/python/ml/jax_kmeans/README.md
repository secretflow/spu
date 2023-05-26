# Jax Kmeans Example

This example demonstrates how to use SPU to train K-Means clustering privately.

1. Launch SPU backend runtime

    ```sh
    bazel run -c opt //examples/python/utils:nodectl -- up
    ```

2. Run `jax_kmeans` example

    ```sh
    bazel run -c opt //examples/python/ml/jax_kmeans
    ```
