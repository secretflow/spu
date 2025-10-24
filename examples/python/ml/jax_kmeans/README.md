# Jax Kmeans Example

This example demonstrates how to use SPU to train K-Means clustering privately.

1. Launch SPU backend runtime

    ```sh
    uv run examples/python/utils/nodectl.py up
    ```

2. Run `jax_kmeans` example

    ```sh
    uv run examples/python/ml/jax_kmeans/jax_kmeans.py
    ```
