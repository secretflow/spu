# Jax SVM Example

This example demonstrates how to use SPU to train an SVM model privately.

1. Launch SPU backend runtime

    ```sh
    uv run examples/python/utils/nodectl.py up
    ```

2. Run `jax_svm` example

    ```sh
    uv run examples/python/ml/jax_svm/jax_svm.py
    ```
