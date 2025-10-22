# Stax MNIST Classifier Example

This example demonstrates how to use SPU to train a neural network model privately for MNIST classification with
[Stax](https://jax.readthedocs.io/en/latest/jax.example_libraries.stax.html) library.

1. Launch SPU backend runtime

    ```sh
    uv run examples/python/utils/nodectl.py up
    ```

2. Run `stax_mnist_classifier` example

    ```sh
    uv run examples/python/ml/stax_mnist_classifier/stax_mnist_classifier.py
    ```
