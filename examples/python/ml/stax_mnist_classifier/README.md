# Stax MNIST Classifier Example

This example demonstrates how to use SPU to train a neural network model privately for MNIST classification with
[Stax](https://jax.readthedocs.io/en/latest/jax.example_libraries.stax.html) library.

1. Launch SPU backend runtime

    ```sh
    bazel run -c opt //examples/python/utils:nodectl -- up
    ```

2. Run `stax_mnist_classifier` example

    ```sh
    bazel run -c opt //examples/python/ml/stax_mnist_classifier
    ```
