# TensorFlow Example

This example demonstrates how to use SPU to train a logistic regression privately with TensorFlow.

Currently, SPU's support of TensorFlow is **experimental**.

1. Launch SPU backend runtime

    ```sh
    python examples/python/utils/nodectl.py up
    ```

2. Run `tf_experiment` example

    ```sh
    python examples/python/ml/tf_experiment/tf_experiment.py
    ```
