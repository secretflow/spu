# TensorFlow Example

This example demonstrates how to use SPU to train a logistic regression privately with TensorFlow.

Currently, SPU's support of TensorFlow is **experimental**.

1. Launch SPU backend runtime

    ```sh
    bazel run -c opt //examples/python/utils:nodectl -- up
    ```

2. Run `tf_experiment` example

    ```sh
    bazel run -c opt //examples/python/ml/tf_experiment
    ```
