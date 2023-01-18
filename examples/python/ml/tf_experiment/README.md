This example demonstrates how to use SPU to train a logistic regression privately with TensorFlow. 

Currently, SPU's support of TensorFlow is **experimental**.

1. Launch SPU backend runtime
    ```
    bazel run -c opt //examples/python/utils:nodectl -- up
    ```

2. Run `tf_experiment` example
    ```
    bazel run -c opt //examples/python/ml/tf_experiment
    ```
