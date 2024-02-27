# Torch Example

This example demonstrates how to use SPU to make private inferences on PyTorch models.

**Note**: Currently, SPU's support of PyTorch is **experimental**.

1. Install a third-party dependency [PyTorch/XLA](https://github.com/pytorch/xla).

    ```sh
    pip install torch==2.2.0 torch_xla==2.2.0
    ```

2. Launch SPU backend runtime

    ```sh
    bazel run -c opt //examples/python/utils:nodectl -- up
    ```

3. Run `torch_lr_experiment` example

    ```sh
    bazel run -c opt //examples/python/ml/torch_lr_experiment
    ```
