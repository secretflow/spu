# Torch Example

This example demonstrates how to use SPU to make inferences on a linear regression model privately with PyTorch.

The model is trained with plaintext publicly. Currently, SPU's support of PyTorch is **experimental** and we only tested on Linux.

1. Install a third-party dependency [Torch-MLIR](https://github.com/llvm/torch-mlir).

    ```sh
    pip install https://github.com/llvm/torch-mlir/releases/download/snapshot-20220830.581/torch-1.13.0.dev20220830+cpu-cp38-cp38-linux_x86_64.whl
    pip install https://github.com/llvm/torch-mlir/releases/download/snapshot-20220830.581/torch_mlir-20220830.581-cp38-cp38-linux_x86_64.whl
    ```

2. Launch SPU backend runtime

    ```sh
    bazel run -c opt //examples/python/utils:nodectl -- up
    ```

3. Run `torch_experiment` example

    ```sh
    bazel run -c opt //examples/python/ml/torch_experiment
    ```
