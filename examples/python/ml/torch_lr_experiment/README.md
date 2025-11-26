# Torch Example

This example demonstrates how to use SPU to make private inferences on PyTorch models.

**Note**: Currently, SPU's support of PyTorch is **experimental**.

1. Launch SPU backend runtime

    ```sh
    python examples/python/utils/nodectl.py up
    ```

2. Run `torch_lr_experiment` example

    ```sh
    python examples/python/ml/torch_lr_experiment/torch_lr_experiment.py
    ```
