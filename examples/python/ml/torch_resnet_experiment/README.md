# Torch Example

This example demonstrates how to use SPU to make private inferences on PyTorch models.

**Note**: Currently, SPU's support of PyTorch is **experimental**.

1. Launch SPU backend runtime

    ```sh
    uv run examples/python/utils/nodectl.py up
    ```

2. Run `torch_resnet_experiment` example

    ```sh
    uv run examples/python/ml/torch_resnet_experiment/torch_resnet_experiment.py
    ```
