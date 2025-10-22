# Flax MLP Example

This example demonstrates how to use SPU to train an MLP model privately.

1. Launch SPU backend runtime

    ```sh
    uv run examples/python/utils/nodectl.py up
    ```

2. Run `flax_mlp` example

    ```sh
    uv run examples/python/ml/flax_mlp/flax_mlp.py
    ```
