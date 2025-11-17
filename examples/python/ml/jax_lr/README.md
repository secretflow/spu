# Jax LR Example

This example demonstrates how to use SPU to train a logistic regression model privately.

1. Launch SPU backend runtime

    ```sh
    uv run examples/python/utils/nodectl.py -c examples/python/conf/2pc_semi2k.json up
    ```

2. Run `jax_lr` example

    ```sh
    uv run examples/python/ml/jax_lr/jax_lr.py
    ```
