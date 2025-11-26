# Jraph GNN Example

This example demonstrates how to use SPU to train a Graph Convolutional Network model privately.

This example comes from Jraph official github repo:

<https://github.com/deepmind/jraph/blob/master/jraph/examples/zacharys_karate_club.py>

1. Set runtime configuration

    This example requires a higher precision setting than the default.

    Set `"fxp_fraction_bits": 24` in `SPU runtime_config`.

    The default configuration file locates at [examples/python/conf/3pc.json](../../conf/3pc.json).

2. Launch SPU backend runtime

    ```sh
    python examples/python/utils/nodectl.py up
    ```

3. Run `jraph_gnn` example

    ```sh
    python examples/python/ml/jraph_gnn/jraph_gnn.py
    ```
