This example demonstrates how to use SPU to train an LSTM model privately.

This example comes from Haiku official github repo:

https://github.com/deepmind/dm-haiku/blob/main/examples/haiku_lstms.ipynb

1. Install dependencies
    ```
    pip install -r requirements.txt
    ```

2. Set runtime configuration

    This example requires a higher precision setting than the default.

    Set `"fxp_fraction_bits": 24` in `SPU runtime_config`.

    The default configuration file locates at [examples/python/conf/3pc.json](../../conf/3pc.json).

3. Launch SPU backend runtime
    ```
    bazel run -c opt //examples/python/utils:nodectl -- up
    ```

4. Run `haiku_lstm` example
    ```
    bazel run -c opt //examples/python/ml/haiku_lstm -- --output_dir `pwd`
    ```

5. Check results
    When training is finished, you can check the generated images in the specified `output_dir` and compare the results to CPU versions.
