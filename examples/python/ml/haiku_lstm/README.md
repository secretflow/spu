# haiku LSTM Example

This example demonstrates how to use SPU to train an LSTM model privately.

This example comes from Haiku official github repo:

<https://github.com/deepmind/dm-haiku/blob/main/examples/haiku_lstms.ipynb>

1. Launch SPU backend runtime

    ```sh
    uv run examples/python/utils/nodectl.py up
    ```

2. Run `haiku_lstm` example

    ```sh
    uv run examples/python/ml/haiku_lstm/haiku_lstm.py --output_dir `pwd`
    ```

3. Check results
    When training is finished, you can check the generated images in the specified `output_dir` and compare the results to CPU versions.
