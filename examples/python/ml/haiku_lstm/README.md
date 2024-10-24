# haiku LSTM Example

This example demonstrates how to use SPU to train an LSTM model privately.

This example comes from Haiku official github repo:

<https://github.com/deepmind/dm-haiku/blob/main/examples/haiku_lstms.ipynb>

1. Install dependencies

    ```sh
    pip install -r requirements.txt
    ```

2. Launch SPU backend runtime

    ```sh
    bazel run -c opt //examples/python/utils:nodectl -- up
    ```

3. Run `haiku_lstm` example

    ```sh
    bazel run -c opt //examples/python/ml/haiku_lstm -- --output_dir `pwd`
    ```

4. Check results
    When training is finished, you can check the generated images in the specified `output_dir` and compare the results to CPU versions.
