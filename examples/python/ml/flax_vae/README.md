# Flax VAE Example

This example demonstrates how to use SPU to train the [VAE](http://arxiv.org/abs/1312.6114) model privately.

This example comes from Flax official github repo:

<https://github.com/google/flax/tree/main/examples/vae>

1. Launch SPU backend runtime

    ```sh
    bazel run -c opt //examples/python/utils:nodectl -- up
    ```

2. Run `flax_vae` example

    ```sh
    bazel run -c opt //examples/python/ml/flax_vae -- --output_dir `pwd` --num_epochs 5
    ```

3. Check results
    When training is finished, you can check the generated images in the specified `output_dir` and compare the results of SPU and CPU versions.
