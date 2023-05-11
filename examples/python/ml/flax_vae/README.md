This example demonstrates how to use SPU to train the [VAE](http://arxiv.org/abs/1312.6114) model privately.

This example comes from Flax official github repo:

https://github.com/google/flax/tree/main/examples/vae

1. Set runtime configuration
This example requires a higher precision setting than the default.
Set `"fxp_fraction_bits": 24` in `SPU runtime_config`, 
the default configuration file locates at [examples/python/conf/3pc.json](../../conf/3pc.json).

2. Launch SPU backend runtime
```
bazel run -c opt //examples/python/utils:nodectl -- up
```

3. Run `flax_vae` example
```
bazel run -c opt //examples/python/ml/flax_vae:flax_vae -- --output_dir `pwd` --num_epochs 5
```

4. Check results
When training is finished, you can check the generated images in the specified `output_dir` and compare the results of SPU and CPU versions.