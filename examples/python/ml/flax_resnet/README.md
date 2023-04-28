This example demonstrates how to use SPU to train the [ResNet](https://arxiv.org/abs/1512.03385) model privately.

This example comes from Flax official github repo:

https://github.com/google/flax/tree/main/examples/imagenet

1. Launch SPU backend runtime
    ```
    bazel run -c opt //examples/python/utils:nodectl -- --config `pwd`/examples/python/ml/flax_resnet/3pc.json up
    ```

2. Run `flax_resnet` example
    ```
    bazel run -c opt //examples/python/ml/flax_resnet -- --config `pwd`/examples/python/ml/flax_resnet/3pc.json --num_epochs 5
    ```
