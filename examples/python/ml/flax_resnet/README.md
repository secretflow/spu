# Flax ResNet Example

This example demonstrates how to use SPU to train and evaluate the [ResNet](https://arxiv.org/abs/1512.03385) model privately.

This training example comes from Flax official github repo:

<https://github.com/google/flax/tree/main/examples/imagenet>

and the inference example comes from pre-trained microsoft resnet-50 model on huggingface:

<https://huggingface.co/microsoft/resnet-50>

1. Launch SPU backend runtime

    ```sh
    uv run examples/python/utils/nodectl.py --config `pwd`/examples/python/ml/flax_resnet/3pc.json up
    ```

2. Run `flax_resnet_training` example

    ```sh
    uv run examples/python/ml/flax_resnet/flax_resnet_training.py --config `pwd`/examples/python/ml/flax_resnet/3pc.json --num_epochs 5
    ```

3. Run `flax_resnet_inference` example

    ```sh
    uv run examples/python/ml/flax_resnet/flax_resnet_inference.py --config `pwd`/examples/python/ml/flax_resnet/3pc.json
    ```
