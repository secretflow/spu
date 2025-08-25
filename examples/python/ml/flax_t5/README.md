# Flax Whisper Example

This example demonstrates how to use SPU to run private inference on a pre-trained
[T5](https://huggingface.co/docs/transformers/model_doc/t5#transformers.FlaxT5ForConditionalGeneration) model.

1. Enable While with secret value

    Edit libspu/kernel/hlo/control_flow.cc, change `ENABLE_DEBUG_ONLY_REVEAL_SECRET_CONDITION` to `true`.

2. Launch SPU backend runtime

    ```sh
    python examples/python/utils/nodectl.py --config `pwd`/examples/python/ml/flax_t5/3pc.json up
    ```

3. Run `flax_t5` example

    ```sh
    python examples/python/ml/flax_t5/flax_t5.py --config `pwd`/examples/python/ml/flax_t5/3pc.json
    ```
