# Flax Whisper Example

This example demonstrates how to use SPU to run private inference on a pre-trained
[Whisper](https://huggingface.co/docs/transformers/model_doc/whisper#transformers.FlaxWhisperForConditionalGeneration) model.

1. Enable While with secret value

    Edit libspu/kernel/hlo/control_flow.cc, change `ENABLE_DEBUG_ONLY_REVEAL_SECRET_CONDITION` to `true`.

2. Launch SPU backend runtime

    ```sh
    uv run examples/python/utils/nodectl.py --config `pwd`/examples/python/ml/flax_whisper/3pc.json up
    ```

3. Run `flax_whisper` example

    ```sh
    uv run examples/python/ml/flax_whisper/flax_whisper.py --config `pwd`/examples/python/ml/flax_whisper/3pc.json
    ```
