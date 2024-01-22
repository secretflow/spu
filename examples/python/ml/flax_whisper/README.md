# Flax Whisper Example

This example demonstrates how to use SPU to run private inference on a pre-trained
[Whisper](https://huggingface.co/docs/transformers/model_doc/whisper#transformers.FlaxWhisperForConditionalGeneration) model.

1. Install huggingface transformers library

    ```sh
    pip install 'transformers[flax]' soundfile librosa
    ```

2. Enable While with secret value

    Edit libspu/kernel/hlo/control_flow.cc, change `ENABLE_DEBUG_ONLY_REVEAL_SECRET_CONDITION` to `true`.

3. Launch SPU backend runtime

    ```sh
    bazel run -c opt //examples/python/utils:nodectl -- --config `pwd`/examples/python/ml/flax_whisper/3pc.json up
    ```

4. Run `flax_whisper` example

    ```sh
    bazel run -c opt //examples/python/ml/flax_whisper -- --config `pwd`/examples/python/ml/flax_whisper/3pc.json
    ```
