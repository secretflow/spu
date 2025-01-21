# Flax GPT2 Example

This example demonstrates how to use SPU to run private inference on a pre-trained
[GPT2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) model.

1. Install huggingface transformers library

    ```sh
    pip install 'transformers[flax]'
    ```

2. Launch SPU backend runtime

    ```sh
    bazel run -c opt //examples/python/utils:nodectl -- --config `pwd`/examples/python/ml/flax_gpt2/3pc.json up
    ```

3. Run `flax_gpt2` example

    ```sh
    bazel run -c opt //examples/python/ml/flax_gpt2 -- --config `pwd`/examples/python/ml/flax_gpt2/3pc.json
    ```
