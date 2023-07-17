# Flax Llama-7B Example with Puma

This example demonstrates how to use SPU to run secure inference on a pre-trained
[Llama-7B](https://research.facebook.com/publications/llama-open-and-efficient-foundation-language-models/) model.

1. Install huggingface transformers library

    ```sh
    pip install 'transformers[flax]'
    ```

2. Download EasyML library to support Flax-Llama-7B

    ```sh
    git clone https://github.com/young-geng/EasyLM.git
    cd EasyLM
    export PYTHONPATH="${PWD}:$PYTHONPATH"
    ```

    and for ```EasyLM/models/llama/llama_model.py```, change line 561

    ```python
    x = self.w2(nn.silu(self.w1(x)) * self.w3(x))
    ```

    as

    ```python
    x = self.w2(jax.nn.gelu(self.w1(x)) * self.w3(x))
    ```

    for hacking gelu.

    Note that current SecretFlow does not support `numpy.complex64`,
    you should re-implement line 321~351 without `numpy.complex64`,
    or comment them for computational & communication evaluation.

3. Launch SPU backend runtime

    ```sh
    bazel run -c opt //examples/python/utils:nodectl -- --config `pwd`/examples/python/ml/flax_llama7b/3pc.json up
    ```

4. Run `flax_llama7b` example

    ```sh
    bazel run -c opt //examples/python/ml/flax_llama7b -- --config `pwd`/examples/python/ml/flax_llama7b/3pc.json
    ```

5. Run `secure inferce of GPT2 with Puma`:

    load gpt2 model, replace line 55-57 as follows:

    ```python
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    pretrained_model = FlaxGPT2LMHeadModel.from_pretrained("gpt2")
    ```

    prepare the configuration, replace line 128-129 as follows:

    ```python
    config = GPT2Config()
    model = FlaxGPT2LMHeadModel(config=config)
    ```

    The other parts are similar to run `Flax-Llama7B`.
