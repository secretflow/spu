# Flax Llama-7B Example with Puma

This example demonstrates how to use SPU to run secure inference on a pre-trained
[Llama-7B](https://research.facebook.com/publications/llama-open-and-efficient-foundation-language-models/) model using [Puma](https://arxiv.org/abs/2307.12533)

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

5. To reproduce the benchmarks results in the [Puma paper](https://arxiv.org/abs/2307.12533), please check [here](https://github.com/AntCPLab/puma_benchmarks).

   
