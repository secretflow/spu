# Flax Llama-7B Example with Puma

This example demonstrates how to use SPU to run secure inference on a pre-trained
[Llama-7B](https://research.facebook.com/publications/llama-open-and-efficient-foundation-language-models/) model using [Puma](https://arxiv.org/abs/2307.12533).

> **_NOTE:_**  To run LLaMA-7B with ABY3, each node requires at least 1TB of RAM

1. Install huggingface transformers library

    ```sh
    pip install 'transformers[flax]'
    ```

2. Download EasyML library to support Flax-LLaMA-7B

    ```sh
    git clone https://github.com/young-geng/EasyLM.git
    cd EasyLM
    export PYTHONPATH="${PWD}:$PYTHONPATH"
    cd ./EasyLM/models/llama
    ```

    Since EasyLM have an issue，so we have to make a samll change to support the option "streaming=false".
    Open and edit "convert_hf_to_easylm.py", chang this:

    ```python
       parser.add_argument("--streaming", action="store_true", default=True, help="whether is model weight saved stream format",)
    ```

    to:

    ```python
       parser.add_argument("--streaming", action="store_true", default=False, help="whether is model weight saved stream format",)
    ```

    Download trained LLaMA-B[PyTroch-Version] from [Hugging Face](https://huggingface.co/openlm-research/open_llama_7b)
    , and convert it to Flax.msgpack as:

    ```sh
    python convert_hf_to_easylm.py  \
       --checkpoint_dir     path-to-flax-llama7b-dir    \
       --output_file path-to-flax-llama7b-EasyLM.msgpack  \
       --model_size 7b \
       --streaming false
    ```

3. Launch SPU backend runtime

    ```sh
    uv run examples/python/utils/nodectl.py  --config ../ml/flax_llama7b/3pc.json up
    ```

4. Run `flax_llama7b` example

    ```sh
    uv run examples/python/ml/flax_llama7b/flax_llama7b_split.py --model_path dir-to-flax-llama7b-EasyLM   --config ./3pc.json
    ```

    and you can get the following results from our example:

    ```md
    ------
    Run on CPU
    Q: What is the largest animal?
    A: The largest animal is the blue whale.
    Q: What is the smallest animal?
    A: The smallest animal is the bee.

    ------
    Run on SPU
    Q: What is the largest animal?
    A: The largest animal is the blue whale.
    Q: What is the smallest animal?
    A: The smallest animal is the bee.
    ```

5. To reproduce the benchmarks results in the [Puma paper](https://arxiv.org/abs/2307.12533), please check [here](https://github.com/AntCPLab/puma_benchmarks).
