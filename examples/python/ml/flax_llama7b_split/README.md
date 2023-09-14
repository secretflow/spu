# Flax Llama-7B Example with Model Split

This example demonstrates how to use SPU to run secure inference on a pre-trained
[Llama-7B](https://research.facebook.com/publications/llama-open-and-efficient-foundation-language-models/) model with model split.

1. Install huggingface transformers library

    ```sh
    pip install 'transformers[flax]'
    ```

2. Download EasyML library to support Flax-LLaMA-7B

    ```sh
    git clone https://github.com/young-geng/EasyLM.git
    cd EasyLM
    export PYTHONPATH="${PWD}:$PYTHONPATH"
    ```

    Download trained LLaMA-7B[PyTroch-Version] from "https://github.com/facebookresearch/llama", and convert it to EasyLM format as:

    ```sh
    cd path_to_EasyLM/EasyLM/models/llama
    python convert_hf_to_easylm.py  \
       --checkpoint_dir     path_to_llama_weights    \
       --output_file path_to_outputfile  \
       --model_size 7b \
       --streaming
    ```

    Dove the python file to EasyLM

    ```sh
    cp path-to-llama_model_split_transformer_py path_to_EasyLM/EasyLM/models/llama
    ```

3. Launch SPU backend runtime

    ```sh
    bazel run -c opt //examples/python/utils:nodectl -- --config `pwd`/examples/python/ml/flax_llama7b_split/3pc.json up
    ```

4. Run `flax_llama7b_split` example

    ```sh
    bazel run -c opt //examples/python/ml/flax_llama7b -- --config `pwd`/examples/python/ml/flax_llama7b_split/3pc.json
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


