# Flax Llama-7B Example with Model Split

This example demonstrates how to use SPU to run secure inference on a pre-trained
[Llama-7B](https://research.facebook.com/publications/llama-open-and-efficient-foundation-language-models/) model with model split.

1. Motivation

- Time:
Using the full Llama model for inference on SPU can take a significant amount of time. If only a portion of the model is passed through SPU to ensure privacy, it can greatly improve inference efficiency.

- RAM Usage:
Using the full Llama model for inference on SPU requires a large amount of memory(more than 256 GB). Splitting the model can significantly reduce memory usage, making it available for use in hardware-constrained environments.

2. Download EasyML library to support Flax-LLaMA-7B

    ```sh
    git clone https://github.com/young-geng/EasyLM.git
    cd EasyLM
    export PYTHONPATH="${PWD}:$PYTHONPATH"
    ```
    Or use a fork created for transformer split.

    ```sh
    git clone -b split_llama_2 https://github.com/Rainysponge/EasyLM.git
    cd EasyLM
    export PYTHONPATH="${PWD}:$PYTHONPATH"
    ```

    Install EasyLM Environment Before Install Secretflow & SPU
    
    ```sh
    conda env create -f examples/python/ml/flax_llama7b_split/gpu_environment.yml
    conda activate EasyLM
    pip install 'transformers[flax]'
    pip install -U secretflow
    pip install spu
    ```

    If Do not Want to use GPU

    ```sh
        pip uninstall jax jaxlib
        pip install jax==0.4.11 jaxlib==0.4.11
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

    Move the python file to EasyLM if you do not use the fork fork created for transformer split.

    ```sh
    cp path-to-llama_model_split_transformer_py path_to_EasyLM/EasyLM/models/llama
    ```

3. Launch SPU backend runtime

    ```sh
    bazel run -c opt //examples/python/utils:nodectl -- --config `pwd`/examples/python/ml/flax_llama7b_split/3pc.json up
    ```
    
    or
    （recommended）

    ```sh
    cd examples/python/utils
    python nodectl.py  --config `pwd`/examples/python/ml/flax_llama7b_split/3pc.json up
    ```

4. Run `flax_llama7b_split` example

    ```sh
    bazel run -c opt //examples/python/ml/flax_llama7b_split -- --config `pwd`/examples/python/ml/flax_llama7b_split/3pc.json
    ```

    or（recommended）

    ```sh
    cd examples/python/ml/flax_llama7b_split
    python flax_llama7b_split.py --config `pwd`/examples/python/ml/flax_llama7b_split/3pc.json
    ```

    and you can get the following results from our example:

    ```md
    ------
    Run on CPU
    Q: What is the largest animal?
    A: The largest animal is the blue whale.
    generate on CPU: 256.08824276924133 seconds

    ------
    Run on SPU
    [0000-00-00 00:00:00.000] [info] [thread_pool.cc:30] Create a fixed thread pool with size 127
    Q: What is the largest animal?
    A: The largest animal is the blue whale.
    generate  on SPU: 812.9427680969238 seconds
    ```
    RAM peak: 64.5888GB

    And If you set token_num to 30, you can get the following results:
    ```sh
   ------
    Run on CPU
    Q: What is the largest animal?
    A: The largest animal is the blue whale.
    Q: What is the smallest animal?
    A: The smallest animal is the bacterium.
    generate on CPU: 837.0810837745667 seconds

    ------
    Run on SPU
    [0000-00-00 00:00:00.000] [info] [thread_pool.cc:30] Create a fixed thread pool with size 127
    Q: What is the largest animal?
    A: The largest animal is the blue whale.
    Q: What is the smallest animal?
    A: The smallest animal is the bacterium.
    generate  on SPU: 2760.5035905838013 seconds
    ```
