# Ditto: Quantization-aware Secure Inference of Transformers upon MPC
---
This repo contains a proof-of-concept implementation for our [paper](https://openreview.net/forum?id=ZzXNCQGzqT).

This is the **experimental branch** and the codes are still under heavy developments, and should not be used in any security sensitive product.

## Requirements
Our implementations are built on top of the [SPU](https://github.com/secretflow/spu) and [MPCFormer](https://github.com/DachengLi1/MPCFormer) libraries.

NOTE:
1. **MPC-based secure inference**: In this repo, we only provide the MPC-related protocol constructions and the end-to-end implementation of secure LLM inference.
2. **Model quant & distillation**: The quantization-aware distillation of the plaintext model is provided [here](https://github.com/llCurious/MPCFormer).

> You need to train a quantized model using the above link, and then use SPU to load the model to perform secure inference.

**Alternatively, you can directly use SPU to load a original model to evaluate its efficiency.**

## Build
### 1. Prerequisite
Please follow the instructions in SPU.

```bash
Install gcc>=11.2, cmake>=3.18, ninja, nasm>=2.15, python==3.8, bazel==6.2.1, golang

python3 -m pip install -r requirements.txt
python3 -m pip install -r requirements-dev.txt
python3 -m pip install transformers datasets
```

### 2. Parameter Configuration
- 3PC MPC computation configuration: `examples/python/ml/experimental_mp/3pc_nExp.json`
``` json
"runtime_config": {
    // use ABY3 protocol
    "protocol": "ABY3",
    // if true, log detailed runtime/communication cost
    "enable_pphlo_profile": true,
    "enable_hal_profile": true,
    "enable_pphlo_trace": false,
    // the iter for division appr.
    "fxpDivGoldschmidtIters": 5,
    // mmul split is useful when the input matrix is too large
    "experimental_disable_mmul_split": true,
    // enable op parallel
    "experimental_enable_intra_op_par": true,
    // 0 for Taylor series and 1 for Pade appr.
    "fxp_exp_mode": 0,
    // the iteration round for Taylor series
    "fxp_exp_iters": 5
}
```
- Network Setting: `examples/python/ml/experimental_mp/net_env.sh`

``` bash
# LAN with a bandwidth of 5Gbps and 0.4ms round-trip time
bash net_env.sh lan
# WAN with a band- width of 400Mbps and 40ms round-trip time
bash net_env.sh wan
# remove the network limit
bash net_env.sh del
```

- Non-linear function approximations: For `GeLU` and `Softmax`, we support several configufations, which are in line with MPCFormer. **NOTE: Ditto uses `quad` appr. for GeLU and use standard Softmax for the sake of accuracy.**
```python
parser.add_argument("--gelu", default="raw", help="['raw', 'quad', 'poly']")
parser.add_argument("--softmax", default="raw", help="['raw', '2relu', '2quad']")
```


### 3. Running Script

#### 3.1 Ring Cast Kernel Simulation
You can run `spu/tests/jnp_dynamic_ring_test.py` to test the type cast functionality in SPU.

We support the mutual conversion between data of types = [jnp.int32, jnp.int64, jnp.float16, jnp.float32].

```bash
bazel run //spu/tests:jnp_dynamic_ring_test
```

#### 3.2 3PC Transformer inference
1. First, you need to launch the SPU backend.
```bash
bazel run -c opt //examples/python/utils:nodectl -- --config `pwd`/examples/python/ml/experimental_mp/3pc_nExp.json up
```
2. Second, you can refer to the `shell_cmd.sh` for scripts for different baselines.

Example run script:
```bash
bazel run -c opt //examples/python/ml/experimental_mp:{gpt2_bench, bert_bench} -- --gelu quad --config examples/python/ml/experimental_mp/3pc_nExp.json
```

### 4. Modify Some Python Codes
The above codes implement the secure inference of Transformers, which are the basis of our work.

Based on SPU, Ditto incorporates the **functionality of dynamic rings upon MPC**, which in turn supports the quantization-aware secure inference of Transformers.

The code for this part is provided in `libspu/kernel/hal/type_cast.cc` and `libspu/mpc/aby3/conversion.cc`.

To automatically activate such functionality, we need to manually configure the frontend Python programs, which are modified in a mixed-precision manner.

Currently, we use `jnp.float16` and `jnp.float32` for low-, and high-precision, which correspond to 32-bit ring and 64-bit ring, respectively.

We modify the code in Huggingface.
In general, we compute the linear layers with jnp.float16, while for those non_linear functions like Softmax, LayerNorm, we use jnp.float32.
NOTE: The plaintext configuration serves as the inputs to SPU compiler, which an automatically computes over dynamic rings.

```python
# transformers/models/gpt2/modeling_flax_gpt2.py#540
# linear computations use low-precision
self.low_dtype = jnp.float16
self.blocks = [
    FlaxGPT2Block(self.config, name=str(i), dtype=self.low_dtype)
    for i in range(self.config.num_hidden_layers)
]

# transformers/models/bert/modeling_flax_bert.py#636
self.low_dtype = jnp.float16
self.layer = FlaxBertLayerCollection(
    self.config,
    dtype=self.low_dtype,
    gradient_checkpointing=self.gradient_checkpointing,
)

# transformers/models/gpt2/modeling_flax_gpt2.py#262
attn_weights = dot_product_attention_weights(
    query,
    key,
    bias=attention_bias,
    dropout_rng=dropout_rng,
    dropout_rate=self.config.attn_pdrop,
    deterministic=deterministic,
    dtype=self.dtype,
    precision=None,
)
# attn_weights cast to high-precision
attn_weights = attn_weights.astype(jnp.float32)

# transformers/models/gpt2/modeling_flax_gpt2.py#291
# self.act = ACT2FN[self.config.activation_function]
self.act = jax.nn.gelu # this is used to hack gelu function.

# jax/_src/nn/functions.py#400
def _softmax(x, axis, where, initial):
    x_max = jnp.max(x, axis, where=where, initial=initial, keepdims=True)
    # hack
    x_bias = x - lax.stop_gradient(x_max)
    x_bias = x_bias.astype(jnp.float32)
    unnormalized = jnp.exp(x_bias)
    result = unnormalized / jnp.sum(unnormalized, axis, where=where, keepdims=True)
    if where is not None:
        result = jnp.where(where, result, 0)
    return result

# flax/linen/normalization.py#87
# promote x to at least float32
dtype = jnp.promote_types(dtype, jnp.float32)
```

### 5. Accuracy concern
In order to avoid excessive accuracy loss, you need to load the quantized models after quantization-aware distillation, which is provided in [here](https://github.com/llCurious/MPCFormer).

### 6. Citation
This paper is accepted in ICML 2024.

```bib

@InProceedings{pmlr-v235-wu24d,
  title = 	 {Ditto: Quantization-aware Secure Inference of Transformers upon {MPC}},
  author =       {Wu, Haoqi and Fang, Wenjing and Zheng, Yancheng and Ma, Junming and Tan, Jin and Wang, Lei},
  booktitle = 	 {Proceedings of the 41st International Conference on Machine Learning},
  pages = 	 {53346--53365},
  year = 	 {2024},
  editor = 	 {Salakhutdinov, Ruslan and Kolter, Zico and Heller, Katherine and Weller, Adrian and Oliver, Nuria and Scarlett, Jonathan and Berkenkamp, Felix},
  volume = 	 {235},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {21--27 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://raw.githubusercontent.com/mlresearch/v235/main/assets/wu24d/wu24d.pdf},
  url = 	 {https://proceedings.mlr.press/v235/wu24d.html},
  abstract = 	 {Due to the rising privacy concerns on sensitive client data and trained models like Transformers, secure multi-party computation (MPC) techniques are employed to enable secure inference despite attendant overhead. Existing works attempt to reduce the overhead using more MPC-friendly non-linear function approximations. However, the integration of quantization widely used in plaintext inference into the MPC domain remains unclear. To bridge this gap, we propose the framework named Ditto to enable more efficient quantization-aware secure Transformer inference. Concretely, we first incorporate an MPC-friendly quantization into Transformer inference and employ a quantization-aware distillation procedure to maintain the model utility. Then, we propose novel MPC primitives to support the type conversions that are essential in quantization and implement the quantization-aware MPC execution of secure quantized inference. This approach significantly decreases both computation and communication overhead, leading to improvements in overall efficiency. We conduct extensive experiments on Bert and GPT2 models to evaluate the performance of Ditto. The results demonstrate that Ditto is about $3.14\sim 4.40\times$ faster than MPCFormer (ICLR 2023) and $1.44\sim 2.35\times$ faster than the state-of-the-art work PUMA with negligible utility degradation.}
}
```

### Appendix
NOTE: This is not bounded to Ditto.
Yet, we here demonstrate how to use Haiku and JMP to perform mixed-precision training.

This example comes from Haiku official github [repo](https://github.com/deepmind/dm-haiku/tree/main/examples).

1. Install packages

    ``` sh
    pip install dm-haiku jmp
    ```

2. (Currently Optional) Launch SPU backend runtime

    ``` sh
    bazel run -c opt //examples/python/utils:nodectl -- --config `pwd`/examples/python/ml/experimental_mp/3pc.json up
    ```

3. Run `mlp_mp` example

    ``` sh
    bazel run -c opt //examples/python/ml/experimental_mp:mlp_mp
    ```

4. Run `resnet_mp` example

    ``` sh
    bazel run -c opt //examples/python/ml/experimental_mp:resnet_mp
    ```
