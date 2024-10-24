# Flax BERT Example

This example demonstrates how to use SPU to run private inference on a GLUE benchmark.

1. Install huggingface transformers library

    ```sh
    pip install 'transformers[flax]'
    ```

2. Launch SPU backend runtime

    ```sh
    bazel run -c opt //examples/python/utils:nodectl -- --config `pwd`/examples/python/ml/flax_bert/2pc.json up
    bazel run -c opt //examples/python/utils:nodectl -- --config `pwd`/examples/python/ml/flax_bert/3pc.json up
    ```

3. Run `flax_bert` example

    ```sh
    bazel run -c opt //examples/python/ml/flax_bert -- --config `pwd`/examples/python/ml/flax_bert/2pc.json
    bazel run -c opt //examples/python/ml/flax_bert -- --config `pwd`/examples/python/ml/flax_bert/3pc.json
    ```

4. 实现了Nibums中对Bumblebee (仓库中叫Cheetah) 协议的改进。使用时需要注意/home/lizhengyi.lzy/ppu/libspu/mpc/cheetah/arithmetic.cc中的profiling的参数。此参数设计的目的是：协议中将模型参数的密文提前处理好存入disk，运行时异步载入内存。将profiling设置为False，实现中需要通过一次额外的通信完成，此模式可以验证协议的正确性。如果要测试协议的性能，将profilng设置为True。此模式不需要管正确性，参数密文将随机生成做运算。

5. 测试入口：/home/lizhengyi.lzy/ppu/spu/tests/jnp_debug.py测试线性层计算和/home/lizhengyi.lzy/ppu/spu/tests/jnp_debug_act_func.py测试非线性层计算。
