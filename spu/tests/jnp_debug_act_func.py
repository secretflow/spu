# Copyright 2021 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import pdb
import unittest
import itertools
import math
import jax
import jax.numpy as jnp
import jax.lax as lax
import flax.linen as nn
import numpy as np
import torch
import json

# import spu.intrinsic as si
import spu
import spu.spu_pb2 as spu_pb2
import spu.utils.simulation as ppsim
import spu.utils.distributed as ppd
import spu.intrinsic as si

import os, sys
sys.path.append('/home/lizhengyi.lzy/sparse_ppml2')
sys.path.append('/home/lizhengyi.lzy/ppu/local/fake_nonlinear.py')
from GLUE.model.sparse_gelu_module import Threshold_GELU_Flax
from GLUE.model.sparse_attention_module import Threshold_SoftMax_Flax
from softmax import puma_softmax, ours_fake_softmax, puma_exp, ours_fake_exp, mpcformer_fake_softmax, bolt_fake_softmax, mpcformer_fake_exp, bolt_fake_exp
from gelu import puma_gelu, ours_fake_gelu, iron_gelu, bolt_fake_gelu, mpcformer_fake_gelu
from layernorm import puma_layernorm
# bazel run -c opt //examples/python/utils:nodectl -- --config `pwd`/examples/python/ml/flax_bert/2pc.json up
# bazel run -c opt spu/tests/jnp_debug_act_func



def test_func(x, mode):
    if mode == "comparsion":
        return (x>0)*x
    elif mode == "multiplication2":
        return jnp.square(x)
    elif mode == "multiplication3":
        return jnp.square(x)*x
    elif mode == "multiplication4":
        x2=jnp.square(x)
        x3=x2*x
        x4=jnp.square(x2)
        return x4+x3
    elif mode == "fused":
        mask = (x>0)*x
        x2=jnp.square(x)
        return mask+x2
    else:
        raise("error")

if __name__ == "__main__":
    """
    You can modify the code below for debug purpose only.
    Please DONT commit it unless it will cause build break.
    """
    with open("/home/lizhengyi.lzy/ppu/examples/python/ml/flax_bert/2pc.json", 'r') as file:
        conf = json.load(file)

    ppd.init(conf["nodes"], conf["devices"])

    # sim = ppsim.Simulator.simple(
    #     2, spu_pb2.ProtocolKind.CHEETAH, spu_pb2.FieldType.FM64,
    # )
    copts = spu_pb2.CompilerOptions()
    # Tweak compiler options
    copts.disable_div_sqrt_rewrite = True
    copts.enable_optimize_denominator_with_broadcast = True

    parser = argparse.ArgumentParser(description='Process the test flags.')
    parser.add_argument('--test_func', action='store_true', help='Set TEST_FUNC to True')
    parser.add_argument('--mode', type=str, default='fused', help='Mode string argument (default: fused)')

    parser.add_argument('--test_oursreal_gelu', action='store_true', help='Set TEST_GELU to True')
    parser.add_argument('--test_oursreal_softmax', action='store_true', help='Set TEST_SOFTMAX to True')
    parser.add_argument('--test_oursfake_gelu', action='store_true', help='Set TEST_SOFTMAX to True')
    parser.add_argument('--test_oursfake_softmax', action='store_true', help='Set TEST_SOFTMAX to True')
    parser.add_argument('--test_oursfake_exp', action='store_true', help='Set TEST_SOFTMAX to True')
    parser.add_argument('--test_pumafake_gelu', action='store_true', help='Set TEST_SOFTMAX to True')
    parser.add_argument('--test_pumafake_softmax', action='store_true', help='Set TEST_SOFTMAX to True')
    parser.add_argument('--test_pumafake_exp', action='store_true', help='Set TEST_SOFTMAX to True')
    parser.add_argument('--test_ironfake_exp', action='store_true', help='Set TEST_SOFTMAX to True')
    parser.add_argument('--test_ironfake_gelu', action='store_true', help='Set TEST_SOFTMAX to True')
    parser.add_argument('--test_ironfake_softmax', action='store_true', help='Set TEST_SOFTMAX to True')
    parser.add_argument('--test_mpcformerfake_exp', action='store_true', help='Set TEST_SOFTMAX to True')
    parser.add_argument('--test_mpcformerfake_gelu', action='store_true', help='Set TEST_SOFTMAX to True')
    parser.add_argument('--test_mpcformerfake_softmax', action='store_true', help='Set TEST_SOFTMAX to True')
    parser.add_argument('--test_boltfake_exp', action='store_true', help='Set TEST_SOFTMAX to True')
    parser.add_argument('--test_boltfake_gelu', action='store_true', help='Set TEST_SOFTMAX to True')
    parser.add_argument('--test_boltfake_softmax', action='store_true', help='Set TEST_SOFTMAX to True')
    parser.add_argument('--test_pumafake_layernorm', action='store_true', help='Set TEST_SOFTMAX to True')
    parser.add_argument('--test_pumafake_qk', action='store_true', help='Set TEST_SOFTMAX to True')
    parser.add_argument('--test_pumafake_pv', action='store_true', help='Set TEST_SOFTMAX to True')
    # Add positional arguments for integers
    parser.add_argument('--b', type=int, help='Integer argument b')
    parser.add_argument('--token_num', type=int, help='Integer argument token_num')
    parser.add_argument('--hidden', type=int, default=768, help='Integer argument hidden')

    # Parse the arguments
    args = parser.parse_args()
    TEST_FUNC = args.test_func
    test_oursreal_gelu = args.test_oursreal_gelu
    test_oursreal_softmax = args.test_oursreal_softmax
    test_oursfake_gelu = args.test_oursfake_gelu
    test_oursfake_softmax = args.test_oursfake_softmax
    test_oursfake_exp = args.test_oursfake_exp
    test_pumafake_gelu = args.test_pumafake_gelu
    test_pumafake_softmax = args.test_pumafake_softmax
    test_pumafake_exp = args.test_pumafake_exp
    test_ironfake_exp = args.test_ironfake_exp
    test_ironfake_gelu = args.test_ironfake_gelu
    test_ironfake_softmax = args.test_ironfake_softmax
    test_mpcformerfake_exp = args.test_mpcformerfake_exp
    test_mpcformerfake_gelu = args.test_mpcformerfake_gelu
    test_mpcformerfake_softmax = args.test_mpcformerfake_softmax
    test_boltfake_exp = args.test_boltfake_exp
    test_boltfake_gelu = args.test_boltfake_gelu
    test_boltfake_softmax = args.test_boltfake_softmax

    test_pumafake_layernorm = args.test_pumafake_layernorm
    test_pumafake_qk = args.test_pumafake_qk
    test_pumafake_pv = args.test_pumafake_pv

    b = args.b
    token_num = args.token_num
    hidden = args.hidden
    if TEST_FUNC: mode = args.mode


    if TEST_FUNC:
        x = np.random.randn(b, token_num, hidden)
        fn = lambda x : test_func(x, mode)

        # spu_fn = ppsim.sim_jax(sim, fn, copts=copts)
        # spu_out = spu_fn(x)
        cpu_out = test_func(x, mode)

        x = ppd.device("P2")(lambda x: x)(x)
        spu_out = ppd.device("SPU")(
            fn, copts=copts
        )(x)
        spu_out = ppd.get(spu_out)


        print(np.isclose(cpu_out, spu_out, atol=0.001).all())

    if test_oursreal_gelu:
        x = np.random.randn(b, token_num, hidden)

        gelu=Threshold_GELU_Flax(strategy="clip_poly", order=2)
        def ours_gelu(x):
            return gelu(x)

        # spu_fn = ppsim.sim_jax(sim, fn, copts=copts)
        # spu_out = spu_fn(x)
        cpu_out = ours_gelu(x)

        x = ppd.device("P2")(lambda x: x)(x)
        spu_out = ppd.device("SPU")(
            ours_gelu, copts=copts
        )(x)
        spu_out = ppd.get(spu_out)

        print(np.isclose(cpu_out, spu_out, atol=0.001).all())

    if test_oursreal_softmax:
        head_num=12
        clip_thres=-4.0

        x = torch.randn((b, head_num, token_num, token_num))
        sparse_mask = torch.zeros((b, head_num, token_num, token_num))
        attention_mask = torch.zeros(b, 1, 1, token_num)

        # attention mask 可以有，单输入时总是0，注释即可
        # for i in range(bs):
        #     n = torch.randint(hidden, (1,)).item()
        #     attention_mask[i, :,:,n:] = -10000000000000

        value_for_make=-10000000000000
        expanded_attention_mask=attention_mask/2+attention_mask.transpose(-1, -2)/2
        expanded_attention_mask[expanded_attention_mask<-100000]=value_for_make
        expanded_attention_mask=expanded_attention_mask.repeat(1,head_num,1,1)
        hard_attention_mask=torch.ones_like(expanded_attention_mask) # 0/1 version of the attention_mask
        hard_attention_mask[expanded_attention_mask<-100000]=0

        x = jnp.array(x)
        sparse_mask = jnp.array(sparse_mask)
        hard_attention_mask = jnp.array(hard_attention_mask)


        softmax=Threshold_SoftMax_Flax(dim=-1, strategy="clip_poly", clip_thres=clip_thres, num_bits=5)
        def ours_softmax(x):
            return softmax(x, sparse_mask, hard_attention_mask)
        cpu_out, _ = ours_softmax(x)

        x = ppd.device("P2")(lambda x: x)(x)
        spu_out, _ = ppd.device("SPU")(
            ours_softmax, copts=copts
        )(x)
        spu_out = ppd.get(spu_out)


        # spu_fn = ppsim.sim_jax(sim, fn, copts=copts)
        # spu_out, _ = spu_fn(x, sparse_mask, hard_attention_mask)

        # print("cpu_out", cpu_out)
        # print("spu_out", spu_out)

        print(np.isclose(cpu_out, spu_out, atol=0.001).all())

    if test_pumafake_gelu:
        x = np.random.randn(b, token_num, hidden)

        # spu_fn = ppsim.sim_jax(sim, fn, copts=copts)
        # spu_out = spu_fn(x)
        cpu_out = puma_gelu(x)

        print(ppd.device("SPU")(puma_gelu).dump_pphlo(x))
        x = ppd.device("P2")(lambda x: x)(x)
        spu_out = ppd.device("SPU")(
            puma_gelu, copts=copts
        )(x)
        spu_out = ppd.get(spu_out)

        print(np.isclose(cpu_out, spu_out, atol=0.01).all(),(np.absolute(cpu_out-spu_out).max()),np.absolute(cpu_out-spu_out).mean())

    if test_pumafake_softmax:
        head_num=12

        x = np.random.randn(b, head_num, token_num, token_num)

        cpu_out = puma_softmax(x)

        print(ppd.device("SPU")(puma_softmax).dump_pphlo(x))
        x = ppd.device("P2")(lambda x: x)(x)
        spu_out = ppd.device("SPU")(
            puma_softmax, copts=copts
        )(x)
        spu_out = ppd.get(spu_out)

        print(np.isclose(cpu_out, spu_out, atol=0.01).all(),(np.absolute(cpu_out-spu_out).max()),np.absolute(cpu_out-spu_out).mean())

    if test_mpcformerfake_softmax:
        head_num=12

        x = np.random.randn(b, head_num, token_num, token_num)

        cpu_out = mpcformer_fake_softmax(x)

        print(ppd.device("SPU")(mpcformer_fake_softmax).dump_pphlo(x))
        x = ppd.device("P2")(lambda x: x)(x)
        spu_out = ppd.device("SPU")(
            mpcformer_fake_softmax, copts=copts
        )(x)
        spu_out = ppd.get(spu_out)

        print(np.isclose(cpu_out, spu_out, atol=0.01).all(),(np.absolute(cpu_out-spu_out).max()),np.absolute(cpu_out-spu_out).mean())

    if test_boltfake_softmax:
        head_num=12

        x = np.random.randn(b, head_num, token_num, token_num)

        cpu_out = bolt_fake_softmax(x)

        print(ppd.device("SPU")(bolt_fake_softmax).dump_pphlo(x))
        x = ppd.device("P2")(lambda x: x)(x)
        spu_out = ppd.device("SPU")(
            bolt_fake_softmax, copts=copts
        )(x)
        spu_out = ppd.get(spu_out)

        print(np.isclose(cpu_out, spu_out, atol=0.01).all(),(np.absolute(cpu_out-spu_out).max()),np.absolute(cpu_out-spu_out).mean())

    if test_ironfake_softmax:
        head_num=12

        x = np.random.randn(b, head_num, token_num, token_num)

        cpu_out = puma_softmax(x)

        print(ppd.device("SPU")(puma_softmax).dump_pphlo(x))
        x = ppd.device("P2")(lambda x: x)(x)
        spu_out = ppd.device("SPU")(
            puma_softmax, copts=copts
        )(x)
        spu_out = ppd.get(spu_out)

        print(np.isclose(cpu_out, spu_out, atol=0.01).all(),(np.absolute(cpu_out-spu_out).max()),np.absolute(cpu_out-spu_out).mean())


    if test_pumafake_exp:
        head_num=12

        x = np.random.randn(b, head_num, token_num, token_num)

        cpu_out = puma_exp(x)

        print(ppd.device("SPU")(puma_exp).dump_pphlo(x))
        x = ppd.device("P2")(lambda x: x)(x)
        spu_out = ppd.device("SPU")(
            puma_exp, copts=copts
        )(x)
        spu_out = ppd.get(spu_out)

        print(np.isclose(cpu_out, spu_out, atol=0.01).all(),(np.absolute(cpu_out-spu_out).max()),np.absolute(cpu_out-spu_out).mean())

    if test_mpcformerfake_exp:
        head_num=12

        x = np.random.randn(b, head_num, token_num, token_num)

        cpu_out = mpcformer_fake_exp(x)

        print(ppd.device("SPU")(mpcformer_fake_exp).dump_pphlo(x))
        x = ppd.device("P2")(lambda x: x)(x)
        spu_out = ppd.device("SPU")(
            mpcformer_fake_exp, copts=copts
        )(x)
        spu_out = ppd.get(spu_out)

        print(np.isclose(cpu_out, spu_out, atol=0.01).all(),(np.absolute(cpu_out-spu_out).max()),np.absolute(cpu_out-spu_out).mean())

    if test_boltfake_exp:
        head_num=12

        x = np.random.randn(b, head_num, token_num, token_num)

        cpu_out = bolt_fake_exp(x)

        print(ppd.device("SPU")(bolt_fake_exp).dump_pphlo(x))
        x = ppd.device("P2")(lambda x: x)(x)
        spu_out = ppd.device("SPU")(
            bolt_fake_exp, copts=copts
        )(x)
        spu_out = ppd.get(spu_out)

        print(np.isclose(cpu_out, spu_out, atol=0.01).all(),(np.absolute(cpu_out-spu_out).max()),np.absolute(cpu_out-spu_out).mean())

    if test_ironfake_exp:
        head_num=12

        x = np.random.randn(b, head_num, token_num, token_num)

        cpu_out = puma_exp(x)

        print(ppd.device("SPU")(puma_exp).dump_pphlo(x))
        x = ppd.device("P2")(lambda x: x)(x)
        spu_out = ppd.device("SPU")(
            puma_exp, copts=copts
        )(x)
        spu_out = ppd.get(spu_out)

        print(np.isclose(cpu_out, spu_out, atol=0.01).all(),(np.absolute(cpu_out-spu_out).max()),np.absolute(cpu_out-spu_out).mean())

    if test_oursfake_gelu:
        x = np.random.randn(b, token_num, hidden)

        # spu_fn = ppsim.sim_jax(sim, fn, copts=copts)
        # spu_out = spu_fn(x)
        cpu_out = ours_fake_gelu(x)

        print(ppd.device("SPU")(ours_fake_gelu).dump_pphlo(x))
        x = ppd.device("P2")(lambda x: x)(x)
        spu_out = ppd.device("SPU")(
            ours_fake_gelu, copts=copts
        )(x)
        spu_out = ppd.get(spu_out)

        print(np.isclose(cpu_out, spu_out, atol=0.001).all(),(np.absolute(cpu_out-spu_out).max()),np.absolute(cpu_out-spu_out).mean())

    if test_mpcformerfake_gelu:
        x = np.random.randn(b, token_num, hidden)

        # spu_fn = ppsim.sim_jax(sim, fn, copts=copts)
        # spu_out = spu_fn(x)
        cpu_out = mpcformer_fake_gelu(x)

        print(ppd.device("SPU")(mpcformer_fake_gelu).dump_pphlo(x))
        x = ppd.device("P2")(lambda x: x)(x)
        spu_out = ppd.device("SPU")(
            mpcformer_fake_gelu, copts=copts
        )(x)
        spu_out = ppd.get(spu_out)

        print(np.isclose(cpu_out, spu_out, atol=0.001).all(),(np.absolute(cpu_out-spu_out).max()),np.absolute(cpu_out-spu_out).mean())

    if test_boltfake_gelu:
        x = np.random.randn(b, token_num, hidden)

        # spu_fn = ppsim.sim_jax(sim, fn, copts=copts)
        # spu_out = spu_fn(x)
        cpu_out = bolt_fake_gelu(x)

        print(ppd.device("SPU")(bolt_fake_gelu).dump_pphlo(x))
        x = ppd.device("P2")(lambda x: x)(x)
        spu_out = ppd.device("SPU")(
            bolt_fake_gelu, copts=copts
        )(x)
        spu_out = ppd.get(spu_out)

        print(np.isclose(cpu_out, spu_out, atol=0.001).all(),(np.absolute(cpu_out-spu_out).max()),np.absolute(cpu_out-spu_out).mean())

    if test_ironfake_gelu:
        x = np.random.randn(b, token_num, hidden)

        # spu_fn = ppsim.sim_jax(sim, fn, copts=copts)
        # spu_out = spu_fn(x)
        cpu_out = iron_gelu(x)

        print(ppd.device("SPU")(iron_gelu).dump_pphlo(x))
        x = ppd.device("P2")(lambda x: x)(x)
        spu_out = ppd.device("SPU")(
            iron_gelu, copts=copts
        )(x)
        spu_out = ppd.get(spu_out)

        print(np.isclose(cpu_out, spu_out, atol=0.001).all(),(np.absolute(cpu_out-spu_out).max()),np.absolute(cpu_out-spu_out).mean())

    if test_oursfake_softmax:
        head_num=12

        x = np.random.randn(b, head_num, token_num, token_num)

        cpu_out = ours_fake_softmax(x)

        print(ppd.device("SPU")(ours_fake_softmax).dump_pphlo(x))
        x = ppd.device("P2")(lambda x: x)(x)
        spu_out = ppd.device("SPU")(
            ours_fake_softmax, copts=copts
        )(x)
        spu_out = ppd.get(spu_out)

        print(np.isclose(cpu_out, spu_out, atol=0.001).all(),(np.absolute(cpu_out-spu_out).max()),np.absolute(cpu_out-spu_out).mean())

    if test_oursfake_exp:
        head_num=12

        x = np.random.randn(b, head_num, token_num, token_num)

        cpu_out = ours_fake_exp(x)

        print(ppd.device("SPU")(ours_fake_exp).dump_pphlo(x))
        x = ppd.device("P2")(lambda x: x)(x)
        spu_out = ppd.device("SPU")(
            ours_fake_exp, copts=copts
        )(x)
        spu_out = ppd.get(spu_out)

        print(np.isclose(cpu_out, spu_out, atol=0.01).all(),(np.absolute(cpu_out-spu_out).max()),np.absolute(cpu_out-spu_out).mean())

    if test_pumafake_layernorm:
        # bazel run -c opt spu/tests/jnp_debug_act_func -- --b 1 --token_num 128 --hidden 768 --test_pumafake_layernorm

        layernorm= nn.LayerNorm(dtype=jnp.float32)

        x = np.random.randn(b, token_num, hidden)
        variables = layernorm.init(jax.random.key(1), x)
        def puma_layernorm(x):
            return layernorm.apply( variables,x)

        cpu_out = puma_layernorm(x)

        print(ppd.device("SPU")(puma_layernorm).dump_pphlo(x))
        x = ppd.device("P2")(lambda x: x)(x)
        spu_out = ppd.device("SPU")(
            puma_layernorm, copts=copts
        )(x)
        spu_out = ppd.get(spu_out)

        print(np.isclose(cpu_out, spu_out, atol=0.001).all(),np.absolute(cpu_out-spu_out).max(),np.absolute(cpu_out-spu_out).mean())

    if test_pumafake_qk:
        # bazel run -c opt spu/tests/jnp_debug_act_func -- --b 12 --token_num 128 --hidden 768 --test_pumafake_qk
        pn,pk=8,8
        mask = np.ones((b, int(hidden/b), token_num), dtype=np.bool_)
        prune_pattern = np.array([pk, pn], dtype=np.int64)

        x = np.random.randn(b, token_num, int(hidden/b))
        y = np.random.randn(b, int(hidden/b), token_num)

        def qk(x, y):
            return si.sparse_dot_general(
                x,
                y,
                mask,
                prune_pattern,
                dimension_numbers=(((x.ndim - 1,), (1,)), ((0,), (0,))),
            )

        cpu_out=np.matmul(x, y)

        x = ppd.device("P1")(lambda x: x)(x)
        y = ppd.device("P2")(lambda x: x)(y)
        spu_out = ppd.device("SPU")(
            qk, copts=copts
        )(x,y)
        spu_out = ppd.get(spu_out)

        print(np.isclose(cpu_out, spu_out, atol=0.001).all(),np.absolute(cpu_out-spu_out).max(),np.absolute(cpu_out-spu_out).mean())

    if test_pumafake_pv:
        # bazel run -c opt spu/tests/jnp_debug_act_func -- --b 1 --token_num 128 --hidden 768 --test_pumafake_pv
        pn,pk=8,8
        mask = np.ones((b, token_num, hidden), dtype=np.bool_)
        prune_pattern = np.array([pk, pn], dtype=np.int64)

        x = np.random.randn(b, token_num, token_num)
        y = np.random.randn(b, token_num, hidden)

        def pv(x, y):
            return si.sparse_dot_general(
                x,
                y,
                mask,
                prune_pattern,
                dimension_numbers=(((x.ndim - 1,), (1,)), ((0,), (0,))),
            )

        cpu_out=np.matmul(x, y)

        x = ppd.device("P1")(lambda x: x)(x)
        y = ppd.device("P2")(lambda x: x)(y)
        spu_out = ppd.device("SPU")(
            pv, copts=copts
        )(x,y)
        spu_out = ppd.get(spu_out)

        print(np.isclose(cpu_out, spu_out, atol=0.001).all(),np.absolute(cpu_out-spu_out).max(),np.absolute(cpu_out-spu_out).mean())
