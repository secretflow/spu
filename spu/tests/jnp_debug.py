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

import sys
import json
import pdb
import unittest
import itertools
import math
import jax.numpy as jnp
import jax.lax as lax
import numpy as np

import spu.intrinsic as si
import spu.spu_pb2 as spu_pb2
import spu.utils.simulation as ppsim
import spu.utils.distributed as ppd

# Let client receive encrypted weight to compute HE matmul
# change arithmetic.cc->profiling to use fake encrypted weight for profiling

# bazel run -c opt //examples/python/utils:nodectl -- --config `pwd`/examples/python/ml/flax_bert/2pc.json up
# bazel run -c opt spu/tests/jnp_debug

if __name__ == "__main__":
    """
    You can modify the code below for debug purpose only.
    Please DONT commit it unless it will cause build break.
    """

    # sim = ppsim.Simulator.simple(
    #     2, spu_pb2.ProtocolKind.CHEETAH, spu_pb2.FieldType.FM64
    # )
    with open("/home/lizhengyi.lzy/ppu/examples/python/ml/flax_bert/2pc.json", 'r') as file:
        conf = json.load(file)

    ppd.init(conf["nodes"], conf["devices"])

    copts = spu_pb2.CompilerOptions()
    # Tweak compiler options
    copts.disable_div_sqrt_rewrite = True

    if len(sys.argv) == 5:
        b = int(sys.argv[1])
        m = int(sys.argv[2])
        k = int(sys.argv[3])
        n = int(sys.argv[4])
    else:
        # b = np.random.randint(1, 10)

        # m = np.random.randint(100, 20000)
        # k = np.random.randint(200, 600)
        # n = np.random.randint(1, 500)

        # pn = np.random.randint(60, 100)
        # pk = np.random.randint(60, 100)

        # force to perform 1 * vector
        b=1
        m,k,n=8192,768,512
    pn,pk=8,8

    mask = np.ones((b, k, n), dtype=np.bool_)

    n_blocks = math.ceil(n / pn)
    k_blocks = math.ceil(k / pk)

    # for k_idx, n_idx in itertools.product(range(k_blocks), range(n_blocks)):
    #     for b_idx in range(b):
    #         if np.random.random() > 0.2:
    #             # prune
    #             mask[
    #                 b_idx, k_idx * pk : (k_idx + 1) * pk, n_idx * pn : (n_idx + 1) * pn
    #             ] = 0

    x = np.random.randn(b, m, k)
    y = np.random.randn(b, k, n)

    prune_pattern = np.array([pk, pn], dtype=np.int64)

    fn = lambda x, y: si.sparse_dot_general(
        x,
        y,
        mask,
        prune_pattern,
        dimension_numbers=(((x.ndim - 1,), (1,)), ((0,), (0,))),
    )
    # fn = lambda x, y: lax.dot_general(x, y,dimension_numbers = (((x.ndim - 1,), (0,)), ((), ())),)
    # fn = lambda x, y: jnp.matmul(x, y)
    # cpu_out = fn(x, y)

    # spu_fn = ppsim.sim_jax(sim, fn, copts=copts)
    # spu_out = spu_fn(x, y)
    cpu_out=np.matmul(x, y)
    x = ppd.device("P1")(lambda x: x)(x)
    y = ppd.device("P2")(lambda x: x)(y)
    spu_out = ppd.device("SPU")(
        fn, copts=copts
    )(x,y)
    spu_out = ppd.get(spu_out)


    print("cpu_out", cpu_out)
    print("spu_out", spu_out)

    diff=cpu_out-spu_out
    diff[np.absolute(diff)<0.001]=0
    diff[np.absolute(diff)>0.001]=1
    print("correctness count:",np.sum(1-diff))
    print("correctness ratio:",1-np.mean(diff))
    print("Overall correctness:",np.isclose(cpu_out, spu_out, atol=0.0005).all())
    print("maximal difference:",np.max(np.absolute(cpu_out-spu_out)))
    print("mean difference:",np.mean(np.absolute(cpu_out-spu_out)))
    # if not np.isclose(cpu_out, spu_out, atol=0.0005).all():
    #     with open("/home/lizhengyi.lzy/ppu/spu/tests/logs.txt", "a") as f:
    #         f.write(f"failed: b={b}, m={m}, k={k}, n={n}\n")
    # else:
    #     with open("/home/lizhengyi.lzy/ppu/spu/tests/logs.txt", "a") as f:
    #         f.write(f"passed: b={b}, m={m}, k={k}, n={n}\n")
