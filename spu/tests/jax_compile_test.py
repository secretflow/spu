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


import unittest

from flax import linen as nn
from jax import numpy as jnp
from jax import random

import spu.spu_pb2 as spu_pb2
import spu.utils.simulation as ppsim


def identity(x):
    return x


class UnitTests(unittest.TestCase):
    # https://github.com/secretflow/spu/issues/428
    def test_cache_with_static_argnums(self):
        sim = ppsim.Simulator.simple(1, spu_pb2.ProtocolKind.REF2K, 64)

        power_list = [-1, 0, 1, 2, 3]

        for _ in range(0, 50):
            for p in power_list:
                spu_result = ppsim.sim_jax(sim, identity, static_argnums=(0,))(p)
                self.assertEqual(p, spu_result)

            p = power_list[0]
            spu_result = ppsim.sim_jax(sim, identity, static_argnums=(0,))(p)
            self.assertEqual(p, spu_result)

            p = 2
            spu_result = ppsim.sim_jax(sim, identity, static_argnums=(0,))(p)
            self.assertEqual(p, spu_result)

    # https://github.com/secretflow/spu/issues/306
    def test_compile_nn_layer(self):
        sim = ppsim.Simulator.simple(1, spu_pb2.ProtocolKind.REF2K, 64)

        class LinearModel(nn.Module):
            features: int
            dtype: jnp.dtype = jnp.float32

            def setup(self):
                self.layer = nn.Dense(self.features, use_bias=False, dtype=self.dtype)

            def __call__(self, x):
                params = {"params": self.layer.variables['params']}

                # SPU with nn.Module.apply - Doesn't work
                spu_dense = ppsim.sim_jax(sim, self.layer.apply)
                spu_y = spu_dense(params, x)

                return spu_y

        model = LinearModel(features=5)

        key = random.PRNGKey(0)
        params = {'params': {'layer': {'kernel': random.normal(key, (10, 5))}}}
        x = random.normal(key, (10,))

        _ = model.apply(params, x)


if __name__ == "__main__":
    unittest.main()
