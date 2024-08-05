# Copyright 2024 Ant Group Co., Ltd.
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


import itertools
from functools import partial
import jax.numpy as jnp
import jax
import numpy as np

import spu.utils.simulation as ppsim
import spu.spu_pb2 as spu_pb2
import spu.intrinsic as si

jax.config.update("jax_enable_x64", True)


def type_cast(x, to_type):
    new_x = x.astype(to_type)
    return new_x


if __name__ == "__main__":
    """
    You can modify the code below for debug purpose only.
    Please DONT commit it unless it will cause build break.
    """

    sim = ppsim.Simulator.simple(3, spu_pb2.ProtocolKind.ABY3)
    copts = spu_pb2.CompilerOptions()
    # Tweak compiler options
    copts.disable_div_sqrt_rewrite = True

    types = [jnp.int32, jnp.int64, jnp.float16, jnp.float32]

    for t1, t2 in list(itertools.product(types, types)):
        if t1 == t2:
            continue
        print(f"from type: {t1}, to type: {t2}")
        x = np.random.randn(3, 4).astype(t1)
        fn = partial(type_cast, to_type=t2)
        spu_fn = ppsim.sim_jax(sim, fn, copts=copts)
        z = spu_fn(x)
        print(f"cpu input = {x}")
        print(f"spu out = {z}")
        print(f"cpu out = {fn(x)}")
        assert jnp.allclose(z, fn(x), atol=1e-2)
    print(spu_fn.pphlo)
