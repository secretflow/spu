# Copyright 2025 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import jax
import jax.numpy as jnp
import numpy as np

import spu.libspu as libspu  # type: ignore
import spu.utils.simulation as spsim
from sml.utils.utils import sml_reveal


@jax.jit
def reveal_func_single(x):
    y = jnp.log(x)
    # reveal single value
    xx = sml_reveal(x)

    # x is 1-d array, so we fetch the first element
    pred = xx[0] > 0

    def true_branch(xx):
        return jnp.log(xx), True

    def false_branch(xx):
        return jnp.log(-xx), False

    # use jax.lax.cond to replace if-else
    yy, preds = jax.lax.cond(pred, true_branch, false_branch, xx)

    return y, yy, preds


@jax.jit
def reveal_func_list(x, y):
    xx = jnp.log(x)
    yy = jnp.log(y)

    reveal_x, reveal_y = sml_reveal([x, y])

    reveal_x_log, reveal_y_log = jnp.log(reveal_x), jnp.log(reveal_y)

    return xx, yy, reveal_x_log, reveal_y_log


@jax.jit
def reveal_while_loop(x):
    y = sml_reveal(jnp.max(x))

    def cond_fun(carry):
        _, y = carry
        # jnp.max return 0-dim array, so we can fetch y directly
        return y > 3

    def body_fun(carry):
        x, _ = carry
        new_x = x - 1
        new_y = sml_reveal(jnp.max(new_x))
        return new_x, new_y

    x, _ = jax.lax.while_loop(cond_fun, body_fun, (x, y))

    return x


class RevealTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("RevealTest setUpClass\n")
        config64 = libspu.RuntimeConfig(
            protocol=libspu.ProtocolKind.ABY3,
            field=libspu.FieldType.FM64,
            fxp_fraction_bits=18,
        )
        config64.enable_pphlo_profile = False
        config64.enable_hal_profile = False
        cls.sim = spsim.Simulator(3, config64)

    def test_reveal_single(self):
        print("RevealTest test_reveal_single start\n")
        x_int = jnp.array([1, 2, 3]) * (2**25)
        x_float = jnp.array([1.0, 2.0, 3.0]) * (2**25)

        reveal_func_single_spu = spsim.sim_jax(self.sim, reveal_func_single)

        # plain test
        y_int_plain, yy_int_plain, pred_int_plain = reveal_func_single(x_int)
        y_float_plain, yy_float_plain, pred_float_plain = reveal_func_single(x_float)

        # all computed in plaintext, suppose no large diff
        np.testing.assert_allclose(y_int_plain, y_float_plain)
        np.testing.assert_allclose(yy_int_plain, yy_float_plain)
        np.testing.assert_allclose(y_int_plain, yy_int_plain)
        # we get 0-dim array, so we can fetch the element directly
        np.testing.assert_equal(pred_int_plain.item(), True)
        np.testing.assert_equal(pred_float_plain.item(), True)

        # spu test
        y_int_spu, yy_int_spu, pred_int_spu = reveal_func_single_spu(x_int)
        y_float_spu, yy_float_spu, pred_float_spu = reveal_func_single_spu(x_float)

        # For FM64, fxp=18, SPU will compute log(x) with large error which confirm that yy is computed in plaintext.
        int_diff = np.abs(y_int_spu - yy_int_spu)
        self.assertTrue(np.any(int_diff > 1), f"max diff: {np.max(int_diff)}")

        float_diff = np.abs(y_float_spu - yy_float_spu)
        self.assertTrue(np.any(float_diff > 1), f"max diff: {np.max(float_diff)}")

        np.testing.assert_equal(pred_int_spu.item(), True)
        np.testing.assert_equal(pred_float_spu.item(), True)

        print("RevealTest test_reveal_single end\n")

    def test_reveal_list(self):
        print("RevealTest test_reveal_list start\n")
        x_int = jnp.array([1, 2, 3]) * (2**25)
        x_float = jnp.array([1.0, 2.0, 3.0]) * (2**25)
        reveal_func_list_spu = spsim.sim_jax(self.sim, reveal_func_list)

        # plain test
        xx_int, xx_float, reveal_x_log_int, reveal_x_log_float = reveal_func_list(
            x_int, x_float
        )
        np.testing.assert_allclose(xx_int, reveal_x_log_int)
        np.testing.assert_allclose(xx_float, reveal_x_log_float)
        np.testing.assert_allclose(reveal_x_log_int, reveal_x_log_float)

        # spu test
        xx_int_spu, xx_float_spu, reveal_x_log_int_spu, reveal_x_log_float_spu = (
            reveal_func_list_spu(x_int, x_float)
        )
        # For FM64, fxp=18, SPU will compute log(x) with large error which confirm that yy is computed in plaintext.
        int_diff = np.abs(xx_int_spu - reveal_x_log_int_spu)
        self.assertTrue(np.any(int_diff > 1), f"max diff: {np.max(int_diff)}")

        float_diff = np.abs(xx_float_spu - reveal_x_log_float_spu)
        self.assertTrue(np.any(float_diff > 1), f"max diff: {np.max(float_diff)}")

        print("RevealTest test_reveal_list end\n")

    def test_reveal_while_loop(self):
        print("RevealTest test_reveal_while_loop start\n")
        x = jnp.array([1, 2, 3, 4, 5])
        reveal_while_loop_spu = spsim.sim_jax(self.sim, reveal_while_loop)

        # plain test
        y_plain = reveal_while_loop(x)

        # spu test
        y_spu = reveal_while_loop_spu(x)

        # plaintext and spu should be the same
        np.testing.assert_allclose(y_plain, y_spu)

        print("RevealTest test_reveal_while_loop end\n")


if __name__ == "__main__":
    unittest.main()
