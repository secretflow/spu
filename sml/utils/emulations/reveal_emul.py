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

import jax
import jax.numpy as jnp
import numpy as np

import sml.utils.emulation as emulation
from sml.utils.utils import sml_reveal


def reveal_func_single(x):
    y = jnp.log(x)
    # reveal single value
    xx = sml_reveal(x)

    # x is 1-d array, so we fetch the first element
    pred = xx[0] > 0

    def true_branch(xx):
        return jnp.log(xx)

    def false_branch(xx):
        return jnp.log(-xx)

    # use jax.lax.cond to replace if-else
    yy = jax.lax.cond(pred, true_branch, false_branch, xx)

    return y, yy


def reveal_func_list(x, y):
    xx = jnp.log(x)
    yy = jnp.log(y)

    reveal_x, reveal_y = sml_reveal([x, y])

    reveal_x_log, reveal_y_log = jnp.log(reveal_x), jnp.log(reveal_y)

    return xx, yy, reveal_x_log, reveal_y_log


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


def emul_reveal(mode: emulation.Mode.MULTIPROCESS):
    print("start reveal emulation.")

    def _check_reveal_single(emulator):
        x_int = jnp.array([1, 2, 3]) * (2**25)
        x_float = jnp.array([1.0, 2.0, 3.0]) * (2**25)

        # plain test
        y_int_plain, yy_int_plain = reveal_func_single(x_int)
        y_float_plain, yy_float_plain = reveal_func_single(x_float)

        # all computed in plaintext, suppose no large diff
        np.testing.assert_allclose(y_int_plain, y_float_plain)
        np.testing.assert_allclose(yy_int_plain, yy_float_plain)
        np.testing.assert_allclose(y_int_plain, yy_int_plain)

        x_int_seal = emulator.seal(x_int)
        x_float_seal = emulator.seal(x_float)

        y_int_spu, yy_int_spu = emulator.run(reveal_func_single)(x_int_seal)
        y_float_spu, yy_float_spu = emulator.run(reveal_func_single)(x_float_seal)

        # For FM64, fxp=18, SPU will compute log(x) with large error which confirm that yy is computed in plaintext.
        int_diff = np.abs(y_int_spu - yy_int_spu)
        assert np.any(int_diff > 1), f"max diff: {np.max(int_diff)}"

        float_diff = np.abs(y_float_spu - yy_float_spu)
        assert np.any(float_diff > 1), f"max diff: {np.max(float_diff)}"

        print("reveal_func_single pass.")

    def _check_reveal_list(emulator):
        x_int = jnp.array([1, 2, 3]) * (2**25)
        x_float = jnp.array([1.0, 2.0, 3.0]) * (2**25)

        # plain test
        xx_int, xx_float, reveal_x_log_int, reveal_x_log_float = reveal_func_list(
            x_int, x_float
        )
        np.testing.assert_allclose(xx_int, reveal_x_log_int)
        np.testing.assert_allclose(xx_float, reveal_x_log_float)
        np.testing.assert_allclose(reveal_x_log_int, reveal_x_log_float)

        x_int_seal = emulator.seal(x_int)
        x_float_seal = emulator.seal(x_float)

        # spu test
        xx_int_spu, xx_float_spu, reveal_x_log_int_spu, reveal_x_log_float_spu = (
            emulator.run(reveal_func_list)(x_int_seal, x_float_seal)
        )
        # For FM64, fxp=18, SPU will compute log(x) with large error which confirm that yy is computed in plaintext.
        int_diff = np.abs(xx_int_spu - reveal_x_log_int_spu)
        assert np.any(int_diff > 1), f"max diff: {np.max(int_diff)}"

        float_diff = np.abs(xx_float_spu - reveal_x_log_float_spu)
        assert np.any(float_diff > 1), f"max diff: {np.max(float_diff)}"

        print("reveal_func_list pass.")

    def _check_reveal_while_loop(emulator):
        x = jnp.array([1, 2, 3, 4, 5])

        # plain test
        y_plain = reveal_while_loop(x)

        x_seal = emulator.seal(x)

        # spu test
        y_spu = emulator.run(reveal_while_loop)(x_seal)

        # plaintext and spu should be the same
        np.testing.assert_allclose(y_plain, y_spu)

        print("reveal_while_loop pass.")

    try:
        # ABY3, FM64, fxp=18
        conf_path = emulation.CLUSTER_ABY3_3PC
        emulator = emulation.Emulator(conf_path, mode, bandwidth=300, latency=20)
        emulator.up()

        _check_reveal_single(emulator)
        _check_reveal_list(emulator)
        _check_reveal_while_loop(emulator)

        print("reveal emulation pass.")

    finally:
        emulator.down()


if __name__ == "__main__":
    emul_reveal(emulation.Mode.MULTIPROCESS)
