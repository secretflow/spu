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


import collections
import itertools
from os import getenv
from enum import Enum
from functools import partial

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
from absl.testing import parameterized
from jax._src import test_util as jtu

from spu.utils.simulation import sim_jax

if getenv("ENABLE_X64_TEST"):
    from jax import config

    config.update("jax_enable_x64", True)
    float_dtypes = [np.float64]
else:
    float_dtypes = [np.float32]

all_shapes = [(4,), (2, 3, 4)]
extra_large_shapes = [(100000000,)]

int32_dtypes = [np.int16, np.uint16, np.int32, np.uint32]
# Note: less by subtract+msb does not work for int64 when Ring=64
int64_dtypes = [np.int64, np.uint64]
int_dtypes = int32_dtypes
bool_dtypes = [np.bool_]
number_dtypes = float_dtypes + int_dtypes
all_dtypes = number_dtypes + bool_dtypes


class Status(Enum):
    Pass = 1
    """ Not supported by design """
    UnSupport = 2
    """ Not supported either by compiler or runtime.  """
    SysError = 3
    """ Result error """
    Failed = 4
    """ DO NOT generate report """
    PassNoGen = 5


OpRecord = collections.namedtuple(
    "OpRecord",
    ["name", "nargs", "dtypes", "shapes", "rng_factory", "status", "note", "kwargs"],
)


def REC(
    name, nargs, dtypes, shapes, rng_factory, status=Status.Pass, note="", kwargs=None
):
    return OpRecord(name, nargs, dtypes, shapes, rng_factory, status, note, kwargs)


# TODO: rand strategy is too weak.
def _rand_dtype(rand, shape, dtype):
    if np.issubdtype(dtype, np.integer):
        return rand.randint(100, size=shape).astype(dtype)
    elif np.issubdtype(dtype, np.floating):
        return rand.uniform(size=shape, low=0.1, high=10).astype(dtype)
    elif dtype == np.bool_:
        return rand.choice([True, False], size=shape).astype(dtype)
    else:
        raise Exception("unknown dtype={}", dtype)


def rand_default(rng):
    return partial(_rand_dtype, rng)


def rand_not_small_nonzero(rng):
    def post(x):
        x = np.where(x == 0, np.array(1, dtype=x.dtype), x)
        return x + np.where(x > 0, 0.1, -0.1)  # Make value at lease 0.1

    return partial(jtu._rand_dtype, rng.randn, scale=3, post=post)


JAX_ONE_TO_ONE_OP_RECORDS = [
    REC("abs", 1, all_dtypes, all_shapes, rand_default),
    REC("add", 2, all_dtypes, all_shapes, rand_default),
    REC("ceil", 1, number_dtypes, all_shapes, rand_default),
    REC("conj", 1, number_dtypes, all_shapes, rand_default),
    REC("equal", 2, all_dtypes, all_shapes, jtu.rand_some_equal),
    REC("exp", 1, number_dtypes, all_shapes, jtu.rand_small),
    REC("fabs", 1, float_dtypes, all_shapes, rand_default),
    REC("float_power", 2, float_dtypes, all_shapes, jtu.rand_small_positive),
    REC("floor", 1, number_dtypes, all_shapes, rand_default),
    REC("greater", 2, all_dtypes, all_shapes, jtu.rand_some_equal),
    REC("greater_equal", 2, all_dtypes, all_shapes, jtu.rand_some_equal),
    REC(
        "i0", 1, float_dtypes, all_shapes, rand_default, Status.Failed, "accuracy"
    ),  # FIXME: accuracy
    REC(
        "ldexp", 2, int_dtypes, all_shapes, rand_default, Status.Failed, "IEEE-754"
    ),  # FIXME: IEEE-754
    REC("less", 2, all_dtypes, all_shapes, jtu.rand_some_equal),
    REC("less_equal", 2, all_dtypes, all_shapes, jtu.rand_some_equal),
    REC("log", 1, number_dtypes, all_shapes, jtu.rand_positive),
    REC("logical_and", 2, all_dtypes, all_shapes, jtu.rand_bool),
    REC("logical_not", 1, all_dtypes, all_shapes, jtu.rand_bool),
    REC("logical_or", 2, all_dtypes, all_shapes, jtu.rand_bool),
    REC("logical_xor", 2, all_dtypes, all_shapes, jtu.rand_bool),
    REC("maximum", 2, all_dtypes, all_shapes, rand_default),
    REC("minimum", 2, all_dtypes, all_shapes, rand_default),
    REC("multiply", 2, all_dtypes, all_shapes, rand_default),
    REC("negative", 1, number_dtypes, all_shapes, rand_default),
    REC(
        "nextafter",
        2,
        float_dtypes,
        all_shapes,
        rand_default,
        Status.Failed,
        "IEEE-754",
    ),  # FIXME: IEEE-754
    REC("not_equal", 2, all_dtypes, all_shapes, jtu.rand_some_equal),
    REC("array_equal", 2, number_dtypes, all_shapes, jtu.rand_some_equal),
    REC("array_equiv", 2, number_dtypes, all_shapes, jtu.rand_some_equal),
    REC("reciprocal", 1, float_dtypes, all_shapes, rand_default),
    REC("subtract", 2, number_dtypes, all_shapes, rand_default),
    REC("signbit", 1, number_dtypes, all_shapes, rand_default),
    REC("trunc", 1, number_dtypes, all_shapes, rand_default),
    REC(
        "sin",
        1,
        number_dtypes,
        all_shapes,
        rand_default,
        Status.SysError,
        "stablehlo.sine",
    ),  # FIXME: stablehlo.sine
    REC(
        "cos",
        1,
        number_dtypes,
        all_shapes,
        rand_default,
        Status.SysError,
        "stablehlo.cosine",
    ),  # FIXME: stablehlo.cosine
    REC(
        "tan",
        1,
        number_dtypes,
        all_shapes,
        partial(jtu.rand_uniform, low=-1.5, high=1.5),
        Status.SysError,
        "stablehlo.sine",
    ),  # FIXME: stablehlo.sine
    REC(
        "sinh",
        1,
        number_dtypes,
        all_shapes,
        rand_default,
        Status.SysError,
        "stablehlo.cosine",
    ),  # FIXME: stablehlo.cosine
    REC("cosh", 1, number_dtypes, all_shapes, jtu.rand_small),
    REC("tanh", 1, number_dtypes, all_shapes, jtu.rand_default),
    REC(
        "arcsin",
        1,
        number_dtypes,
        all_shapes,
        jtu.rand_small,
        Status.SysError,
        "stablehlo.atan2",
    ),  # FIXME: stablehlo.atan2
    REC(
        "arccos",
        1,
        number_dtypes,
        all_shapes,
        jtu.rand_small,
        Status.SysError,
        "stablehlo.atan2",
    ),  # FIXME: stablehlo.atan2
    REC(
        "arctan",
        1,
        number_dtypes,
        all_shapes,
        jtu.rand_small,
        Status.SysError,
        "stablehlo.atan2",
    ),  # FIXME: stablehlo.atan2
    REC(
        "arctan2",
        2,
        float_dtypes,
        all_shapes,
        jtu.rand_small,
        Status.SysError,
        "stablehlo.atan2",
    ),  # FIXME: stablehlo.atan2
    REC("arcsinh", 1, number_dtypes, all_shapes, jtu.rand_small),
    REC(
        "arccosh", 1, number_dtypes, all_shapes, rand_default, Status.SysError, "nan"
    ),  # FIXME: nan
    REC(
        "arctanh",
        1,
        number_dtypes,
        all_shapes,
        jtu.rand_small,
        Status.SysError,
        "[-1, 1] nan",
    ),  # FIXME: [-1, 1] nan
]


COMPOUND_OP_RECORDS = [
    REC(
        "angle",
        1,
        number_dtypes,
        all_shapes,
        rand_default,
        Status.SysError,
        "stablehlo.atan2",
    ),  # FIXME: stablehlo.atan2
    REC(
        "angle",
        1,
        number_dtypes,
        all_shapes,
        rand_default,
        Status.SysError,
        "stablehlo.atan2",
        kwargs={'deg': True},
    ),  # FIXME: stablehlo.atan2
    REC("atleast_1d", 1, number_dtypes, all_shapes, rand_default),
    REC("atleast_2d", 1, number_dtypes, all_shapes, rand_default),
    REC("atleast_3d", 1, number_dtypes, all_shapes, rand_default),
    REC(
        "cbrt",
        1,
        number_dtypes,
        all_shapes,
        jtu.rand_some_inf,
        Status.SysError,
        "stablehlo.cbrt",
    ),  # FIXME: stablehlo.cbrt
    REC("conjugate", 1, number_dtypes, all_shapes, rand_default),
    REC("deg2rad", 1, float_dtypes, all_shapes, rand_default),
    REC(
        "divide", 2, number_dtypes, all_shapes, rand_not_small_nonzero
    ),  # FIXME: when denominator is very small, spu can have not small abs error
    REC(
        "divmod", 2, number_dtypes, all_shapes, rand_not_small_nonzero
    ),  # FIXME: when denominator is very small, spu can have not small abs error
    REC("exp2", 1, number_dtypes, all_shapes, jtu.rand_small_positive),
    REC("expm1", 1, number_dtypes, all_shapes, jtu.rand_small_positive),
    REC("fix", 1, number_dtypes, all_shapes, rand_default),
    REC(
        "floor_divide", 2, number_dtypes, all_shapes, rand_not_small_nonzero
    ),  # FIXME: when denominator is very small, spu can have not small abs error
    REC("fmin", 2, number_dtypes, all_shapes, rand_default),
    REC("fmax", 2, number_dtypes, all_shapes, rand_default),
    REC("fmod", 2, number_dtypes, all_shapes, rand_default),
    REC("heaviside", 2, number_dtypes, all_shapes, rand_default),
    REC("hypot", 2, number_dtypes, all_shapes, rand_default),
    REC("kron", 2, number_dtypes, all_shapes, rand_default),
    REC("outer", 2, number_dtypes, all_shapes, rand_default),
    REC("imag", 1, number_dtypes, all_shapes, jtu.rand_some_inf),
    REC("iscomplex", 1, number_dtypes, all_shapes, jtu.rand_some_inf),
    REC(
        "isfinite",
        1,
        float_dtypes,
        all_shapes,
        jtu.rand_some_inf_and_nan,
        Status.UnSupport,
    ),  # UnSupport
    REC(
        "isinf",
        1,
        float_dtypes,
        all_shapes,
        jtu.rand_some_inf_and_nan,
        Status.UnSupport,
    ),  # UnSupport
    REC(
        "isnan",
        1,
        float_dtypes,
        all_shapes,
        jtu.rand_some_inf_and_nan,
        Status.UnSupport,
    ),  # UnSupport
    REC("isneginf", 1, float_dtypes, all_shapes, jtu.rand_some_inf_and_nan),
    REC("isposinf", 1, float_dtypes, all_shapes, jtu.rand_some_inf_and_nan),
    REC("isreal", 1, number_dtypes, all_shapes, jtu.rand_some_inf),
    REC("isrealobj", 1, number_dtypes, all_shapes, jtu.rand_some_inf),
    REC("log2", 1, number_dtypes, all_shapes, jtu.rand_positive),
    REC("log10", 1, number_dtypes, all_shapes, jtu.rand_positive),
    REC("log1p", 1, number_dtypes, all_shapes, jtu.rand_positive),
    REC("log1p", 1, number_dtypes, all_shapes, jtu.rand_small_positive),
    REC("logaddexp", 2, float_dtypes, all_shapes, rand_default),
    REC("logaddexp2", 2, float_dtypes, all_shapes, rand_default),
    REC("polyval", 2, number_dtypes, all_shapes, rand_default),
    REC("positive", 1, number_dtypes, all_shapes, rand_default),
    REC("power", 2, number_dtypes, all_shapes, jtu.rand_small_positive),
    REC("rad2deg", 1, float_dtypes, all_shapes, rand_default),
    REC("ravel", 1, all_dtypes, all_shapes, rand_default),
    REC("real", 1, number_dtypes, all_shapes, rand_default),
    REC("remainder", 2, number_dtypes, all_shapes, jtu.rand_small),
    REC("mod", 2, number_dtypes, all_shapes, jtu.rand_nonzero),
    REC("modf", 1, number_dtypes, all_shapes, rand_default),
    REC(
        "rint",
        1,
        float_dtypes,
        all_shapes,
        rand_default,
        Status.SysError,
        "stablehlo.round_nearest_even",
    ),  # FIXME: stablehlo.round_nearest_even
    REC("sign", 1, number_dtypes, all_shapes, jtu.rand_default),
    REC(
        "copysign", 2, number_dtypes, all_shapes, rand_default, Status.SysError, "shift"
    ),  # FIXME: shift
    REC(
        "sinc",
        1,
        number_dtypes,
        all_shapes,
        rand_default,
        Status.SysError,
        "stablehlo.sine",
    ),  # FIXME: stablehlo.sine
    REC("square", 1, number_dtypes, all_shapes, rand_default),
    REC("sqrt", 1, number_dtypes, all_shapes, jtu.rand_positive),
    REC("transpose", 1, all_dtypes, all_shapes, rand_default),
    REC(
        "true_divide", 2, number_dtypes, all_shapes, rand_not_small_nonzero
    ),  # FIXME: when denominator is very small, spu can have not small abs error
    REC("ediff1d", 3, [np.int32], all_shapes, rand_default),
    REC("unwrap", 1, float_dtypes, all_shapes, rand_default),
    REC("isclose", 2, all_dtypes, all_shapes, jtu.rand_some_equal),
    REC(
        "gcd", 2, int_dtypes, all_shapes, rand_default, Status.SysError, "secret while"
    ),  # FIXME: secret while
    REC(
        "lcm", 2, int_dtypes, all_shapes, rand_default, Status.SysError, "secret while"
    ),  # FIXME: secret while
]

BITWISE_OP_RECORDS = [
    REC("bitwise_and", 2, int_dtypes, all_shapes, jtu.rand_bool),
    REC("bitwise_not", 1, int_dtypes, all_shapes, jtu.rand_bool),
    REC("invert", 1, int_dtypes, all_shapes, jtu.rand_bool),
    REC("bitwise_or", 2, int_dtypes, all_shapes, jtu.rand_bool),
    REC("bitwise_xor", 2, int_dtypes, all_shapes, jtu.rand_bool),
]

REDUCER_RECORDS = [
    REC("mean", 1, number_dtypes, all_shapes, rand_default),
    REC("mean", 1, float_dtypes, extra_large_shapes, jtu.rand_small, Status.PassNoGen),
    REC("prod", 1, all_dtypes, all_shapes, jtu.rand_small_positive),
    REC("sum", 1, int_dtypes, all_shapes, rand_default),
    REC(
        "nanmean", 1, float_dtypes, all_shapes, jtu.rand_some_nan, Status.UnSupport
    ),  # UnSupport
    REC(
        "nanprod", 1, all_dtypes, all_shapes, jtu.rand_some_nan, Status.UnSupport
    ),  # UnSupport
    REC(
        "nansum", 1, number_dtypes, all_shapes, jtu.rand_some_nan, Status.UnSupport
    ),  # UnSupport
]

# No tests
REDUCER_INITIAL_RECORDS = [
    REC("prod", 1, all_dtypes, all_shapes, jtu.rand_small_positive),
    REC("sum", 1, all_dtypes, all_shapes, rand_default),
    REC("max", 1, all_dtypes, all_shapes, rand_default),
    REC("min", 1, all_dtypes, all_shapes, rand_default),
    REC("nanprod", 1, float_dtypes, all_shapes, jtu.rand_small_positive),
    REC("nansum", 1, float_dtypes, all_shapes, rand_default),
    REC("nanmax", 1, float_dtypes, all_shapes, rand_default),
    REC("nanmin", 1, float_dtypes, all_shapes, rand_default),
]

SHIFT_RECORDS = [
    REC("left_shift", 1, int_dtypes, all_shapes, jtu.rand_int),
    REC("right_shift", 1, int_dtypes, all_shapes, jtu.rand_int),
]

# No tests
REDUCER_WHERE_NO_INITIAL_RECORDS = [
    REC("all", 1, bool_dtypes, all_shapes, jtu.rand_some_zero),
    REC("any", 1, bool_dtypes, all_shapes, jtu.rand_some_zero),
    REC("mean", 1, all_dtypes, all_shapes, rand_default),
    REC("var", 1, all_dtypes, all_shapes, rand_default),
    REC("std", 1, all_dtypes, all_shapes, rand_default),
    REC("nanmean", 1, float_dtypes, all_shapes, rand_default),
    REC("nanvar", 1, float_dtypes, all_shapes, rand_default),
    REC("nanstd", 1, float_dtypes, all_shapes, rand_default),
]

# No tests
REDUCER_NO_DTYPE_RECORDS = [
    REC("all", 1, all_dtypes, all_shapes, jtu.rand_some_zero),
    REC("any", 1, all_dtypes, all_shapes, jtu.rand_some_zero),
    REC("max", 1, all_dtypes, all_shapes, rand_default),
    REC("min", 1, all_dtypes, all_shapes, rand_default),
    REC("var", 1, all_dtypes, all_shapes, rand_default),
    REC("std", 1, all_dtypes, all_shapes, rand_default),
    REC("nanmax", 1, all_dtypes, all_shapes, jtu.rand_some_nan),
    REC("nanmin", 1, all_dtypes, all_shapes, jtu.rand_some_nan),
    REC("nanvar", 1, all_dtypes, all_shapes, jtu.rand_some_nan),
    REC("nanstd", 1, all_dtypes, all_shapes, jtu.rand_some_nan),
    REC("ptp", 1, number_dtypes, all_shapes, rand_default),
]

ARGMINMAX_RECORDS = [
    REC("argmin", 1, all_dtypes, all_shapes, jtu.rand_some_equal),
    REC("argmax", 1, all_dtypes, all_shapes, jtu.rand_some_equal),
    REC("nanargmin", 1, number_dtypes, all_shapes, jtu.rand_some_nan, Status.UnSupport),
    REC("nanargmax", 1, number_dtypes, all_shapes, jtu.rand_some_nan, Status.UnSupport),
]


SORT_RECORDS = [
    REC("sort", 1, number_dtypes, all_shapes, jtu.rand_some_equal),
    REC("median", 1, number_dtypes, all_shapes, jtu.rand_some_equal),
]


class JnpTests:
    class JnpTestBase(parameterized.TestCase):
        def setUp(self):
            self._sim = None
            # make test deterministic
            self._rng = np.random.RandomState(seed=0)

        @parameterized.parameters(
            (rec.name, rec.status, rec.nargs, dtype, shape, rec.rng_factory)
            for rec in itertools.chain(
                JAX_ONE_TO_ONE_OP_RECORDS,
                BITWISE_OP_RECORDS,
                COMPOUND_OP_RECORDS,
                SORT_RECORDS,
            )
            for (dtype, shape) in itertools.product(rec.dtypes, rec.shapes)
        )
        def test_ops(self, name, status, nargs, dtype, shape, rnd_factory):
            if status != Status.Pass:
                return
            jnp_op = getattr(jnp, name)
            spu_op = sim_jax(self._sim, jnp_op)
            rnd = rnd_factory(self._rng)
            args = [rnd(shape, dtype) for _ in range(nargs)]
            jnp_out = jnp_op(*args)
            spu_out = spu_op(*args)

            # print("inputs: ", args)
            # print(spu_op.pphlo)
            npt.assert_almost_equal(
                spu_out,
                jnp_out,
                decimal=1,
                err_msg="{} faild, spu = {}, jnp = {}, args = {}".format(
                    name, spu_out, jnp_out, args
                ),
            )

        @parameterized.parameters(
            (
                rec.name,
                rec.status,
                dtype,
                shape,
                rec.rng_factory,
                axis,
                keepdims,
            )
            for rec in REDUCER_RECORDS
            for (dtype, shape) in itertools.product(rec.dtypes, rec.shapes)
            for axis in list(range(len(shape))) + [None]
            for keepdims in [False, True]
        )
        def test_reducer(self, name, status, dtype, shape, rnd_factory, axis, keepdims):
            if status != Status.Pass or status != Status.PassNoGen:
                return
            jnp_op = getattr(jnp, name)
            jnp_fn = lambda x: jnp_op(x, axis=axis, keepdims=keepdims)
            spu_fn = sim_jax(self._sim, jnp_fn)
            rnd = rnd_factory(self._rng)
            args = [rnd(shape, dtype)]
            jnp_out = jnp_fn(*args)
            spu_out = spu_fn(*args)
            npt.assert_almost_equal(
                spu_out,
                jnp_out,
                decimal=1,
                err_msg="{} faild, spu = {}, jnp = {}".format(name, spu_out, jnp_out),
            )

        @parameterized.parameters(
            (
                rec.name,
                rec.status,
                dtype,
                shape,
                rec.rng_factory,
            )
            for rec in ARGMINMAX_RECORDS
            for (dtype, shape) in itertools.product(rec.dtypes, rec.shapes)
        )
        def test_argminmax(self, name, status, dtype, shape, rnd_factory):
            if status != Status.Pass:
                return
            jnp_op = getattr(jnp, name)
            jnp_fn = lambda x: jnp_op(x, axis=len(shape) - 1)
            spu_fn = sim_jax(self._sim, jnp_fn)
            rnd = rnd_factory(self._rng)
            args = [rnd(shape, dtype)]
            jnp_out = jnp_fn(*args)
            spu_out = spu_fn(*args)
            npt.assert_equal(
                spu_out,
                jnp_out,
                err_msg="{} faild.\nlhs = {},\nspu = {}\njnp = {}".format(
                    name, *args, spu_out, jnp_out
                ),
            )

        @parameterized.parameters(
            (
                rec.name,
                rec.status,
                dtype,
                shape,
                rec.rng_factory,
            )
            for rec in SHIFT_RECORDS
            for (dtype, shape) in itertools.product(rec.dtypes, rec.shapes)
        )
        def test_shift(self, name, status, dtype, shape, rnd_factory):
            if status != Status.Pass:
                return
            jnp_op = getattr(jnp, name)
            rhs_rnd = jtu.rand_int(self._rng, low=0, high=32)
            rhs = rhs_rnd(shape, dtype)
            jnp_fn = lambda x: jnp_op(x, rhs)
            spu_fn = sim_jax(self._sim, jnp_fn)
            rnd = rnd_factory(self._rng)
            args = [rnd(shape, dtype)]
            jnp_out = jnp_fn(*args)
            spu_out = spu_fn(*args)
            npt.assert_equal(
                spu_out,
                jnp_out,
                err_msg="{} faild.\nlhs = {}, rhs = {}\nspu = {}\njnp = {}".format(
                    name, *args, rhs, spu_out, jnp_out
                ),
            )

        def test_gather(self):
            jnp_fn = lambda x, indices: jnp.take(x, indices)
            spu_fn = sim_jax(self._sim, jnp_fn)
            x_rng = jtu.rand_int(self._rng, low=0, high=32)
            indices_rng = jtu.rand_int(self._rng, low=0, high=9)
            args = [x_rng((10,), np.int32), indices_rng((3,), np.int32)]
            jnp_out = jnp_fn(*args)
            spu_out = spu_fn(*args)
            npt.assert_equal(
                spu_out,
                jnp_out,
                err_msg="take faild.\nx = {}, indices = {}\nspu = {}\njnp = {}".format(
                    args[0], args[1], spu_out, jnp_out
                ),
            )
