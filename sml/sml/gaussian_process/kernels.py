# Copyright 2023 Ant Group Co., Ltd.
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

import jax
import jax.numpy as jnp


def sqeuclidean(u, v):
    uv_diff = u - v
    return jnp.dot(uv_diff, uv_diff)


def cdist(XA, XB, metric="sqeuclidean"):
    if metric != "sqeuclidean":
        raise NotImplementedError("Only sqeuclidean is supported now.")

    # todo: can get metric function from local dict.
    return jax.vmap(lambda xa: jax.vmap(lambda xb: sqeuclidean(xa, xb))(XB))(XA)


class RBF:
    """Radial basis function kernel (aka squared-exponential kernel).

    The RBF kernel is a stationary kernel. It is also known as the
    "squared exponential" kernel. It is parameterized by a length scale
    parameter :math:`l>0`, which is a scalar (isotropic variant
    of the kernel), The kernel is given by:

    .. math::
        k(x_i, x_j) = \\exp\\left(- \\frac{d(x_i, x_j)^2}{2l^2} \\right)

    This kernel is infinitely differentiable, which implies that GPs with this
    kernel as covariance function have mean square derivatives of all orders,
    and are thus very smooth.

    """

    def __init__(self, length_scale=1.0):
        self.length_scale = length_scale

    def tree_flatten(self):
        return ((self.length_scale,),)

    @classmethod
    def tree_unflatten(cls, _, children):
        return cls(*children)

    # def __call__(self, X, Y=None):
    #     # TODO: seems hard to do pdist with jax and vmap?
    #     # to utilize concurrency, do some additional computations for Y=None case
    #     if Y == None:
    #         Y = X

    #     K = jax.jit(cdist)(X, Y)
    #     K = jnp.exp(-K / (2 * self.length_scale * self.length_scale))

    #     return K

    def __call__(self, X, Y=None):
        if Y == None:
            Y = X
        K = jnp.zeros((X.shape[0], Y.shape[0]), dtype=jnp.float32)
        for i in range(X.shape[0]):
            for j in range(Y.shape[0]):
                K = K.at[i, j].set((jnp.sum((X[i] - Y[j]) ** 2)))
        return jnp.exp(-K / (2 * self.length_scale))

    def diag(self, X):
        return jnp.ones(X.shape[0])


jax.tree_util.register_pytree_node_class(RBF)
