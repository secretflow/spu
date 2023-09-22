# Copyright 2023 Ant Group Co., Ltd.
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

import numpy as np
import jax.numpy as jnp


def update_w(X, W, H, H_sum, HHt, XHt, l1_reg_W, l2_reg_W):
    # H_sum is not used in 'frobenius'
    if XHt is None:
        XHt = jnp.dot(X, H.T)
    numerator = XHt
    if HHt is None:
        HHt = jnp.dot(H, H.T)
    denominator = jnp.dot(W, HHt)
    denominator = denominator + l1_reg_W + l2_reg_W * W
    numerator /= denominator
    delta_W = numerator
    W *= delta_W
    return W, H_sum, HHt, XHt


def update_h(X, W, H, l1_reg_H, l2_reg_H):
    numerator = jnp.dot(W.T, X)
    denominator = jnp.linalg.multi_dot([W.T, W, H])
    denominator = denominator + l1_reg_H + l2_reg_H * H
    delta_H = numerator
    delta_H /= denominator
    H *= delta_H
    return H


def init_nmf(X, n_components, random_matrixA, random_matrixB):
    avg = jnp.sqrt(np.mean(X) / n_components)
    W = avg * random_matrixB.astype(X.dtype)
    W_new = jnp.abs(W)
    H = avg * random_matrixA.astype(X.dtype)
    H_new = jnp.abs(H)
    return W_new, H_new


def compute_regularization(X, l1_ratio, alpha_W, alpha_H):
    n_samples, n_features = X.shape
    alpha_H = alpha_W
    l1_reg_W = n_features * alpha_W * l1_ratio
    l1_reg_H = n_samples * alpha_H * l1_ratio
    l2_reg_W = n_features * alpha_W * (1.0 - l1_ratio)
    l2_reg_H = n_samples * alpha_H * (1.0 - l1_ratio)
    return l1_reg_W, l1_reg_H, l2_reg_W, l2_reg_H


def _beta_divergence(X, W, H, beta, square_root=False):
    if beta == 'frobenius':
        vec = jnp.ravel(X - jnp.dot(W, H))
        res = jnp.dot(vec, vec) / 2.0
        if square_root:
            res = jnp.sqrt(res * 2)
    return res


class NMF:
    def __init__(
        self,
        n_components: int,
        max_iter: int = 200,
        l1_ratio: float = 0.0,
        alpha_W: float = 0.0,
        beta_loss: str = 'frobenius',
        alpha_H=None,
        random_matrixA=None,
        random_matrixB=None,
    ):
        """Compute Non-negative Matrix Factorization (NMF).

        Find two non-negative matrices (W, H) whose product approximates the non-
        negative matrix X.

        Parameters
        ----------
        n_components : int, default=None
            Number of components.
        max_iter : int, default=200
            Maximum number of iterations.
        l1_ratio : float, default=0.0
            The regularization mixing parameter, with 0 <= l1_ratio <= 1.
            For l1_ratio = 0 the penalty is an elementwise L2 penalty
            (aka Frobenius Norm).
            For l1_ratio = 1 it is an elementwise L1 penalty.
            For 0 < l1_ratio < 1, the penalty is a combination of L1 and L2.
        alpha_W : float, default=0.0
            Constant that multiplies the regularization terms of `W`. Set it to zero
            (default) to have no regularization on `W`.
        alpha_H : None
            Constant that multiplies the regularization terms of `H`. Set it takes
            the same value as `alpha_W`.
        beta_loss : str, default='frobenius'
            Beta divergence to be minimized, measuring the distance between X
            and the dot product WH. Only support 'frobenius' now.
        random_matrixA : None
            Non-negative random matrices, scaled with:
            sqrt(X.mean() / n_components), represents matrix H
        random_matrixB : None
            Non-negative random matrices, scaled with:
            sqrt(X.mean() / n_components), represents matrix W

        References
        ----------
        Lee, D. D., & Seung, H., S. (2001). Algorithms for Non-negative Matrix
        Factorization. Adv. Neural Inform. Process. Syst.. 13.
        """
        # parameter check.
        assert n_components > 0, f"n_components should >0"
        assert beta_loss == 'frobenius', f"beta_loss only support frobenius now"

        if alpha_H is None:
            alpha_H = alpha_W
        self._beta_loss = beta_loss
        self._n_components = n_components
        self._max_iter = max_iter
        self._l1_ratio = l1_ratio
        self._alpha_W = alpha_W
        self._alpha_H = alpha_H
        self._random_matrixA = random_matrixA
        self._random_matrixB = random_matrixB
        self._update_H = True
        self.components_ = None

    def fit(self, X):
        """Learn a NMF model for the data X.
        In the 'mu' solver, we use the multiplicative update algorithm to compute matrix W and H.
        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            Training data.
        Returns
        -------
        self : object
            Returns an instance of self.
        Notes
        -----
        (1) To prevent overflow error when using large data sets or get more accurate results,
        you can modify the definition of simulator as follows:
        config = spu_pb2.RuntimeConfig(
            protocol=spu_pb2.ProtocolKind.ABY3,
            field=spu_pb2.FieldType.FM128,
            fxp_fraction_bits=30,
            )
        sim_aby = spsim.Simulator(3, config)
        (2) The regularization parameter is deprecated in sklearn_1.0 and is removed in sklearn_1.2.
        Use 'alpha_W' and 'alpha_H' instead.
        (3) It's not very convenient to use jax_control_flow in spu right now, so we use 'max_iter' to control
        the iteration stop, and you can use 'reconstruction_err_' to learn about Frobenius norm of the matrix difference.
        If you want to get closer to the sklearn, need to keep the number of iterations consistent.
        (4) Only support 'random' init and 'mu' solver now.
        """
        self.fit_transform(X)
        return self

    def transform(self, X, transform_iter=40):
        assert self.components_ is not None, f"should fit before transform"
        self._update_H = False
        self._max_iter = transform_iter
        W = self.fit_transform(X)
        return W

    def fit_transform(self, X):
        # check random_matrix shape
        assert self._random_matrixA.shape == (
            self._n_components,
            X.shape[1],
        ), f"Expected random_matrixA to be ({self._n_components}, {X.shape[1]}) array, got {self._random_matrixA.shape}"
        assert self._random_matrixB.shape == (
            X.shape[0],
            self._n_components,
        ), f"Expected random_matrixA to be ({X.shape[0]}, {self._n_components}) array, got {self._random_matrixB.shape}"

        # init matrix W&H
        if self._update_H:
            W, H = init_nmf(
                X, self._n_components, self._random_matrixA, self._random_matrixB
            )
        else:
            avg = jnp.sqrt(X.mean() / self._n_components)
            W = jnp.full((X.shape[0], self._n_components), avg, dtype=X.dtype)
            H = self.components_

        # compute the regularization parameters
        l1_reg_W, l1_reg_H, l2_reg_W, l2_reg_H = compute_regularization(
            X, self._l1_ratio, self._alpha_W, self._alpha_H
        )
        H_sum, HHt, XHt = None, None, None

        # use multiplicative update solver
        for _ in range(self._max_iter):
            W, H_sum, HHt, XHt = update_w(X, W, H, H_sum, HHt, XHt, l1_reg_W, l2_reg_W)
            if self._update_H:
                H = update_h(X, W, H, l1_reg_H, l2_reg_H)
                H_sum, HHt, XHt = None, None, None

        # compute the reconstruction error
        if self._update_H:
            self.components_ = H
            self.reconstruction_err_ = _beta_divergence(
                X, W, H, self._beta_loss, square_root=True
            )

        return W

    def inverse_transform(self, X):
        assert self.components_ is not None, f"should fit before inverse_transform"
        return jnp.dot(X, self.components_)
