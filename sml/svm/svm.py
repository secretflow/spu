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

import jax.numpy as jnp

from sml.svm.smo import SMO


class SVM:
    """
    Parameters
    ----------
    kernel : str, default="rbf"
        The kernel function used in the svm algorithm, maps samples
        to a higher dimensional feature space.

    C : float, default=1.0
        Regularization parameter. The strength of the regularization
        is inversely proportional to C. Must be strictly positive. The penalty is a squared l2 penalty.

    gamma : {'scale', 'auto'}, default="scale"
        Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
        if gamma='scale' (default) is passed then it uses 1 / (n_features * X.var()) as value of gamma,
        if 'auto', uses 1 / n_features

    max_iter : int, default=300
        Maximum number of iterations of the svm algorithm for a
        single run.

    tol : float, default=1e-3
        Acceptable error to consider the two to be equal.
    """

    def __init__(self, kernel="rbf", C=1.0, gamma='scale', max_iter=300, tol=1e-3):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.max_iter = max_iter
        self.tol = tol
        self.n_features = None

        self.alpha = None
        self.alpha_y = None
        self.b = None

        self.X = None

    def cal_kernel(self, x, x_):
        """Calculate kernel."""
        assert self.gamma in {'scale', 'auto'}, "Gamma only support 'scale' and 'auto'"
        gamma = {
            'scale': 1 / (self.n_features * x.var()),
            'auto': 1 / self.n_features,
        }[self.gamma]

        assert self.kernel == "rbf", "Kernel function only support 'rbf'"
        kernel_res = jnp.exp(
            -gamma
            * (
                (x**2).sum(1, keepdims=True)
                + (x_**2).sum(1)
                - 2 * jnp.matmul(x, x_.T)
            )
        )

        return kernel_res

    def cal_Q(self, x, y):
        """Calculate Q."""
        kernel_res = self.cal_kernel(x, x)
        Q = y.reshape(-1, 1) * y * kernel_res
        return Q

    def fit(self, X, y):
        """Fit SVM.

        Using the Sequential Minimal Optimization(SMO) algorithm to solve the Quadratic programming problem in
        the SVM, which decomposes the large optimization problem to several small optimization problems. Firstly,
        the SMO algorithm selects alpha_i and alpha_j by 'smo.working_set_select_i()' and 'smo.working_set_select_j'.
        Secondly, the SMO algorithm update the parameter by 'smo.update()'. Last, calculate the bias.

        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            Input data.

        y : {array-like}, shape (n_samples)
            Lable of the input data.

        """

        l, self.n_features = X.shape
        p = -jnp.ones(l)
        smo = SMO(l, self.C, self.tol)
        Q = self.cal_Q(X, y)
        alpha = 0.0 * y
        neg_y_grad = -p * y
        for _ in range(self.max_iter):
            i = smo.working_set_select_i(alpha, y, neg_y_grad)
            j = smo.working_set_select_j(i, alpha, y, neg_y_grad, Q)
            neg_y_grad, alpha = smo.update(i, j, Q, y, alpha, neg_y_grad)

        self.alpha = alpha
        self.b = smo.cal_b(alpha, neg_y_grad, y)
        self.alpha_y = self.alpha * y

        self.X = X

    def predict(self, x):
        """Result estimates.

        Calculate the classification result of the input data.

        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            Input data.

        y : {array-like}, shape (n_samples)
            Lable of the input data.

        x : {array-like}, shape (n_samples, n_features)
            Input data for prediction.

        Returns
        -------
        ndarray of shape (n_samples)
            Returns the classification result of the input data for prediction.
        """

        pred = (
            jnp.matmul(
                self.alpha_y,
                self.cal_kernel(self.X, x),
            )
            + self.b
        )
        return (pred >= 0).astype(int)
