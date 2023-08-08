from abc import ABC, abstractmethod
import jax
from jax import vmap, jit
import jax.numpy as jnp
from jax.scipy.linalg import cho_solve, cho_factor

DEBUG = 0

class Solver(ABC):
    def __init__(self,
                 loss_model,
                 link,
                 max_iter=100,
                 l2_reg_strength=1,
                 n_threads=None,
                 fit_intercept=True,
                 verbose=0,
                 coef=None):
        self.loss_model = loss_model
        self.max_iter = max_iter
        self.n_threads = n_threads
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.link = link
        self.l2_reg_strength = l2_reg_strength
        self.coef = coef

    def predict(self, X):
        return self.link.inverse(X @ self.coef)

    @abstractmethod
    def solve(self, X, y, sample_weight=None):
        # Initialize parameters
        n_samples, n_features = X.shape
        rng_key = jax.random.PRNGKey(0)
        if self.fit_intercept:
            X = jnp.hstack([jnp.ones((n_samples, 1)), X])  # Add the intercept term
            if not self.coef:
                self.coef = jnp.full((n_features + 1, ), 0.5)  # Initialize coef using np.random.rand (uniform distribution between 0 and 1)
        else:
            if not self.coef:
                self.coef = jnp.full((n_features, ), 0.5)  # Initialize coef using np.random.rand (uniform distribution between 0 and 1)
        self.objective = lambda coef: self.loss_model(
            y, self.link.inverse(X @ coef)) + jnp.linalg.norm(coef) * self.l2_reg_strength / 2
        self.objective_grad = jit(jax.grad(self.objective))
        self.hessian_fn = jit(jax.hessian(self.objective))
        return X

    @property
    def iteration(self):
        return self.max_iter


# Define NewtonCholeskySolver class using JAX
class NewtonCholeskySolver(Solver):
    def __init__(self,
                 loss_model,
                 link,
                 l2_reg_strength=1.0,
                 max_iter=100,
                 n_threads=None,
                 fit_intercept=True,
                 verbose=0,
                 coef=None):
        """
        Solver for Newton-Cholesky optimization algorithm.

        Parameters:
        ----------
        loss_model : BaseLoss
            Loss function model.
        link : BaseLink
            Link function model.
        l2_reg_strength : float, optional
            L2 regularization strength. Default is 1.0.
        max_iter : int, optional
            Maximum number of iterations. Default is 100.
        n_threads : int or None, optional
            Number of threads for parallel computation. Default is None, meaning no parallel computation.
        fit_intercept : bool, optional
            Whether to fit the intercept term. Default is True.
        verbose : int, optional
            Verbosity level. Default is 0, no output.
        coef : array-like, shape (n_features,), optional
            Initial coefficient values. Default is None.

        """
        super().__init__(loss_model, link, max_iter, l2_reg_strength, n_threads, fit_intercept,
                         verbose, coef)

    def solve(self, X, y, sample_weight=None):
        """
        Solve generalized linear regression coefficients using Newton-Cholesky algorithm.

        Parameters:
        ----------
        X : array-like, shape (n_samples, n_features)
            Input feature matrix.
        y : array-like, shape (n_samples,)
            Target variable.
        sample_weight : array-like, shape (n_samples,), optional
            Sample weights. Default is None.

        Returns:
        -------
        coef : array-like, shape (n_features,)
            Solved coefficients.

        """
        X = super().solve(X, y)

        # Use Cholesky factorization to solve linear systems
        def cho_solve_wrapper(a, b):
            return cho_solve(cho_factor(a), b)

        # Perform Newton-Raphson steps
        for i in range(self.max_iter):
            grad_value = self.objective_grad(self.coef)
            hessian_val = self.hessian_fn(self.coef)
            step = cho_solve_wrapper(hessian_val, grad_value)
            self.coef = self.coef - step.flatten()

        return self.coef


class LBFGSSolver(Solver):
    def __init__(self,
                 loss_model,
                 link,
                 max_iter=100,
                 l2_reg_strength=1.0,
                 n_threads=None,
                 fit_intercept=True,
                 verbose=0,
                 coef=None):
        """
        Implementation of LBFGS optimization algorithm for generalized linear regression.

        Parameters:
        ----------
        loss_model : object
            Custom loss function model, should inherit from BaseLoss class.
        link : object
            Custom link function model, should inherit from BaseLink class.
        max_iter : int, optional (default=100)
            Maximum number of iterations.
        l2_reg_strength : float, optional (default=1.0)
            Strength of L2 regularization term.
        n_threads : int or None, optional (default=None)
            Number of threads for parallel computation. None means using default value.
        fit_intercept : bool, optional (default=True)
            Whether to fit the intercept term.
        verbose : int, optional (default=0)
            Controls the level of detailed output. 0 means no output, 1 means partial output.
        coef : array-like, shape (n_features,) or None, optional (default=None)
            Initialized model coefficients. None means using default initialization.

        Attributes:
        ----------
        maxcor : int
            The number of stored gradients and steps in BFGS algorithm.
        maxls : int
            The maximum number of line searches in BFGS algorithm.
        gamma : float
            A parameter in BFGS algorithm.

        """
        super().__init__(loss_model, link, max_iter, l2_reg_strength, n_threads, fit_intercept, verbose, coef)
        self.maxcor = 10
        self.maxls = 3
        self.gamma = 1

    def solve(self, X, y, sample_weight=None):
        """
        Solve generalized linear regression using LBFGS optimization algorithm.

        Parameters:
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix.
        y : array-like, shape (n_samples,)
            Target values.
        sample_weight : array-like, shape (n_samples,), optional (default=None)
            Sample weights.

        Returns:
        -------
        coef : array-like, shape (n_features,)
            Optimal model coefficients.

        """
        X = super().solve(X, y)

        d = len(self.coef)
        self.s_history = jnp.zeros((self.maxcor, d))
        self.y_history = jnp.zeros((self.maxcor, d))
        self.rho_history = jnp.zeros((self.maxcor, ))
        f_k, g_k = jax.value_and_grad(self.objective)(self.coef)

        for self.i in range(self.max_iter):
            p_k = self._two_loop_recursion(g_k)
            a_k = self._line_search(p_k, f_k, g_k)
            s_k = a_k * p_k
            self.coef += s_k
            f_k, g_new = jax.value_and_grad(self.objective)(self.coef)
            y_k = g_new - g_k
            g_k = g_new
            rho_k_inv = y_k @ s_k
            rho_k = jnp.reciprocal(rho_k_inv)
            self.gamma = rho_k_inv / (jnp.conj(y_k) @ y_k)
            jnp.roll(self.s_history, -1, axis=0).at[-1, :].set(s_k)
            jnp.roll(self.y_history, -1, axis=0).at[-1, :].set(y_k)
            jnp.roll(self.rho_history, -1, axis=0).at[-1].set(rho_k)

        return self.coef

    def _two_loop_recursion(self, g_k):
        his_size = len(self.rho_history)
        curr_size = his_size
        q = -jnp.conj(g_k)
        a_his = jnp.zeros((self.maxcor, ))

        for j in range(his_size):
            i = his_size - 1 - j
            a_i = self.rho_history[i] * (jnp.conj(self.s_history[i]) @ q)
            a_his = a_his.at[i].set(a_i)
            q = q - a_i * jnp.conj(self.y_history[i])

        q = self.gamma * q

        for j in range(his_size):
            b_i = self.rho_history[j] * (self.y_history[j] @ q)
            q = q + (a_his[j] - b_i) * self.s_history[j]
        return q 

    def _line_search(self, p_k, f_k, g_k):
        a_k = 0.96 ** self.i
        # Build a local quadratic model using quasi-Newton method
        def quadratic_model(a):
            f_a = a * p_k @ g_k
            return jnp.abs(jnp.abs(f_a) - jnp.abs(f_k)) / max(jnp.abs(f_a), jnp.abs(f_k))

        alpha = 0.9
        # alpha = quadratic_model(a_k)
        a_k *= alpha ** self.maxls
        return a_k
