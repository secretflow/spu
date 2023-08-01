from abc import ABC, abstractmethod
import jax
from jax import vmap, jit
import jax.numpy as jnp
from jax.scipy.linalg import cho_solve, cho_factor
from ._lbfgs import *

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
        # 初始化参数
        n_samples, n_features = X.shape
        rng_key = jax.random.PRNGKey(0)
        if self.fit_intercept:
            X = jnp.hstack([jnp.ones((n_samples, 1)), X])  # 添加截距项
            if not self.coef:
                self.coef = jnp.full(
                (n_features + 1, ),
                0.5)  # coef初始化np.random.rand函数生成的是从0到1之间均匀分布的随机数。
        else:
            if not self.coef:
                self.coef = jnp.full(
                (n_features, ),
                0.5)  # coef初始化np.random.rand函数生成的是从0到1之间均匀分布的随机数。
        self.objective = lambda coef: self.loss_model(
            y, self.link.inverse(X @ coef)) + jnp.linalg.norm(coef)*self.l2_reg_strength/2
        self.objective_grad = jit(jax.grad(self.objective))
        self.hessian_fn = jit(jax.hessian(self.objective))
        return X

    @property
    def iteration(self):
        return self.max_iter


# 使用JAX定义NewtonCholeskySolver类
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
        Newton-Cholesky优化算法的求解器。

        Parameters:
        ----------
        loss_model : BaseLoss
            损失函数模型。
        link : BaseLink
            链接函数模型。
        l2_reg_strength : float, optional
            L2正则化强度，默认为1.0。
        max_iter : int, optional
            最大迭代次数，默认为100。
        n_threads : int or None, optional
            并行计算时的线程数。默认为None，表示不使用并行计算。
        fit_intercept : bool, optional
            是否拟合截距项，默认为True。
        verbose : int, optional
            是否输出详细信息，默认为0，不输出。
        coef : array-like, shape (n_features,), optional
            初始系数值，默认为None。

        """
        super().__init__(loss_model, link, max_iter, l2_reg_strength, n_threads, fit_intercept,
                         verbose, coef)

    def solve(self, X, y, sample_weight=None):
        """
        使用Newton-Cholesky算法求解广义线性回归模型的系数。

        Parameters:
        ----------
        X : array-like, shape (n_samples, n_features)
            输入特征矩阵。
        y : array-like, shape (n_samples,)
            目标变量。
        sample_weight : array-like, shape (n_samples,), optional
            样本权重，默认为None。

        Returns:
        -------
        coef : array-like, shape (n_features,)
            求解得到的系数。

        """
        X = super().solve(X, y)

        # 使用Cholesky分解解决线性系统
        def cho_solve_wrapper(a, b):
            return cho_solve(cho_factor(a), b[..., 0])

        # 执行Newton-Raphson步骤
        for i in range(self.max_iter):
            grad_value = self.objective_grad(self.coef)
            hessian_val = self.hessian_fn(self.coef)
            step = cho_solve_wrapper(hessian_val, grad_value[..., None])
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
        LBFGS优化算法求解广义线性回归的实现类。

        Parameters:
        ----------
        loss_model : object
            自定义的损失函数模型，需要继承自BaseLoss类。
        link : object
            自定义的链接函数模型，需要继承自BaseLink类。
        max_iter : int, optional (default=100)
            最大迭代次数。
        l2_reg_strength : float, optional (default=1.0)
            L2正则化项的强度。
        n_threads : int or None, optional (default=None)
            并行计算的线程数。None表示使用默认值。
        fit_intercept : bool, optional (default=True)
            是否拟合截距项。
        verbose : int, optional (default=0)
            控制输出信息的详细程度。0表示不输出详细信息，1表示输出部分信息。
        coef : array-like, shape (n_features,) or None, optional (default=None)
            初始化的模型系数。None表示使用默认初始化。

        Attributes:
        ----------
        maxcor : int
            BFGS算法的历史梯度和步长的存储数目。
        maxls : int
            BFGS算法的线搜索的最大迭代次数。
        gamma : float
            BFGS算法中的一种参数。

        """
        super().__init__(loss_model, link, max_iter, l2_reg_strength, n_threads, fit_intercept, verbose, coef)
        self.maxcor = 10
        self.maxls = 3
        self.gamma = 1

    def solve(self, X, y, sample_weight=None):
        """
        使用LBFGS优化算法求解广义线性回归。

        Parameters:
        ----------
        X : array-like, shape (n_samples, n_features)
            特征矩阵。
        y : array-like, shape (n_samples,)
            目标值。
        sample_weight : array-like, shape (n_samples,), optional (default=None)
            样本权重。

        Returns:
        -------
        coef : array-like, shape (n_features,)
            最优的模型系数。

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

        for j in range(curr_size):
            i = his_size - 1 - j
            a_i = self.rho_history[i] * (jnp.conj(self.s_history[i]) @ q)
            a_his = a_his.at[i].set(a_i)
            q = q - a_i * jnp.conj(self.y_history[i])

        q = self.gamma * q

        for j in range(curr_size):
            i = his_size - curr_size + j
            b_i = self.rho_history[i] * (self.y_history[i] @ q)
            q = q + (a_his[i] - b_i) * self.s_history[i]
        norm_q = jnp.linalg.norm(q)
        return q / norm_q

    def _line_search(self, p_k, f_k, g_k):
        a_k = 0.96**self.i

        # 通过拟牛顿法构建局部二次模型
        def quadratic_model(a):
            f_a = a * p_k @ g_k
            return abs(abs(f_a) - abs(f_k)) / max(abs(f_a), abs(f_k))

        alpha = 0.9
        # alpha = quadratic_model(a_k)
        a_k *= alpha ** self.maxls
        return a_k
