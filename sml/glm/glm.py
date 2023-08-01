import jax
from jax import random
import jax.numpy as jnp
from utils.solver import *
from utils.loss import *
from utils.link import *
from utils._lbfgs import _minimize_lbfgs
import warnings

DEBUG = 0


# 使用JAX定义_GeneralizedLinearRegressor类
class _GeneralizedLinearRegressor:
    def __init__(self,
                 fit_intercept=True,
                 alpha=0,
                 solver="newton-cholesky",
                 max_iter=20,
                 warm_start=False,
                 n_threads=2,
                 tol=None, # deprecated
                 verbose=0):
        self.l2_reg_strength = alpha
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.max_iter = max_iter
        self.warm_start = warm_start
        self.verbose = verbose
        self.n_threads = n_threads
        if tol:
            warnings.warn("spu不支持early stop.", category=DeprecationWarning,stacklevel=2)


    def fit(self, X, y, sample_weight=None):
        self._check_solver_support()
        self.loss_model = self._get_loss()
        self.link_model = self._get_link()
        if not self.warm_start or not hasattr(self, "coef_"):
            self.coef_ = None
        if self.solver == "lbfgs":
            warnings.warn("在SPU平台下无法准确实现lbfgs算法，只能近似实现", UserWarning)
            self._fit_lbfgs(X, y)
        elif self.solver == "newton-cholesky":
            self._fit_newton_cholesky(X, y)
        else:
            raise ValueError(f"Invalid solver={self.solver}.")

    def _get_loss(self):
        return HalfSquaredLoss(self.n_threads)  # 根据需要选择损失函数

    def _get_link(self):
        return IdentityLink()

    def _fit_newton_cholesky(self, X, y):
        # 使用NewtonCholeskySolver类实现Newton-Cholesky优化算法
        solver = NewtonCholeskySolver(loss_model=self.loss_model,
                                      l2_reg_strength=self.l2_reg_strength,
                                      max_iter=self.max_iter,
                                      verbose=self.verbose,
                                      link=self.link_model,
                                      coef=self.coef_)
        self.coef_ = solver.solve(X, y)
        # print(self.coef_)

    def _fit_lbfgs(self, X, y):
        # 使用LBFGSSolver类实现Newton-Cholesky优化算法
        solver = LBFGSSolver(loss_model=self.loss_model,
                             max_iter=self.max_iter,
                             l2_reg_strength=self.l2_reg_strength,
                             verbose=self.verbose,
                             link=self.link_model,
                             coef=self.coef_)
        self.coef_ = solver.solve(X, y)

    def predict(self, X):
        # 计算预测值
        if self.fit_intercept:
            X = jnp.hstack([jnp.ones((X.shape[0], 1)), X])  # 添加截距项
        y_pred = self.link_model.inverse(X @ self.coef_)
        return y_pred

    def score(self, X, y, sample_weight=None):
        """
        D^2是广义线性回归模型的评估指标。
        """

        # 计算模型的预测值
        prediction = self.predict(X)
        squared_error = lambda y_true, prediction: jnp.mean(
            (y_true - prediction)**2)
        # 计算模型的deviance
        deviance = squared_error(y_true=y, prediction=prediction)
        # 计算null deviance
        deviance_null = squared_error(y_true=y,
                                      prediction=jnp.tile(
                                          jnp.average(y), y.shape[0]))
        # 计算D^2
        d2 = 1 - (deviance) / (deviance_null)
        return d2

    def _check_solver_support(self):
        supported_solvers = ["lbfgs", "newton-cholesky"]  # 支持的优化算法列表
        if self.solver not in supported_solvers:
            raise ValueError(
                f"Invalid solver={self.solver}. Supported solvers are {supported_solvers}."
            )


class PoissonRegressor(_GeneralizedLinearRegressor):
    """具有泊松分布的广义线性模型，使用JAX实现。

    该回归器使用'log'链接函数。
    """
    def _get_loss(self):
        return HalfPoissonLoss(self.n_threads)

    def _get_link(self):
        return LogLink()
        # return IdentityLink()


class GammaRegressor(_GeneralizedLinearRegressor):
    def _get_loss(self):
        return HalfGammaLoss(self.n_threads)

    def _get_link(self):
        return LogLink()


class TweedieRegressor(_GeneralizedLinearRegressor):
    def __init__(
        self,
        power=0.5,
    ):
        super().__init__()
        if power > 0: power = -power
        if power > 1: power = 1 / power
        elif power == 1: power = 0.5
        self.power = power

    def _get_loss(self):
        return HalfTweedieLoss(self.power, self.n_threads)

    def _get_link(self):
        if self.power > 0:
            return LogLink()
        else:
            return IdentityLink()