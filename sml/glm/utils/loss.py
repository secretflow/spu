import jax
import jax.numpy as jnp

class BaseLoss:
    def __init__(self, n_threads=1):
        """
        初始化BaseLoss类对象。

        Parameters:
        ----------
        n_threads : int, optional (default=1)
            线程数，用于并行计算损失函数。

        Returns:
        -------
        None

        """
        self.n_threads = n_threads

    def __call__(self, y_true, y_pred, loss_single_sample):
        """
        计算损失函数的平均值。

        Parameters:
        ----------
        y_true : array-like
            真实目标数据。
        y_pred : array-like
            预测目标数据。
        loss_single_sample : function
            用于计算单个样本损失的函数。

        Returns:
        -------
        float
            平均损失值。

        """
        loss_batch = jax.vmap(loss_single_sample, in_axes=(0, 0), n_threads=self.n_threads)
        return jnp.mean(loss_batch(y_true, y_pred))

class HalfSquaredLoss(BaseLoss):
    def __call__(self, y_true, y_pred):
        """
        计算半二次平方损失函数。

        Parameters:
        ----------
        y_true : array-like
            真实目标数据。
        y_pred : array-like
            预测目标数据。

        Returns:
        -------
        float
            平均半二次平方损失值。

        """
        return jnp.mean((y_true - y_pred) ** 2) / 2

class HalfPoissonLoss(BaseLoss):
    def __call__(self, y_true, y_pred):
        """
        计算半泊松损失函数。

        Parameters:
        ----------
        y_true : array-like
            真实目标数据。
        y_pred : array-like
            预测目标数据。

        Returns:
        -------
        float
            平均半泊松损失值。

        """
        def loss_single_sample(y_t, y_p):
            return jnp.mean(y_p - y_t * jnp.log(y_p))

        return super(y_true, y_pred, loss_single_sample)

class HalfGammaLoss(BaseLoss):
    def __call__(self, y_true, y_pred):
        """
        计算半伽马损失函数。

        Parameters:
        ----------
        y_true : array-like
            真实目标数据。
        y_pred : array-like
            预测目标数据。

        Returns:
        -------
        float
            平均半伽马损失值。

        """
        def loss_single_sample(y_t, y_p):
            return jnp.mean(jnp.log(y_p / y_t) + y_t / y_p - 1)

        return super(y_true, y_pred, loss_single_sample)

class HalfTweedieLoss(BaseLoss):
    def __init__(self, power, n_threads=1):
        """
        初始化HalfTweedieLoss类对象。

        Parameters:
        ----------
        power : float
            Tweedie损失函数的幂指数。
        n_threads : int, optional (default=1)
            线程数，用于并行计算损失函数。

        Returns:
        -------
        None

        """
        super().__init__(n_threads)
        self.power = power

    def __call__(self, y_true, y_pred):
        """
        计算半Tweedie损失函数。

        Parameters:
        ----------
        y_true : array-like
            真实目标数据。
        y_pred : array-like
            预测目标数据。

        Returns:
        -------
        float
            平均半Tweedie损失值。

        """
        def loss_single_sample(y_t, y_p):
            p = self.power
            return jnp.mean(y_p ** (2 - p) / (2 - p) - y_t * y_p ** (1 - p) / (1 - p))

        return super(y_true, y_pred, loss_single_sample)
