import jax
import jax.numpy as jnp


class BaseLoss:
    def get_sampleweight(self, sample_weight):
        """
        Normalize sample_weight.
        """
        sample_weight /= jnp.sum(sample_weight)
        self.sample_weight = sample_weight


class HalfSquaredLoss(BaseLoss):
    def __call__(self, y_true, y_pred):
        """
        Calculate the half squared loss function.

        Parameters:
        ----------
        y_true : array-like
            True target data.
        y_pred : array-like
            Predicted target data.

        Returns:
        -------
        float
            Average half squared loss value.

        """
        y_t, y_p = y_true, y_pred
        w = self.sample_weight
        return jnp.sum(((y_t - y_p) ** 2 / 2) * w)


class HalfPoissonLoss(BaseLoss):
    def __call__(self, y_true, y_pred):
        """
        Calculate the half Poisson loss function.

        Parameters:
        ----------
        y_true : array-like
            True target data.
        y_pred : array-like
            Predicted target data.

        Returns:
        -------
        float
            Average half Poisson loss value.

        """
        y_t, y_p = y_true, y_pred
        w = self.sample_weight
        return jnp.sum((y_p - y_t * jnp.log(y_p)) * w)


class HalfGammaLoss(BaseLoss):
    def __call__(self, y_true, y_pred):
        """
        Calculate the half Gamma loss function.

        Parameters:
        ----------
        y_true : array-like
            True target data.
        y_pred : array-like
            Predicted target data.

        Returns:
        -------
        float
            Average half Gamma loss value.

        """
        y_t, y_p = y_true, y_pred
        w = self.sample_weight
        return jnp.sum(w * (jnp.log(y_p / y_t) + y_t / y_p - 1))


class HalfTweedieLoss(BaseLoss):
    def __init__(self, power=0.5):
        """
        Initialize HalfTweedieLoss class.

        Parameters:
        ----------
        power : float
            The power index of the Tweedie loss function.

        Returns:
        -------
        None

        """
        self.power = power

    def __call__(self, y_true, y_pred):
        """
        Calculate the half Tweedie loss function.

        Parameters:
        ----------
        y_true : array-like
            True target data.
        y_pred : array-like
            Predicted target data.

        Returns:
        -------
        float
            Average half Tweedie loss value.

        """
        p = self.power
        y_t, y_p = y_true, y_pred
        w = self.sample_weight
        return jnp.sum((y_p ** (2 - p) / (2 - p) - y_t * y_p ** (1 - p) / (1 - p)) * w)
