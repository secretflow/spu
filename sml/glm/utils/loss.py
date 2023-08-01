import jax
import jax.numpy as jnp

class BaseLoss:
    def get_sampleweight(self, sample_weight):
        """
        Normalize sample_weight.
        """
        # sample_weight /= jnp.sum(sample_weight)
        self.sample_weight = sample_weight

    def __call__(self, y_true, y_pred, loss_single_sample):
        """
        Calculate the average loss function.

        Parameters:
        ----------
        y_true : array-like
            True target data.
        y_pred : array-like
            Predicted target data.
        loss_single_sample : function
            Function to compute loss for a single sample.

        Returns:
        -------
        float
            Average loss value.

        """
        weighted_loss_batch = jax.vmap(lambda y_t, y_p, w: w * loss_single_sample(y_t, y_p),
                                       in_axes=(0, 0, 0),
                                       out_axes=0)
        return jnp.sum(weighted_loss_batch(y_true, y_pred, self.sample_weight))


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
        def loss_single_sample(y_t, y_p):
            return jnp.mean((y_t - y_p) ** 2) / 2

        return super().__call__(y_true, y_pred, loss_single_sample)


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
        def loss_single_sample(y_t, y_p):
            return jnp.mean(y_p - y_t * jnp.log(y_p))

        return super().__call__(y_true, y_pred, loss_single_sample)

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
        def loss_single_sample(y_t, y_p):
            return jnp.mean(jnp.log(y_p / y_t) + y_t / y_p - 1)

        return super().__call__(y_true, y_pred, loss_single_sample)

class HalfTweedieLoss(BaseLoss):
    def __init__(self, power):
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
        def loss_single_sample(y_t, y_p):
            p = self.power
            return jnp.mean(y_p ** (2 - p) / (2 - p) - y_t * y_p ** (1 - p) / (1 - p))

        return super().__call__(y_true, y_pred, loss_single_sample)
