import jax
import jax.numpy as jnp

class BaseLoss:
    def __init__(self, n_threads=1):
        self.n_threads = n_threads

    def __call__(self, y_true, y_pred, loss_single_sample):
        loss_batch = jax.vmap(loss_single_sample, in_axes=(0, 0), n_threads=self.n_threads)
        return jnp.mean(loss_batch(y_true, y_pred))

class HalfSquaredLoss(BaseLoss):
    def __call__(self, y_true, y_pred):
        return jnp.mean((y_true - y_pred) ** 2) / 2

class HalfPoissonLoss(BaseLoss):
    def __call__(self, y_true, y_pred):
        def loss_single_sample(y_t, y_p):
            return jnp.mean(y_p - y_t * jnp.log(y_p))

        return super(y_true,y_pred,loss_single_sample)

class HalfGammaLoss(BaseLoss):
    def __call__(self, y_true, y_pred):
        def loss_single_sample(y_t, y_p):
            return jnp.mean(jnp.log(y_p / y_t) + y_t / y_p - 1)

        return super(y_true,y_pred,loss_single_sample)

class HalfTweedieLoss(BaseLoss):
    def __init__(self, power, n_threads=1):
        super().__init__(n_threads)
        self.power = power

    def __call__(self, y_true, y_pred):
        def loss_single_sample(y_t, y_p):
            p = self.power
            return jnp.mean(y_p ** (2 - p) / (2 - p) - y_t * y_p ** (1 - p) / (1 - p))

        return super(y_true,y_pred,loss_single_sample)
