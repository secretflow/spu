from abc import ABC, abstractmethod
from dataclasses import dataclass
import jax.numpy as jnp

@dataclass
class Interval:
    low: float
    high: float
    low_inclusive: bool
    high_inclusive: bool

    def __post_init__(self):
        """Check that low <= high"""
        if self.low > self.high:
            raise ValueError(
                f"One must have low <= high; got low={self.low}, high={self.high}."
            )

    def includes(self, x):
        """Test whether all values of x are in interval range.

        Parameters
        ----------
        x : ndarray
            Array whose elements are tested to be in interval range.

        Returns
        -------
        result : bool
        """
        if self.low_inclusive:
            low = jnp.greater_equal(x, self.low)
        else:
            low = jnp.greater(x, self.low)

        if not jnp.all(low):
            return False

        if self.high_inclusive:
            high = jnp.less_equal(x, self.high)
        else:
            high = jnp.less(x, self.high)

        # Note: jnp.all returns numpy.bool_
        return bool(jnp.all(high))

class BaseLink(ABC):
    interval_y_pred = Interval(-jnp.inf, jnp.inf, False, False)

    @abstractmethod
    def link(self, y_pred, out=None):
        pass

    @abstractmethod
    def inverse(self, raw_prediction, out=None):
        pass


class IdentityLink(BaseLink):
    """The identity link function g(x)=x."""

    def link(self,  y_pred, out=None):
        return y_pred

    inverse = link


class ExpLink(BaseLink):
    """The exp link function g(x)=exp(x)."""
    def link(self,  y_pred, out=None):
        return jnp.exp(y_pred)

    def inverse(self, raw_prediction):
        return jnp.log(jnp.maximum(raw_prediction,1e-12))


class LogLink(BaseLink):
    """The log link function g(x)=log(x)."""
    interval_y_pred = Interval(0, jnp.inf, False, False)
    def link(self, y_pred, out=None):
        return jnp.log(jnp.maximum(y_pred,1e-12))

    def inverse(self, raw_prediction, out=None):
        return jnp.exp(raw_prediction)