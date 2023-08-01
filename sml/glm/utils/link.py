# Importing necessary libraries
from abc import ABC, abstractmethod
from dataclasses import dataclass
import jax.numpy as jnp

# Define a dataclass to represent an interval
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
        # Check if elements in x are greater than or equal to the lower bound
        if self.low_inclusive:
            low = jnp.greater_equal(x, self.low)
        else:
            low = jnp.greater(x, self.low)

        # If any element is outside the lower bound, return False
        if not jnp.all(low):
            return False

        # Check if elements in x are less than or equal to the upper bound
        if self.high_inclusive:
            high = jnp.less_equal(x, self.high)
        else:
            high = jnp.less(x, self.high)

        # Note: jnp.all returns numpy.bool_
        # Return True only if all elements are within the interval
        return bool(jnp.all(high))


# Define an abstract base class for link functions
class BaseLink(ABC):
    # Default interval for predicted y values is (-inf, inf)
    interval_y_pred = Interval(-jnp.inf, jnp.inf, False, False)

    @abstractmethod
    def link(self, y_pred, out=None):
        pass

    @abstractmethod
    def inverse(self, raw_prediction, out=None):
        pass


# Implementation of the IdentityLink link function g(x) = x
class IdentityLink(BaseLink):
    """The identity link function g(x) = x."""

    def link(self, y_pred, out=None):
        # The link function returns the input as is (identity link)
        return y_pred

    def inverse(self, raw_prediction, out=None):
        # The inverse link function is the same as the link function for the identity link
        return raw_prediction


# Implementation of the ExpLink link function g(x) = exp(x)
class ExpLink(BaseLink):
    """The exp link function g(x) = exp(x)."""

    def link(self, y_pred, out=None):
        # Apply exponential function to the input (exponential link)
        return jnp.exp(y_pred)

    def inverse(self, raw_prediction):
        # Apply the logarithm to reverse the effect of the exponential link
        # Note: jnp.maximum is used to avoid taking the logarithm of values close to zero
        return jnp.log(jnp.maximum(raw_prediction, 1e-12))


# Implementation of the LogLink link function g(x) = log(x)
class LogLink(BaseLink):
    """The log link function g(x) = log(x)."""

    # Define the interval for predicted y values to be (0, inf)
    interval_y_pred = Interval(0, jnp.inf, False, False)

    def link(self, y_pred, out=None):
        # Apply the logarithm to the input (logarithmic link)
        # Note: jnp.maximum is used to avoid taking the logarithm of values close to zero
        return jnp.log(jnp.maximum(y_pred, 1e-12))

    def inverse(self, raw_prediction, out=None):
        # Apply the exponential function to reverse the effect of the logarithmic link
        return jnp.exp(raw_prediction)
