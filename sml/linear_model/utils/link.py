# Copyright 2023 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from abc import ABC, abstractmethod

import jax.lax
import jax.numpy as jnp


def logit(p):
    # The logit function is defined as logit(p) = log(p/(1-p)).
    # use reciprocal can save a mul
    return -jnp.log(jax.lax.reciprocal(p) - 1)


def expit(x):
    # the expit function is defined as expit(x) = 1/(1+exp(-x))
    return 1 / (1 + jnp.exp(-x))


class BaseLink(ABC):
    """
    Define an abstract base class for link functions.

    We won't do interval check in all link function.

    """

    @abstractmethod
    def link(self, y_pred):
        pass

    @abstractmethod
    def inverse(self, raw_prediction):
        pass


class IdentityLink(BaseLink):
    """The identity link function g(x) = x."""

    def link(self, y_pred):
        # The link function returns the input as is (identity link)
        return y_pred

    def inverse(self, raw_prediction):
        # The inverse link function is the same as the link function for the identity link
        return raw_prediction


class ExpLink(BaseLink):
    """The exp link function g(x) = exp(x)."""

    def link(self, y_pred):
        # Apply exponential function to the input (exponential link)
        return jnp.exp(y_pred)

    def inverse(self, raw_prediction):
        # Apply the logarithm to reverse the effect of the exponential link
        # Note: for fxp, this compare is useless, it's only valid for floating-point
        return jnp.log(jnp.maximum(raw_prediction, 1e-12))


class LogLink(BaseLink):
    """The log link function g(x) = log(x), x \in (0, inf)"""

    def link(self, y_pred):
        # Apply the logarithm to the input (logarithmic link)
        # Note: for fxp, this compare is useless, it's only valid for floating-point
        return jnp.log(jnp.maximum(y_pred, 1e-12))

    def inverse(self, raw_prediction):
        # Apply the exponential function to reverse the effect of the logarithmic link
        return jnp.exp(raw_prediction)


class LogitLink(BaseLink):
    """The logit link function g(x)=logit(x), x \in (0,1)"""

    def link(self, y_pred):
        return logit(y_pred)

    def inverse(self, raw_prediction):
        return expit(raw_prediction)
