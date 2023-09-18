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

import jax.numpy as jnp

from sml.linear_model.utils.link import IdentityLink, LogLink


# TODO: can move this to root utils, if other module needs loss function.
class BaseLoss:
    """
    Base class for a loss function of 1-dimensional targets.

    For GLM, Note that y_pred = link.inverse(raw_prediction).

    Parameters
    ----------
    link: link function of this loss
    sample_weight: weight of each sample
    n_classes: The number of classes for classification, else None

    """

    def __init__(self, link=None, sample_weight=None, n_classes=None):
        self.link = link
        self.sample_weight = sample_weight
        self.n_classes = n_classes  # leave for support multinomial loss

    def set_sample_weight(self, sample_weight):
        sample_weight /= jnp.sum(sample_weight)
        self.sample_weight = sample_weight


class HalfSquaredLoss(BaseLoss):
    def __init__(self, link=IdentityLink(), sample_weight=None, n_classes=None):
        super().__init__(link=link, sample_weight=sample_weight, n_classes=n_classes)

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
        return jnp.sum(((y_true - y_pred) ** 2 / 2) * self.sample_weight)


class HalfPoissonLoss(BaseLoss):
    def __init__(self, link=LogLink(), sample_weight=None, n_classes=None):
        super().__init__(link=link, sample_weight=sample_weight, n_classes=n_classes)

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
        return jnp.sum((y_pred - y_true * jnp.log(y_pred)) * self.sample_weight)


class HalfGammaLoss(BaseLoss):
    def __init__(self, link=LogLink(), sample_weight=None, n_classes=None):
        super().__init__(link=link, sample_weight=sample_weight, n_classes=n_classes)

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
        return jnp.sum(
            self.sample_weight * (jnp.log(y_pred / y_true) + y_true / y_pred - 1)
        )


class HalfTweedieLoss(BaseLoss):
    def __init__(self, power=1.5, link=LogLink(), sample_weight=None, n_classes=None):
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
        super().__init__(link=link, sample_weight=sample_weight, n_classes=n_classes)
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
        return jnp.sum(
            (y_pred ** (2 - p) / (2 - p) - y_true * y_pred ** (1 - p) / (1 - p))
            * self.sample_weight
        )
