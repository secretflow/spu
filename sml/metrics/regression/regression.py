# Copyright 2023 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specifi

from sklearn import metrics
import jax.numpy as jnp

import spu.spu_pb2 as spu_pb2
import spu.utils.simulation as spsim


def _mean_tweedie_deviance(y_true, y_pred, sample_weight, power):
    p = power
    if p < 0:
        # 'Extreme stable', y any real number, y_pred > 0
        temp = 1 - p
        temp_power = jnp.power(y_pred, temp)
        dev = 2 * (
            jnp.power(jnp.maximum(y_true, 0), 1 + temp) / (temp * (temp + 1))
            - y_true * temp_power / temp
            + temp_power * y_pred / (temp + 1)
        )
    elif p == 0:
        # Normal distribution, y and y_pred any real number
        dev = (y_true - y_pred) ** 2
    elif p == 1:
        # Poisson distribution
        dev = 2 * (y_true * jnp.log((y_true / y_pred)) - y_true + y_pred)
    elif p == 2:
        # Gamma distribution
        dev = 2 * (jnp.log(y_pred / y_true) + y_true / y_pred - 1)
    else:
        temp = 1 - p
        temp_power = jnp.power(y_pred, temp)
        dev = 2 * (
            jnp.power(y_true, 1 + temp) / (temp * (temp + 1))
            - y_true * temp_power / temp
            + temp_power * y_pred / (temp + 1)
        )
    return jnp.average(dev, weights=sample_weight)


def _d2_tweedie_score(y_true, y_pred, sample_weight, power, re_use=None):
    p = power
    re_value = []
    if p < 0:
        # 'Extreme stable', y any real number, y_pred > 0
        temp = 1 - p
        if re_use is None:
            re_power = jnp.power(jnp.maximum(y_true, 0), 1 + temp) / (temp * (temp + 1))
            re_div = y_true / temp
            re_value.append(re_power)
            re_value.append(re_div)
        else:
            re_power = re_use[0]
            re_div = re_use[1]
        temp_power = jnp.power(y_pred, temp)
        dev = 2 * (re_power - temp_power * re_div + temp_power * y_pred / (temp + 1))
    elif p == 0:
        # Normal distribution, y and y_pred any real number
        if re_use is None:
            re_square = y_true**2
            re_value.append(re_square)
        else:
            re_square = re_use[0]
        dev = re_square + y_pred**2 - 2 * y_true * y_pred
    elif p == 1:
        # Poisson distribution
        if re_use is None:
            re_log = y_true * jnp.log(y_true) - y_true
            re_value.append(re_log)
        else:
            re_log = re_use[0]
        dev = 2 * (re_log - y_true * jnp.log(y_pred) + y_pred)
    elif p == 2:
        if re_use is None:
            re_log = jnp.log(y_true)
            re_value.append(re_log)
        else:
            re_log = re_use[0]
        # Gamma distribution
        dev = 2 * (jnp.log(y_pred) - re_log + y_true / y_pred - 1)
    else:
        temp = 1 - p
        if re_use is None:
            re_power = jnp.power(y_true, 1 + temp) / (temp * (temp + 1))
            re_div = y_true / temp
            re_value.append(re_power)
            re_value.append(re_div)
        else:
            re_power = re_use[0]
            re_div = re_use[1]
        temp_power = jnp.power(y_pred, temp)
        dev = 2 * (re_power - temp_power * re_div + temp_power * y_pred / (temp + 1))
    if sample_weight is None:
        return jnp.sum(dev), re_value
    else:
        return jnp.average(dev, weights=sample_weight), re_value


def d2_tweedie_score(y_true, y_pred, sample_weight=None, power=0):
    numerator, re_use = _d2_tweedie_score(
        y_true, y_pred, sample_weight=sample_weight, power=power
    )
    y_avg = jnp.average(y_true, weights=sample_weight)
    denominator, _ = _d2_tweedie_score(
        y_true, y_avg, sample_weight=sample_weight, power=power, re_use=re_use
    )
    return 1 - numerator / denominator


def explained_variance_score(
    y_true,
    y_pred,
    sample_weight=None,
    multioutput="uniform_average",
):
    y_diff_avg = jnp.average(y_true - y_pred, weights=sample_weight, axis=0)
    numerator = jnp.average(
        (y_true - y_pred - y_diff_avg) ** 2, weights=sample_weight, axis=0
    )

    y_true_avg = jnp.average(y_true, weights=sample_weight, axis=0)
    denominator = jnp.average((y_true - y_true_avg) ** 2, weights=sample_weight, axis=0)
    output_scores = 1 - (numerator / denominator)

    if isinstance(multioutput, str):
        if multioutput == "raw_values":
            # return scores individually
            return output_scores
        elif multioutput == "uniform_average":
            # Passing None as weights to np.average results is uniform mean
            avg_weights = None
        elif multioutput == "variance_weighted":
            avg_weights = denominator
    else:
        avg_weights = multioutput

    return jnp.average(output_scores, weights=avg_weights)


def mean_squared_error(
    y_true, y_pred, sample_weight=None, multioutput="uniform_average", squared=True
):
    output_errors = jnp.average((y_true - y_pred) ** 2, axis=0, weights=sample_weight)

    if not squared:
        output_errors = jnp.sqrt(output_errors)
    if isinstance(multioutput, str):
        if multioutput == "raw_values":
            return output_errors
        elif multioutput == "uniform_average":
            # pass None as weights to np.average: uniform mean
            multioutput = None

    return jnp.average(output_errors, weights=multioutput)


def mean_poisson_deviance(y_true, y_pred, sample_weight=None):
    return _mean_tweedie_deviance(y_true, y_pred, sample_weight=sample_weight, power=1)


def mean_gamma_deviance(y_true, y_pred, sample_weight=None):
    return _mean_tweedie_deviance(y_true, y_pred, sample_weight=sample_weight, power=2)
