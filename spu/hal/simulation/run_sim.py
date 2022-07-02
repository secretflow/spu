# Copyright 2021 Ant Group Co., Ltd.
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
# See the License for the specific language governing permissions and
# limitations under the License.


import spu.hal.simulation.fxp_acc as fxp_acc

import argparse
import random
import numpy as np
import numbers
import math
from pytablewriter import MarkdownTableWriter

DATASET_SIZE = 10000


def create_normal_dataset():
    ds = list(np.random.normal(0, 10, DATASET_SIZE))
    return ds


def create_uniform_dataset(low, high):
    ds = list(np.random.uniform(low, high, DATASET_SIZE))
    return ds


def check(actual, expected):
    total = len(actual)
    valid = 0
    success = 0
    failure = 0
    for i in range(total):
        if not isinstance(actual[i], numbers.Number) or not isinstance(
            expected[i], numbers.Number
        ):
            continue
        valid += 1

        if np.isclose(actual[i], expected[i], rtol=1e-2, atol=0.0001):
            success += 1
        else:
            failure += 1
    return {
        'total': total,
        'valid': valid,
        'success': success,
        'failure': failure,
        'rate': success / (success + failure + 0.0),
    }


def exp_sim():
    res = []

    # Testcase 1
    x = create_uniform_dataset(-10, 20)
    actual = fxp_acc.exp(x)
    expected = list(np.exp(np.array(x)))
    res.append(
        {
            "id": "Testcase 1",
            "pass_rate": check(actual, expected)['rate'],
            "description": "uniform(-10, 20)",
        }
    )

    return res


def reciprocal_sim():
    res = []

    # Testcase 1
    x = create_uniform_dataset(0, 1e3)
    actual = fxp_acc.reciprocal(x)
    expected = list(np.reciprocal(np.array(x)))
    res.append(
        {
            "id": "Testcase 1",
            "pass_rate": check(actual, expected)['rate'],
            "description": "uniform(0, 1e3)",
        }
    )

    # Testcase 2
    x = create_uniform_dataset(1e3, 1e6)
    actual = fxp_acc.reciprocal(x)
    expected = list(np.reciprocal(np.array(x)))
    res.append(
        {
            "id": "Testcase 2",
            "pass_rate": check(actual, expected)['rate'],
            "description": "uniform(1e3, 1e6)",
        }
    )

    # Testcase 3
    x = create_uniform_dataset(1e6, 1e9)
    actual = fxp_acc.reciprocal(x)
    expected = list(np.reciprocal(np.array(x)))
    res.append(
        {
            "id": "Testcase 3",
            "pass_rate": check(actual, expected)['rate'],
            "description": "uniform(1e6, 1e9)",
        }
    )

    # Testcase 4
    x = create_uniform_dataset(-1e3, 0)
    actual = fxp_acc.reciprocal(x)
    expected = list(np.reciprocal(np.array(x)))
    res.append(
        {
            "id": "Testcase 4",
            "pass_rate": check(actual, expected)['rate'],
            "description": "uniform(-1e3, 0)",
        }
    )

    # Testcase 5
    x = create_uniform_dataset(-(2**18), 2**18)
    actual = fxp_acc.reciprocal(x)
    expected = list(np.reciprocal(np.array(x)))
    res.append(
        {
            "id": "Testcase 4",
            "pass_rate": check(actual, expected)['rate'],
            "description": "uniform(-2**18, 2**18)",
        }
    )

    return res


def div_sim():
    res = []

    # Testcase 1
    x = create_uniform_dataset(-(2**18), 2**18)
    y = create_uniform_dataset(-(2**18), 2**18)
    actual = fxp_acc.div(x, y)
    expected = list(np.array(x) / np.array(y))
    res.append(
        {
            "id": "Testcase 1",
            "pass_rate": check(actual, expected)['rate'],
            "description": "both x and y belongs to uniform(-2**18, 2**18)",
        }
    )

    # Testcase 2
    x = create_uniform_dataset(-(2**18), 2**18)
    y = create_uniform_dataset(-(2**9), 2**9)
    actual = fxp_acc.div(x, y)
    expected = list(np.array(x) / np.array(y))
    res.append(
        {
            "id": "Testcase 2",
            "pass_rate": check(actual, expected)['rate'],
            "description": "x: uniform(-2**18, 2**18), y: uniform(-2**9, 2**9)",
        }
    )

    # Testcase 3
    x = create_uniform_dataset(-(2**9), 2**9)
    y = create_uniform_dataset(-(2**18), 2**18)
    actual = fxp_acc.div(x, y)
    expected = list(np.array(x) / np.array(y))
    res.append(
        {
            "id": "Testcase 3",
            "pass_rate": check(actual, expected)['rate'],
            "description": "x: uniform(-2**9, 2**9), y: uniform(-2**18, 2**18)",
        }
    )

    return res


def log_sim():
    res = []

    # Testcase 1
    x = create_uniform_dataset(0, 2**18)
    actual = fxp_acc.log(x)
    expected = list(np.log(np.array(x)))
    res.append(
        {
            "id": "Testcase 1",
            "pass_rate": check(actual, expected)['rate'],
            "description": "uniform(0, 2**18)",
        }
    )

    return res


def logistic_sim():
    res = []

    # Testcase 1
    x = create_uniform_dataset(-10, 10)
    actual = fxp_acc.logistic(x)
    expected = list(1 / (1 + np.exp(-np.array(x))))

    res.append(
        {
            "id": "Testcase 1",
            "pass_rate": check(actual, expected)['rate'],
            "description": "uniform(-10, 10)",
        }
    )

    return res


def summarize_report(rets, output):
    writer = MarkdownTableWriter()
    writer.headers = ['id', 'Pass Rate', 'Description']
    writer.table_name = 'FXP Accuracy Experiments Report'

    res = ''
    for k, v in rets.items():
        writer.table_name = k
        value_matrix = []
        for item in v:
            value_matrix.append([item["id"], item["pass_rate"], item["description"]])
        writer.value_matrix = value_matrix
        writer.write_table()
        writer.write_null_line()
        res += writer.dumps()

    if output:
        with open(output, 'w') as f:
            f.write(res)


def main():
    parser = argparse.ArgumentParser(
        description='Create dataset for fxp acc experiments.'
    )
    parser.add_argument('--output', type=str, help='Output folder.', default='')
    args = parser.parse_args()

    rets = {}
    rets["exp"] = exp_sim()
    rets["reciprocal"] = reciprocal_sim()
    rets["div"] = div_sim()
    rets["log"] = log_sim()
    rets["logistic"] = logistic_sim()
    summarize_report(rets, args.output)


if __name__ == '__main__':
    main()
