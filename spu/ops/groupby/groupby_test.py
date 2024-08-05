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
import os
import sys
import time
import unittest

import numpy as np
import pandas as pd

import spu.spu_pb2 as spu_pb2
import spu.utils.simulation as spsim

# Add the ops directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from groupby import (
    groupby,
    groupby_agg_postprocess,
    groupby_sum_via_shuffle,
    groupby_max_via_shuffle,
    groupby_count,
    view_key,
    view_key_postprocessing,
)


class UnitTests(unittest.TestCase):
    def test_sum(self):
        sim = spsim.Simulator.simple(
            3, spu_pb2.ProtocolKind.ABY3, spu_pb2.FieldType.FM64
        )

        def proc(x1, x2, y):
            return groupby([x1[:, 2], x2[:, 3]], [y])

        def proc_sum_shuffle(cols, seg_end_marks, segment_ids, secret_random_order):
            return groupby_sum_via_shuffle(
                cols, seg_end_marks, segment_ids, secret_random_order
            )

        def proc_view_key(key_cols, segment_end_marks, key):
            return view_key(key_cols, segment_end_marks, key)

        n_rows = 3000
        n_cols = 10
        x1 = np.random.random((n_rows, n_cols))
        x2 = np.random.random((n_rows, n_cols))
        y = np.random.random((n_rows,))
        # groupby category only supports discrete values
        # here we are taking the shortcut, in reality, only the key need to be discrete.
        # (their difference should be large enough so that their fxp repr are difference)
        # we shrink the data value in order to reduce the size of groups.
        # in our test data, we will get a group size about 15
        x1 = (x1 * 10 / 3).astype(int)
        x2 = (x2 * 10 / 3).astype(int)
        y = (y * 10 / 3).astype(int)
        start = time.perf_counter()
        keys, target_cols, segment_ids, segment_end_marks = spsim.sim_jax(sim, proc)(
            x1, x2, y
        )
        end = time.perf_counter()
        print("groupby takes time", end - start)
        X = np.zeros((x1.shape[0], 3))
        X[:, 0] = x1[:, 2]
        X[:, 1] = x2[:, 3]
        X[:, 2] = y
        df = pd.DataFrame(
            X,
            columns=[f'col{i}' for i in range(3)],
        )
        # Perform group by sum using pandas
        pandas_groupby_sum = df.groupby([df.columns[0], df.columns[1]])[
            df.columns[2]
        ].sum()
        num_groups = pandas_groupby_sum.shape[0]
        # num_groups can also be obtained by revealing segment_ids[-1]
        start = time.perf_counter()
        # how to produce a secret random array of shape (row_num, ) is another question (not addressed here).
        p1_random_order = np.random.random((X.shape[0],))
        p2_random_order = np.random.random((X.shape[0],))
        secret_random_order = p1_random_order + p2_random_order
        agg_result = spsim.sim_jax(sim, proc_sum_shuffle)(
            target_cols, segment_end_marks, segment_ids, secret_random_order
        )
        agg_result = groupby_agg_postprocess(
            agg_result[0], agg_result[1], agg_result[2], num_groups
        )
        end = time.perf_counter()
        print("sum takes take", end - start)
        assert (
            np.max(
                abs(pandas_groupby_sum.values.reshape(agg_result.shape) - agg_result)
            )
            < 0.001
        ), f"{pandas_groupby_sum}, ours: \n {agg_result}"

        correct_keys = list(pandas_groupby_sum.index.to_numpy())
        correct_keys = np.array([[*a] for a in correct_keys])

        # we open shuffled keys and take set(keys)
        start = time.perf_counter()
        # how to produce a secret random array of shape (row_num, ) is another question (not addressed here).
        p1_random_order = np.random.random((X.shape[0],))
        p2_random_order = np.random.random((X.shape[0],))
        secret_random_order = p1_random_order + p2_random_order
        keys = spsim.sim_jax(sim, proc_view_key)(
            keys,
            segment_end_marks,
            secret_random_order,
        )

        keys = view_key_postprocessing(keys, num_groups)

        end = time.perf_counter()
        print("view key takes take", end - start)
        assert (
            np.max(abs(correct_keys - keys)) < 0.001
        ), f"value{ max(abs(correct_keys - keys))}, correct_keys, {correct_keys}, keys{keys}"

    def test_count(self):
        sim = spsim.Simulator.simple(
            3, spu_pb2.ProtocolKind.ABY3, spu_pb2.FieldType.FM64
        )

        def proc(x1, x2, y):
            return groupby([x1[:, 2], x2[:, 3]], [y])

        # for count operation, we can open segment_ids and sorted_keys

        n_rows = 3000
        n_cols = 10
        x1 = np.random.random((n_rows, n_cols))
        x2 = np.random.random((n_rows, n_cols))
        y = np.random.random((n_rows,))
        # groupby category only supports discrete values
        # here we are taking the shortcut, in reality, only the key need to be discrete.
        # (their difference should be large enough so that their fxp repr are difference)
        # we shrink the data value in order to reduce the size of groups.
        # in our test data, we will get a group size about 15
        x1 = (x1 * 10 / 3).astype(int)
        x2 = (x2 * 10 / 3).astype(int)
        y = (y * 10 / 3).astype(int)
        start = time.perf_counter()
        keys, _, segment_ids, _ = spsim.sim_jax(sim, proc)(x1, x2, y)
        end = time.perf_counter()
        print("groupby takes time", end - start)
        X = np.zeros((x1.shape[0], 3))
        X[:, 0] = x1[:, 2]
        X[:, 1] = x2[:, 3]
        X[:, 2] = y
        df = pd.DataFrame(
            X,
            columns=[f'col{i}' for i in range(3)],
        )
        # Perform group by sum using pandas
        pandas_groupby_sum = df.groupby([df.columns[0], df.columns[1]])[
            df.columns[2]
        ].count()

        start = time.perf_counter()
        count_result = groupby_count(segment_ids)
        end = time.perf_counter()
        print("count takes take", end - start)
        assert (
            np.max(
                abs(
                    pandas_groupby_sum.values.reshape(count_result.shape) - count_result
                )
            )
            < 0.001
        ), f"{pandas_groupby_sum}, ours: \n {count_result}"

        correct_keys = list(pandas_groupby_sum.index.to_numpy())
        correct_keys = np.array([[*a] for a in correct_keys])

        # we open shuffled keys and take set(keys)
        start = time.perf_counter()
        # how to produce a secret random array of shape (row_num, ) is another question (not addressed here).

        keys = view_key_postprocessing(keys, segment_ids[-1] + 1)

        end = time.perf_counter()
        print("view key takes take", end - start)
        assert (
            np.max(abs(correct_keys - keys)) < 0.001
        ), f"value{ max(abs(correct_keys - keys))}, correct_keys, {correct_keys}, keys{keys}"

    def test_max(self):
        sim = spsim.Simulator.simple(
            3, spu_pb2.ProtocolKind.ABY3, spu_pb2.FieldType.FM64
        )

        def proc(x1, x2, y):
            return groupby([x1[:, 2], x2[:, 3]], [y])

        def proc_max_shuffle(cols, seg_end_marks, segment_ids, secret_random_order):
            return groupby_max_via_shuffle(
                cols, seg_end_marks, segment_ids, secret_random_order
            )

        def proc_view_key(key_cols, segment_end_marks, key):
            return view_key(key_cols, segment_end_marks, key)

        n_rows = 30000
        n_cols = 10
        x1 = np.random.random((n_rows, n_cols))
        x2 = np.random.random((n_rows, n_cols))
        y = np.random.random((n_rows,))
        # groupby category only supports discrete values
        # here we are taking the shortcut, in reality, only the key need to be discrete.
        # (their difference should be large enough so that their fxp repr are difference)
        # we shrink the data value in order to reduce the size of groups.
        # in our test data, we will get a group size about 15
        x1 = (x1 * 10 / 3).astype(int)
        x2 = (x2 * 10 / 3).astype(int)
        y = (y * 10 / 3).astype(int)
        start = time.perf_counter()
        keys, target_cols, segment_ids, segment_end_marks = spsim.sim_jax(sim, proc)(
            x1, x2, y
        )
        end = time.perf_counter()
        print("groupby takes time", end - start)
        X = np.zeros((x1.shape[0], 3))
        X[:, 0] = x1[:, 2]
        X[:, 1] = x2[:, 3]
        X[:, 2] = y
        df = pd.DataFrame(
            X,
            columns=[f'col{i}' for i in range(3)],
        )
        # Perform group by sum using pandas
        pandas_groupby_sum = df.groupby([df.columns[0], df.columns[1]])[
            df.columns[2]
        ].max()
        num_groups = pandas_groupby_sum.shape[0]
        # num_groups can also be obtained by revealing segment_ids[-1]
        start = time.perf_counter()
        # how to produce a secret random array of shape (row_num, ) is another question (not addressed here).
        p1_random_order = np.random.random((X.shape[0],))
        p2_random_order = np.random.random((X.shape[0],))
        secret_random_order = p1_random_order + p2_random_order
        agg_result = spsim.sim_jax(sim, proc_max_shuffle)(
            target_cols, segment_end_marks, segment_ids, secret_random_order
        )
        agg_result = groupby_agg_postprocess(
            agg_result[0], agg_result[1], agg_result[2], num_groups
        )
        end = time.perf_counter()
        print("sum takes take", end - start)
        assert (
            np.max(
                abs(pandas_groupby_sum.values.reshape(agg_result.shape) - agg_result)
            )
            < 0.001
        ), f"{pandas_groupby_sum}, ours: \n {agg_result}"

        correct_keys = list(pandas_groupby_sum.index.to_numpy())
        correct_keys = np.array([[*a] for a in correct_keys])

        # we open shuffled keys and take set(keys)
        start = time.perf_counter()
        # how to produce a secret random array of shape (row_num, ) is another question (not addressed here).
        p1_random_order = np.random.random((X.shape[0],))
        p2_random_order = np.random.random((X.shape[0],))
        secret_random_order = p1_random_order + p2_random_order
        keys = spsim.sim_jax(sim, proc_view_key)(
            keys,
            segment_end_marks,
            secret_random_order,
        )

        keys = view_key_postprocessing(keys, num_groups)

        end = time.perf_counter()
        print("view key takes take", end - start)
        assert (
            np.max(abs(correct_keys - keys)) < 0.001
        ), f"value{ max(abs(correct_keys - keys))}, correct_keys, {correct_keys}, keys{keys}"


if __name__ == "__main__":
    unittest.main()
