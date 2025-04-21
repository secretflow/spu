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
# See the License for the specific language governing permissions and
# limitations under the License.

# Start nodes.
# > bazel run -c opt //examples/python/utils:nodectl -- up
#
# Run this example script.
# > bazel run -c opt //examples/python/ml/jax_kmeans:jax_kmeans


import argparse
import json

import jax
import jax.numpy as jnp
import numpy as np

import spu.utils.distributed as ppd

parser = argparse.ArgumentParser(description="distributed driver.")
parser.add_argument("-c", "--config", default="examples/python/conf/3pc.json")
args = parser.parse_args()

with open(args.config, "r") as file:
    conf = json.load(file)

ppd.init(conf["nodes"], conf["devices"])


def kmeans(data_alice, data_bob, k, random_numbers, n_epochs=10):
    # data: data set
    # k: number of clusters
    # return: cluster centers, labels

    data = jnp.append(data_alice, data_bob, axis=0)

    n = data.shape[0]
    m = data.shape[1]

    # initialize cluster centers
    centers = jnp.zeros((k, m))

    def loop_for_1(i, stat):
        centers = stat
        centers = centers.at[i].set(data[random_numbers[i]])
        return centers

    centers = jax.lax.fori_loop(0, k, loop_for_1, centers)

    # initialize labels
    labels = jnp.zeros(n)

    def loop_while_func(stat):
        rounds, centers, labels = stat

        def loop_for_1(i, stat):
            labels = stat
            min_dist = jnp.linalg.norm(data[i] - centers[0])
            labels = labels.at[i].set(0)

            def loop_for_2(j, stat):
                min_dist, labels = stat
                dist = jnp.linalg.norm(data[i] - centers[j])
                labels = labels.at[i].set(jnp.where(min_dist > dist, j, labels[i]))
                min_dist = jnp.minimum(dist, min_dist)
                return min_dist, labels

            stat = (min_dist, labels)
            min_dist, labels = jax.lax.fori_loop(1, k, loop_for_2, stat)
            return labels

        labels = jax.lax.fori_loop(0, n, loop_for_1, labels)

        def loop_for_3(i, stat):
            new_centers = stat
            sums = jnp.zeros(m)

            def loop_for_4(j, sums):
                sums = sums + jnp.where(labels[j] == i, data[j], jnp.zeros(m))
                return sums

            sums = jax.lax.fori_loop(0, n, loop_for_4, sums)
            cluster_numbers = jnp.sum(jnp.where(labels == i, 1, 0))
            avg = jnp.where(cluster_numbers != 0, sums / cluster_numbers, jnp.zeros(m))
            new_centers = new_centers.at[i].set(avg)
            return new_centers

        centers = jax.lax.fori_loop(0, k, loop_for_3, jnp.zeros((k, m)))

        return rounds + 1, centers, labels

    stat = (0, centers, labels)
    res = jax.lax.while_loop(lambda stat: stat[0] < n_epochs, loop_while_func, stat)

    return res[1], res[2]


@ppd.device("P1")
def points_from_alice(npoints):
    np.random.seed(0xDEADBEEF)
    return np.random.normal(size=(npoints, 2)) * 5


@ppd.device("P2")
def points_from_bob(npoints):
    np.random.seed(0xC0FFEE)
    return np.random.normal(size=(npoints, 2)) * 5


def run_on_cpu():
    # Two party share a dataset, and use kmeans to cluster them.

    n_points_each_party = 30
    k = 2
    n_epochs = 10

    np.random.seed(0xDEADBEEF)
    x = np.random.normal(size=(n_points_each_party, 2)) * 5

    np.random.seed(0xC0FFEE)
    y = np.random.normal(size=(n_points_each_party, 2)) * 5

    np.random.seed(0xCAFEBABE)
    random_numbers = np.random.randint(n_points_each_party * 2, size=(k,))

    centers, labels = jax.jit(kmeans, static_argnums=(2, 4))(
        x, y, k, random_numbers, n_epochs
    )

    print(centers)
    print(labels)
    return centers, labels


def run_on_spu():
    # Two party share a dataset, and use kmeans to cluster them.

    n_points_each_party = 30
    k = 2
    n_epochs = 10

    x = points_from_alice(n_points_each_party)
    y = points_from_bob(n_points_each_party)

    # This random_number is used to initialize cluster centers. It's better generated in CPU instead of SPU.
    np.random.seed(0xCAFEBABE)
    random_numbers = np.random.randint(n_points_each_party * 2, size=(k,))

    centers_spu, labels_spu = ppd.device("SPU")(kmeans, static_argnums=(2, 4))(
        x, y, k, random_numbers, n_epochs
    )
    centers = ppd.get(centers_spu)
    labels = ppd.get(labels_spu)

    print(centers)
    print(labels)
    return centers, labels


if __name__ == "__main__":
    print('Run on CPU\n------\n')
    run_on_cpu()
    print('Run on SPU\n------\n')
    run_on_spu()
