# Copyright 2025 Ant Group Co., Ltd.
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
import jax
import jax.numpy as jnp


def mpc_kneighbors_graph(
    X,
    n_neighbors,
    *,
    mode="distance",
    metric="minkowski",
    p=2,
):
    """
    Compute the k-nearest neighbors graph in a privacy-preserving manner using Secure Multi-Party Computation (MPC).

    This function calculates the Euclidean distance between all pairs of input samples,
    finds the k-nearest neighbors for each sample, and returns a matrix where non-nearest
    neighbor distances are set to zero.

    Args:
        X: A (num_samples, num_features) matrix containing the input samples.
        n_neighbors: The number of nearest neighbors to retain for each sample, excluding the sample itself.
        mode: Specifies the output format (default: "distance").
        metric: The distance metric used (default: "minkowski").
        p: The order of the Minkowski distance (default: 2 for Euclidean distance).

    Returns:
        Knn3: A (num_samples, num_samples) matrix where each row contains the distances
              to its k-nearest neighbors, with all other entries set to zero.
    """

    assert mode == "distance", "mode must be 'distance'"
    assert metric == "minkowski", "metric must be 'minkowski'"
    assert p == 2, "p must be 2"

    num_samples = X.shape[0]

    # Calculate the square of the Euclidean distance between every two samples
    X_expanded = jnp.expand_dims(X, axis=1) - jnp.expand_dims(X, axis=0)
    X_expanded = jnp.square(X_expanded)
    Dis = jnp.sum(X_expanded, axis=-1)

    # Sort each row of Dis
    Index_Dis = jnp.argsort(Dis, axis=1)
    Knn = Dis
    Knn = jax.lax.sort(Knn, dimension=-1)

    # Find the square root of the Euclidean distance of the nearest neighbor previously calculated, and set the distance of non nearest neighbors to 0
    Knn2 = jnp.zeros((num_samples, num_samples))

    def update_knn_row(i, Knn_row, n_neighbors):
        def update_element(j, Knn_value):
            return jnp.where(j <= n_neighbors, jnp.sqrt(Knn_value), 0)

        # Vectorize the inner loop over `j`
        Knn_row_updated = jax.vmap(update_element, in_axes=(0, 0))(
            jnp.arange(Knn_row.shape[0]), Knn_row
        )
        return Knn_row_updated

    # Vectorize the outer loop over `i`
    Knn2 = jax.vmap(lambda i, Knn_row: update_knn_row(i, Knn_row, n_neighbors))(
        jnp.arange(num_samples), Knn
    )

    # Reverse permutation of Dis to restore the previous order
    Knn3 = jax.lax.sort(operand=(Index_Dis, Knn2), dimension=-1, num_keys=1)[1]

    return Knn3
