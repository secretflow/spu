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


import jax.numpy as jnp


class DecisionTreeClassifier:
    """A decision tree classifier based on [GTree](https://arxiv.org/abs/2305.00645).

    Adopting a MPC-based linear scan method (i.e. oblivious_array_access), GTree
    designs a new GPU-friendly oblivious decision tree training protocol, which is
    more efficient than the prior works. The current implementation supports the training
    of decision tree with binary features (i.e. {0, 1}) and multi-class labels (i.e. {0, 1, 2, \dots}).

    We provide a simple example to show how to use GTree to train a decision tree classifier
    in sml/tree/emulations/tree_emul.py. For training, the memory and time complexity is around
    O(n_samples * n_labels * n_features * 2 ** max_depth).

    Parameters
    ----------
    criterion : {"gini"}, default="gini"
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity.

    splitter : {"best"}, default="best"
        The strategy used to choose the split at each node. Supported
        strategies are "best" to choose the best split.

    max_depth : int
        The maximum depth of the tree. Must specify an integer > 0.

    n_labels: int, the max number of labels.
    """

    def __init__(self, criterion, splitter, max_depth, n_labels):
        assert criterion == "gini", "criteria other than gini is not supported."
        assert splitter == "best", "splitter other than best is not supported."
        assert (
            max_depth is not None and max_depth > 0
        ), "max_depth should not be None and must > 0."
        self.max_depth = max_depth
        self.n_labels = n_labels
        # self.sample_weight = sample_weight

    def fit(self, X, y, sample_weight=None):
        self.T, self.F = odtt(X, y, self.max_depth, self.n_labels, sample_weight)
        return self

    def predict(self, X):
        assert self.T != None, "the model has not been trained yet."
        return odti(X, self.T, self.max_depth)


'''
The protocols of GTree.
'''


def oblivious_array_access(array, index):
    '''
    Extract elements from array according to index.

    If array is 1D, then output [array[i] for i in index].
    e.g.: array = [1, 2, 3, 4, 5], index = [0, 2, 4], output = [1, 3, 5].

    If array is 2D, then output [[array[j, i] for i in index] for j in range(array.shape[0])].
    e.g. array = [[1, 2, 3], [4, 5, 6]], index_array = [0, 2], output = [[1, 3], [4, 6]].
    '''
    # (n_array)
    count_array = jnp.arange(0, array.shape[-1])
    # (n_array, n_index)
    E = jnp.equal(index, count_array[:, jnp.newaxis])

    assert len(array.shape) <= 2, "OAA protocol only supports 1D or 2D array."

    # OAA basic case
    if len(array.shape) == 1:
        # (n_array, n_index)
        O = array[:, jnp.newaxis] * E  # select shares
        zu = jnp.sum(O, axis=0)
    # OAA vectorization variant
    elif len(array.shape) == 2:
        # (n_arrays, n_array, n_index)
        O = array[:, :, jnp.newaxis] * E[jnp.newaxis, :, :]  # select shares
        zu = jnp.sum(O, axis=1)
    return zu


def oaa_elementwise(array, index_array):
    '''
    Given index_array, output [array[i, index[i]] for i in range(len(array))].

    e.g.: array = [[1, 2, 3], [4, 5, 6]], index = [0, 2], output = [1, 6].
    '''
    assert index_array.shape[0] == array.shape[0], "n_arrays must be equal to n_index."
    assert len(array.shape) == 2, "OAAE protocol only supports 2D array."
    count_array = jnp.arange(0, array.shape[-1])
    # (n_array, n_index)
    E = jnp.equal(index_array[:, jnp.newaxis], count_array)
    if len(array.shape) == 2:
        O = array * E
        zu = jnp.sum(O, axis=1)
    return zu


# def oblivious_learning(X, y, T, F, M, Cn, h):
def oblivious_learning(X, y, T, F, M, h, Cn, n_labels, sample_weight=None):
    '''partition the data and count the number of data samples.

    params:
        D:  data samples, which is splitted into X, y.  X: (n_samples, n_features), y: (n_samples, 1).
        T:  tree structure reprensenting split features. (total_nodes)
        F:  tree structure reprensenting node types. (total_nodes)
            0 for internal, 1 for leaf, 2 for dummy.
        M:  which leave node does D[i] belongs to (for level h-1).  (n_samples)
        Cn: statical information of the data samples. (n_leaves, n_labels+1, 2*n_features)
        h:  int, current depth of the tree.
    '''
    # line 1-5, partition the datas into new leaves.
    n_d, n_f = X.shape
    n_h = 2**h
    if h != 0:
        Tval = oaa(T, M)
        Dval = oaae(X, Tval)
        M = 2 * M + Dval + 1

    LCidx = jnp.arange(0, n_h)
    isLeaf = jnp.equal(F[n_h - 1 : 2 * n_h - 1], jnp.ones(n_h))
    LCF = jnp.equal(M[:, jnp.newaxis] - n_h + 1, LCidx)
    LCF = LCF * isLeaf

    Cd = jnp.zeros((n_d, n_h, n_labels + 1, 2 * n_f))
    if sample_weight is not None:
        Cd = Cd.at[:, :, 0, 0::2].set(
            jnp.tile((1 - X)[:, jnp.newaxis, :] * sample_weight[:, jnp.newaxis, jnp.newaxis], (1, n_h, 1))
        )
        Cd = Cd.at[:, :, 0, 1::2].set(
            jnp.tile((X)[:, jnp.newaxis, :] * sample_weight[:, jnp.newaxis, jnp.newaxis], (1, n_h, 1))
        )
    else:
        Cd = Cd.at[:, :, 0, 0::2].set(jnp.tile((1 - X)[:, jnp.newaxis, :], (1, n_h, 1)))
        Cd = Cd.at[:, :, 0, 1::2].set(jnp.tile((X)[:, jnp.newaxis, :], (1, n_h, 1)))

    for i in range(n_labels):
        if sample_weight is not None:
            Cd = Cd.at[:, :, i + 1, 0::2].set(
                jnp.tile(
                    ((1 - X)[:, jnp.newaxis, :] * (i == y)[:, jnp.newaxis, jnp.newaxis] * sample_weight[:, jnp.newaxis, jnp.newaxis]),
                    (1, n_h, 1)
                )
            )
            Cd = Cd.at[:, :, i + 1, 1::2].set(
                jnp.tile(
                    ((X)[:, jnp.newaxis, :] * (i == y)[:, jnp.newaxis, jnp.newaxis] * sample_weight[:, jnp.newaxis, jnp.newaxis]),
                    (1, n_h, 1)
                )
            )
        else:
            Cd = Cd.at[:, :, i + 1, 0::2].set(
                jnp.tile(
                    ((1 - X) * (i == y)[:, jnp.newaxis])[:, jnp.newaxis, :],
                    (1, n_h, 1)
                )
            )
            Cd = Cd.at[:, :, i + 1, 1::2].set(
                jnp.tile(
                    ((X) * (i == y)[:, jnp.newaxis])[:, jnp.newaxis, :],
                    (1, n_h, 1)
                )
            )

    Cd = Cd * LCF[:, :, jnp.newaxis, jnp.newaxis]

    new_Cn = jnp.sum(Cd, axis=0)

    if h != 0:
        Cn = Cn.repeat(2, axis=0)
    new_Cn = new_Cn[:, :, :] + Cn[:, :, :] * (1 - isLeaf[:, jnp.newaxis, jnp.newaxis])

    return new_Cn, M



def oblivious_heuristic_computation(Cn, gamma, F, h, n_labels):
    '''Compute gini index, find the best feature, and update F.

    params:
        Cn:         statical information of the data samples. (n_leaves, n_labels+1, 2*n_features)
        gamma:      gamma[n][i] indicates if feature si has been assigned at node n. (n_leaves, n_features)
        F:          tree structure reprensenting node types. (total_nodes)
                    0 for internal, 1 for leaf, 2 for dummy.
        h:          int, current depth of the tree.
        n_labels:   int, number of labels.
    '''
    n_leaves = Cn.shape[0]
    n_features = gamma.shape[1]
    Ds0 = Cn[:, 0, 0::2]
    Ds1 = Cn[:, 0, 1::2]
    D = Ds0 + Ds1
    Q = D * Ds0 * Ds1
    P = jnp.zeros(gamma.shape)
    for i in range(n_labels):
        P = P - Ds1 * (Cn[:, i + 1, 0::2] ** 2) - Ds0 * (Cn[:, i + 1, 1::2] ** 2)
    gini = Q / (Q + P + 1)
    gini = gini * gamma
    # (n_leaves)
    SD = jnp.argmax(gini, axis=1)
    index = jnp.arange(0, n_features)
    gamma = gamma * jnp.not_equal(index[jnp.newaxis, :], SD[:, jnp.newaxis])
    new_gamma = jnp.zeros((n_leaves * 2, n_features))
    new_gamma = new_gamma.at[0::2, :].set(gamma)
    new_gamma = new_gamma.at[1::2, :].set(gamma)

    # # modification.
    psi = jnp.zeros((n_leaves, n_labels))
    for i in range(n_labels):
        psi = psi.at[:, i].set(Cn[:, i + 1, 0] + Cn[:, i + 1, 1])
    total = jnp.sum(psi, axis=1)
    psi = total[:, jnp.newaxis] - psi
    psi = jnp.prod(psi, axis=1)
    F = F.at[2**h - 1 : 2 ** (h + 1) - 1].set(
        jnp.equal(psi * F[2**h - 1 : 2 ** (h + 1) - 1], 0)
    )
    F = F.at[2 ** (h + 1) - 1 : 2 ** (h + 2) - 1 : 2].set(
        2 - jnp.equal(F[2**h - 1 : 2 ** (h + 1) - 1], 0)
    )
    F = F.at[2 ** (h + 1) : 2 ** (h + 2) - 1 : 2].set(
        F[2 ** (h + 1) - 1 : 2 ** (h + 2) - 1 : 2]
    )
    return SD, new_gamma, F


def oblivious_node_split(SD, T, F, Cn, h, max_depth):
    '''Convert each node into its internal node and generates new leaves at the next level.'''

    T = T.at[2**h - 1 : 2 ** (h + 1) - 1].set(SD)
    return T, Cn


def oblivious_DT_training(X, y, max_depth, n_labels, sample_weight=None):
    n_samples, n_features = X.shape
    T = jnp.zeros((2 ** (max_depth + 1) - 1))
    F = jnp.ones((2**max_depth - 1))
    M = jnp.zeros(n_samples)
    gamma = jnp.ones((1, n_features))
    Cn = jnp.zeros((1, n_labels + 1, 2 * n_features))

    h = 0
    while h < max_depth:
        if sample_weight is not None:
            Cn, M = ol(X, y, T, F, M, h, Cn, n_labels, sample_weight)
        else:
            Cn, M = ol(X, y, T, F, M, h, Cn, n_labels)

        SD, gamma, F = ohc(Cn, gamma, F, h, n_labels)

        T, Cn = ons(SD, T, F, Cn, h, max_depth)

        h += 1

    n_leaves = 2**h
    psi = jnp.zeros((n_leaves, n_labels))
    for i in range(2 ** (h - 1)):
        t1 = oaa(Cn[i, 1:], 2 * SD[i : i + 1]).squeeze()
        t2 = oaa(Cn[i, 1:], 2 * SD[i : i + 1] + 1).squeeze()
        psi = psi.at[2 * i, :].set(t1)
        psi = psi.at[2 * i + 1, :].set(t2)
    T = T.at[n_leaves - 1 :].set(jnp.argmax(psi, axis=1))
    return T, F


def oblivious_DT_inference(X, T, max_height):
    n_samples, n_features = X.shape
    Tidx = jnp.zeros((n_samples))
    i = 0
    while i < max_height:
        Tval = oaa(T, Tidx)
        Dval = oaae(X, Tval)
        Tidx = Tidx * 2 + Dval + 1
        i += 1
    Tval = oaa(T, Tidx)
    return Tval


oaa = oblivious_array_access
oaae = oaa_elementwise
ol = oblivious_learning
ohc = oblivious_heuristic_computation
ons = oblivious_node_split
odtt = oblivious_DT_training
odti = oblivious_DT_inference
