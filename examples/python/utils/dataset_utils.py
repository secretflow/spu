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

import numpy as np
import array
import gzip
import os
from os import path
import struct
import urllib.request


def standardize(data):
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.fit(data)
    return scaler.transform(data)


def add_constant_col(data):
    return np.c_[data, np.ones((data.shape[0], 1))]


def breast_cancer(
    col_slicer=slice(None, None, None),
    train: bool = True,
    *,
    normalize: bool = True,
):
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split

    ds = load_breast_cancer()
    x, y = ds['data'], ds['target']
    if normalize:
        x = (x - np.min(x)) / (np.max(x) - np.min(x))
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    if train:
        x_ = x_train
        y_ = y_train
    else:
        x_ = x_test
        y_ = y_test
    x_ = x_[:, col_slicer]
    return x_, y_


_DATA = "/tmp/jax_example_data/"


def _download(url, filename):
    """Download a url to a file in the JAX data temp directory."""
    if not path.exists(_DATA):
        os.makedirs(_DATA)
    out_file = path.join(_DATA, filename)
    if not path.isfile(out_file):
        urllib.request.urlretrieve(url, out_file)
        print("downloaded {} to {}".format(url, _DATA))


def _partial_flatten(x):
    """Flatten all but the first dimension of an ndarray."""
    return np.reshape(x, (x.shape[0], -1))


def _one_hot(x, k, dtype=np.float32):
    """Create a one-hot encoding of x of size k."""
    return np.array(x[:, None] == np.arange(k), dtype)


def mnist_raw():
    """Download and parse the raw MNIST dataset."""
    # CVDF mirror of http://yann.lecun.com/exdb/mnist/
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"

    def parse_labels(filename):
        with gzip.open(filename, "rb") as fh:
            _ = struct.unpack(">II", fh.read(8))
            return np.array(array.array("B", fh.read()), dtype=np.uint8)

    def parse_images(filename):
        with gzip.open(filename, "rb") as fh:
            _, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))
            return np.array(array.array("B", fh.read()), dtype=np.uint8).reshape(
                num_data, rows, cols
            )

    for filename in [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]:
        _download(base_url + filename, filename)

    train_images = parse_images(path.join(_DATA, "train-images-idx3-ubyte.gz"))
    train_labels = parse_labels(path.join(_DATA, "train-labels-idx1-ubyte.gz"))
    test_images = parse_images(path.join(_DATA, "t10k-images-idx3-ubyte.gz"))
    test_labels = parse_labels(path.join(_DATA, "t10k-labels-idx1-ubyte.gz"))

    return train_images, train_labels, test_images, test_labels


def mnist(permute_train=False):
    """Download, parse and process MNIST data to unit scale and one-hot labels."""
    train_images, train_labels, test_images, test_labels = mnist_raw()

    train_images = train_images / np.float32(255.0)
    test_images = test_images / np.float32(255.0)
    train_labels = _one_hot(train_labels, 10)
    test_labels = _one_hot(test_labels, 10)

    if permute_train:
        perm = np.random.RandomState(0).permutation(train_images.shape[0])
        train_images = train_images[perm]
        train_labels = train_labels[perm]

    return train_images, train_labels, test_images, test_labels


def mock_classification(n_samples, n_features, hardness=0.1, random_seed=None):
    """Generate a mock classification dataset.
    Use scikit learn make classification utils,
    which is much better than naively randomly sampled data.
    https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html#sklearn.datasets.make_classification

    hardness should be between 0 and 1.
    1 -> completely random dataset
    0 -> completely clean dataset: no noisy feature/label
         and great separation between classes
    """
    from sklearn.datasets import make_classification

    hardness = max(min(hardness, 1.0), 0.0)
    informative_ratio = 1.0 - hardness
    redundant_ratio = (1.0 - informative_ratio) * hardness
    class_sep = 1.0 - hardness
    flip_y = 0.5 * hardness
    X, y = make_classification(
        n_samples,
        n_features,
        n_informative=int(informative_ratio * n_features),
        n_redundant=int(redundant_ratio * n_features),
        class_sep=class_sep,
        flip_y=flip_y,
        random_state=random_seed,
    )
    return X, y


def mock_regression(n_samples, n_features, hardness=0.1, random_seed=None):
    """Generate a mock regression dataset.
    Use scikit learn make regression utils,
    which is much better than naively randomly sampled data.
    https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html#sklearn.datasets.make_regression

    hardness should be between 0 and 1.
    1 -> completely random dataset
    0 -> completely clean dataset: no noisy feature and y values
    """
    from sklearn.datasets import make_regression

    hardness = max(min(hardness, 1.0), 0.0)
    informative_ratio = 1.0 - hardness
    X, y, coef = make_regression(
        n_samples,
        n_features,
        n_informative=int(informative_ratio * n_features),
        noise=hardness,
        coef=True,
        random_state=random_seed,
    )
    # coef is the underlying true coefficients for the linear model
    return X, y, coef


def mock_two_party_split(X, ratio=0.5):
    """
    The ratio fraction of X is data of p1,
    and the (1-ratio) fraction of X is data of p2"""
    p1_col_num = int(X.shape[0] * ratio)
    X_a = X[:, :p1_col_num]
    X_b = X[:, p1_col_num:]
    return X_a, X_b


# TODO(zoupeicheng.zpc): validate configuration, assumes correct for now.
def load_dataset_by_config(config):
    """
    This function loads the dataset from a configuration file in folder examples/python/conf/.

    ML algorithms run on different datasets. In order to quickly swap datasets to test the same algorithms,
    or make comparisons between different algorithms on the same datasets without changing the source code frequently,
    we designed a dataset configuration.

    Currently, we provide support for vertical two-party split datasets for some builtin datasets and mock datasets.

    dataset configurations should have prefix ds_.

    TODO(zoupeicheng.zpc): support for more datasets.

    If a dataset is trying to use mock data,
    the config file must be like the following sample configuration:
    {
        "use_mock_data": true,
        "n_samples": 1000,
        "n_features": 100,
        "problem_type": "regression",
        "random_seed": 9237,
        "hardness": 0.1,
        "left_slice_feature_ratio": 0.5
    }

    If a dataset is trying to use sklearn builtin toy data,
    the config file must be like the following sample configuration:
    {
        "use_mock_data": false,
        "builtin_dataset_name": "breast_cancer",
        "left_slice_feature_ratio": 0.5,
    }
    """
    if config["use_mock_data"]:
        if config["problem_type"] == "regression":
            X, y, _ = mock_regression(
                config["n_samples"],
                config["n_features"],
                config["hardness"],
                config["random_seed"],
            )
        elif config["problem_type"] == "classification":
            X, y = mock_classification(
                config["n_samples"],
                config["n_features"],
                config["hardness"],
                config["random_seed"],
            )
    else:
        if config["builtin_dataset_name"] == "breast_cancer":
            from sklearn.datasets import load_breast_cancer

            ds = load_breast_cancer()
        elif config["builtin_dataset_name"] == "diabetes":
            from sklearn.datasets import load_diabetes

            ds = load_diabetes()
        X, y = ds['data'], ds['target']
        # normalize, TODO(zoupeicheng.zpc): make preprocessing configurable
        X = (X - np.min(X)) / (np.max(X) - np.min(X))
    split_index = int(X.shape[1] * config["left_slice_feature_ratio"])
    return X[:, :split_index], X[:, split_index:], y


def load_feature_r1(x, y):
    return x, y


def load_feature_r2(x):
    return x
