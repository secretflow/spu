# Copyright 2025 Ant Group Co., Ltd.
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

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import random
from sklearn.datasets import load_digits, load_iris
from sklearn.manifold import TSNE as SklearnTSNE
from sklearn.preprocessing import StandardScaler

# Use libspu enums and spsim directly
import spu.libspu as libspu
import spu.utils.simulation as spsim

# Add base directory to path to import sml library
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))
from sml.decomposition.tsne import basic_tsne  # Replace with the actual module name


class TestTSNEComparison(unittest.TestCase):
    def test_tsne_similarity(self):
        # Load dataset
        data = load_digits(n_class=6)
        # data = load_iris()
        X = data.data
        y = data.target

        # Parameters
        n_components = 2
        perplexity = 30
        max_iter = 500
        random_state = 42

        # Run scikit-learn t-SNE
        sklearn_tsne = SklearnTSNE(
            n_components=n_components,
            perplexity=perplexity,
            n_iter=max_iter,
            random_state=random_state,
        )
        Y_sklearn = sklearn_tsne.fit_transform(X)

        # Run JAX t-SNE
        Y_jax = basic_tsne(
            X,
            n_components=n_components,
            perplexity=perplexity,
            max_iter=max_iter,
            random_state=random_state,
        )

        # Check shapes
        self.assertEqual(Y_sklearn.shape, Y_jax.shape)

        print(Y_jax[:5, 0], Y_jax[:5, 1])

        # Optional: Visualize the embeddings
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].scatter(Y_sklearn[:, 0], Y_sklearn[:, 1], c=y)
        axes[0].set_title("Scikit-learn t-SNE")
        axes[1].scatter(Y_jax[:, 0], Y_jax[:, 1], c=y)
        axes[1].set_title("JAX t-SNE")

        # Save the figure to a file (you can specify your desired path)
        plt.savefig("tsne_comparison.png")
        plt.close()  # Close the figure after saving to avoid displaying it


if __name__ == "__main__":
    unittest.main()
