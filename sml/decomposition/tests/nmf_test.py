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
import unittest

import jax.numpy as jnp
import numpy as np
from jax import random
from sklearn.decomposition import NMF as SklearnNMF
import spu.spu_pb2 as spu_pb2
import spu.utils.simulation as spsim

# Add the sml directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

from sml.decomposition.nmf import NMF


class UnitTests(unittest.TestCase):
    def test_nmf(self):
        config = spu_pb2.RuntimeConfig(
            protocol=spu_pb2.ProtocolKind.ABY3,
            field=spu_pb2.FieldType.FM128,
            fxp_fraction_bits=30,
        )
        sim = spsim.Simulator(3, config)

        # Test fit_transform
        def proc1(X, random_matrixA, random_matrixB):
            model = NMF(
                n_components=n_components,
                l1_ratio=l1_ratio,
                alpha_W=alpha_W,
                random_matrixA=random_matrixA,
                random_matrixB=random_matrixB,
            )

            W = model.fit_transform(X)
            H = model._components
            X_reconstructed = model.inverse_transform(W)
            return W, H, X_reconstructed

        # Test fit and transform
        def proc2(X, random_matrixA, random_matrixB):
            model = NMF(
                n_components=n_components,
                l1_ratio=l1_ratio,
                alpha_W=alpha_W,
                random_matrixA=random_matrixA,
                random_matrixB=random_matrixB,
            )

            model.fit(X)
            W = model.transform(X, transform_iter=40)
            H = model._components
            X_reconstructed = model.inverse_transform(W)
            return W, H, X_reconstructed

        # Create a simple dataset and random_matrix
        X = np.random.randint(1,100,(1000,10))
        X = np.array(X,dtype=float)
        n_samples, n_features = X.shape
        n_components = 5
        random_seed = 0
        random_state = np.random.RandomState(random_seed)
        A = random_state.standard_normal(size=(n_components, n_features))
        B = random_state.standard_normal(size=(n_samples, n_components))
        l1_ratio = 0
        alpha_W = 0.01

        # Run the simulation
        W, H, X_reconstructed = spsim.sim_jax(sim, proc1)(X, A, B)
        print("W_matrix_spu: ", W)
        print("H_matrix_spu: ", H)
        print("X_reconstructed_spu: ", X_reconstructed)

        # Run the simulation_seperate
        W_seperate, H_seperate, X_reconstructed_seperate = spsim.sim_jax(sim, proc2)(X, A, B)
        print("W_matrix_spu_seperate: ", W_seperate)
        print("H_matrix_spu_seperate: ", H_seperate)
        print("X_reconstructed_spu_seperate: ", X_reconstructed_seperate)

        # sklearn
        model = SklearnNMF(n_components=n_components, init='random', random_state=random_seed, l1_ratio=l1_ratio, solver="mu", alpha_W=alpha_W)
        W_Sklearn = model.fit_transform(X)
        H_Sklearn = model.components_
        X_reconstructed_Sklearn = model.inverse_transform(W_Sklearn)
        print("W_matrix_sklearn: ", W_Sklearn)
        print("H_matrix_sklearn: ", H_Sklearn)
        print("X_reconstructed_sklearn: ", X_reconstructed_Sklearn)

        # sklearn_seperate
        model = SklearnNMF(n_components=n_components, init='random', random_state=random_seed, l1_ratio=l1_ratio, solver="mu", alpha_W=alpha_W)
        model.fit(X)
        W_Sklearn_seperate = model.transform(X)
        H_Sklearn_seperate = model.components_
        X_reconstructed_Sklearn_seperate = model.inverse_transform(W_Sklearn_seperate)
        print("W_matrix_sklearn_seperate: ", W_Sklearn_seperate)
        print("H_matrix_sklearn_seperate: ", H_Sklearn_seperate)
        print("X_reconstructed_sklearn_seperate: ", X_reconstructed_Sklearn_seperate)


if __name__ == "__main__":
    unittest.main()
