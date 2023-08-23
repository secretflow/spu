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

import numpy as np
from sklearn.decomposition import NMF as SklearnNMF

# Add the sml directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

import sml.utils.emulation as emulation
from sml.decomposition.nmf import NMF


def emul_nmf(mode: emulation.Mode.MULTIPROCESS):
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
        err = model.reconstruction_err_
        return W, H, X_reconstructed, err

    try:
        # bandwidth and latency only work for docker mode
        conf_path = "sml/decomposition/emulations/3pc.json"
        emulator = emulation.Emulator(conf_path, mode, bandwidth=300, latency=20)
        emulator.up()
        # Create a simple dataset and random_matrix
        X = np.random.randint(1, 100, (1000, 10))
        X = np.array(X, dtype=float)
        n_samples, n_features = X.shape
        n_components = 5
        random_seed = 0
        random_state = np.random.RandomState(random_seed)
        A = random_state.standard_normal(size=(n_components, n_features))
        B = random_state.standard_normal(size=(n_samples, n_components))
        l1_ratio = 0.1
        alpha_W = 0.01

        X_spu = emulator.seal(X)
        A_spu = emulator.seal(A)
        B_spu = emulator.seal(B)

        # Run the emulation
        W, H, X_reconstructed, err = emulator.run(proc1)(X_spu, A_spu, B_spu)
        print("reconstruction_error: ", err)

        # sklearn
        model = SklearnNMF(
            n_components=n_components,
            init='random',
            random_state=random_seed,
            l1_ratio=l1_ratio,
            solver="mu",
            alpha_W=alpha_W,
        )
        W_Sklearn = model.fit_transform(X)
        H_Sklearn = model.components_
        X_reconstructed_Sklearn = model.inverse_transform(W_Sklearn)
        err = model.reconstruction_err_
        print("reconstruction_error_sklearn: ", err)
        assert np.allclose(W_Sklearn, W, atol=5e-1)
        assert np.allclose(H_Sklearn, H, atol=5e-1)
        assert np.allclose(X_reconstructed_Sklearn, X_reconstructed, atol=5e-1)

        # sklearn_seperate
        model = SklearnNMF(
            n_components=n_components,
            init='random',
            random_state=random_seed,
            l1_ratio=l1_ratio,
            solver="mu",
            alpha_W=alpha_W,
        )
        model.fit(X)
        W_Sklearn_seperate = model.transform(X)
        H_Sklearn_seperate = model.components_
        X_reconstructed_Sklearn_seperate = model.inverse_transform(W_Sklearn_seperate)
        print("W_matrix_sklearn_seperate: ", W_Sklearn_seperate)
        print("H_matrix_sklearn_seperate: ", H_Sklearn_seperate)
        print("X_reconstructed_sklearn_seperate: ", X_reconstructed_Sklearn_seperate)

    finally:
        emulator.down()


def emul_nmf_seperate(mode: emulation.Mode.MULTIPROCESS):
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

    try:
        # bandwidth and latency only work for docker mode
        conf_path = "sml/decomposition/emulations/3pc.json"
        emulator = emulation.Emulator(conf_path, mode, bandwidth=300, latency=20)
        emulator.up()
        # Create a simple dataset and random_matrix
        X = np.random.randint(1, 100, (1000, 10))
        X = np.array(X, dtype=float)
        n_samples, n_features = X.shape
        n_components = 5
        random_seed = 0
        random_state = np.random.RandomState(random_seed)
        A = random_state.standard_normal(size=(n_components, n_features))
        B = random_state.standard_normal(size=(n_samples, n_components))
        l1_ratio = 0.1
        alpha_W = 0.01

        X_spu = emulator.seal(X)
        A_spu = emulator.seal(A)
        B_spu = emulator.seal(B)

        # Run the emulation_seperate
        W_seperate, H_seperate, X_reconstructed_seperate = emulator.run(proc2)(X_spu, A_spu, B_spu)
        print("W_matrix_spu_seperate: ", W_seperate[:5,:5])
        print("H_matrix_spu_seperate: ", H_seperate[:5,:5])
        print("X_reconstructed_spu_seperate: ", X_reconstructed_seperate[:5,:5])

        # sklearn_seperate
        model = SklearnNMF(
            n_components=n_components,
            init='random',
            random_state=random_seed,
            l1_ratio=l1_ratio,
            solver="mu",
            alpha_W=alpha_W,
        )
        model.fit(X)
        W_Sklearn_seperate = model.transform(X)
        H_Sklearn_seperate = model.components_
        X_reconstructed_Sklearn_seperate = model.inverse_transform(W_Sklearn_seperate)
        print("W_matrix_sklearn_seperate: ", W_Sklearn_seperate[:5,:5])
        print("H_matrix_sklearn_seperate: ", H_Sklearn_seperate[:5,:5])
        print("X_reconstructed_sklearn_seperate: ", X_reconstructed_Sklearn_seperate[:5,:5])

    finally:
        emulator.down()


if __name__ == "__main__":
    emul_nmf(emulation.Mode.MULTIPROCESS)
    emul_nmf_seperate(emulation.Mode.MULTIPROCESS)
