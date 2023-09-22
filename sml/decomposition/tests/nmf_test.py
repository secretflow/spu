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

import numpy as np
from sklearn.decomposition import NMF as SklearnNMF

import spu.spu_pb2 as spu_pb2
import spu.utils.simulation as spsim

# Add the sml directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

from sml.decomposition.nmf import NMF


class UnitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print(" ========= start test of NMF package ========= \n")
        cls.random_seed = 0
        np.random.seed(cls.random_seed)
        # NMF must use FM128 now, for heavy use of non-linear & matrix operations
        config = spu_pb2.RuntimeConfig(
            protocol=spu_pb2.ProtocolKind.ABY3,
            field=spu_pb2.FieldType.FM128,
            fxp_fraction_bits=30,
        )
        cls.sim = spsim.Simulator(3, config)

        # generate some dummy test datas
        cls.test_data = np.random.randint(1, 100, (100, 10)) * 1.0
        n_samples, n_features = cls.test_data.shape

        # random matrix should be generated in plaintext.
        cls.n_components = 5
        random_state = np.random.RandomState(cls.random_seed)
        cls.random_A = random_state.standard_normal(size=(cls.n_components, n_features))
        cls.random_B = random_state.standard_normal(size=(n_samples, cls.n_components))

        # test hyper-parameters settings
        cls.l1_ratio = 0.1
        cls.alpha_W = 0.01

    @classmethod
    def tearDownClass(cls):
        print(" ========= test of NMF package end ========= \n")

    def _nmf_test_main(self, plaintext=True, mode="uniform"):
        # uniform means model is fitted by fit_transform method
        # seperate means model is fitted by first fit then transform
        assert mode in ["uniform", "seperate"]

        # must define here, because test may run simultaneously
        model = (
            SklearnNMF(
                n_components=self.n_components,
                init='random',
                random_state=self.random_seed,
                l1_ratio=self.l1_ratio,
                solver="mu",  # sml only implement this solver now.
                alpha_W=self.alpha_W,
            )
            if plaintext
            else NMF(
                n_components=self.n_components,
                l1_ratio=self.l1_ratio,
                alpha_W=self.alpha_W,
                random_matrixA=self.random_A,
                random_matrixB=self.random_B,
            )
        )

        def proc(x):
            if mode == "uniform":
                W = model.fit_transform(x)
            else:
                model.fit(x)
                W = model.transform(x)

            H = model.components_
            X_reconstructed = model.inverse_transform(W)
            err = model.reconstruction_err_

            return W, H, X_reconstructed, err

        run_func = (
            proc
            if plaintext
            else spsim.sim_jax(
                self.sim,
                proc,
            )
        )

        return run_func(self.test_data)

    def test_nmf_uniform(self):
        print("==============  start test of nmf uniform ==============\n")

        W, H, X_reconstructed, err = self._nmf_test_main(False, "uniform")
        W_sk, H_sk, X_reconstructed_sk, err_sk = self._nmf_test_main(True, "uniform")

        np.testing.assert_allclose(err, err_sk, rtol=1, atol=1e-1)
        np.testing.assert_allclose(W, W_sk, rtol=1, atol=1e-1)
        np.testing.assert_allclose(H, H_sk, rtol=1, atol=1e-1)
        np.testing.assert_allclose(
            X_reconstructed, X_reconstructed_sk, rtol=1, atol=1e-1
        )

        print("==============  nmf uniform test pass  ==============\n")

    def test_nmf_seperate(self):
        print("==============  start test of nmf seperate ==============\n")

        W, H, X_reconstructed, err = self._nmf_test_main(False, "seperate")
        W_sk, H_sk, X_reconstructed_sk, err_sk = self._nmf_test_main(True, "seperate")

        np.testing.assert_allclose(err, err_sk, rtol=1, atol=1e-1)
        np.testing.assert_allclose(W, W_sk, rtol=1, atol=1e-1)
        np.testing.assert_allclose(H, H_sk, rtol=1, atol=1e-1)
        np.testing.assert_allclose(
            X_reconstructed, X_reconstructed_sk, rtol=1, atol=1e-1
        )

        print("==============  nmf seperate test pass ==============\n")


if __name__ == "__main__":
    unittest.main()
