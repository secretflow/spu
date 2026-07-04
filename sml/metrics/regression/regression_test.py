# Copyright 2023 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
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
from sklearn import metrics

import spu.spu_pb2 as spu_pb2
import spu.utils.simulation as spsim

# add ops dir to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

from sml.metrics.regression.regression import (
    explained_variance_score,
    mean_squared_error,
    mean_poisson_deviance,
    mean_gamma_deviance,
    d2_tweedie_score,
)


class UnitTests(unittest.TestCase):
    def test_d2_tweedie_score(self):
        sim = spsim.Simulator.simple(
            2, spu_pb2.ProtocolKind.CHEETAH, spu_pb2.FieldType.FM128
        )

        power_list = [-1, 0, 1, 2, 3]
        weight_list = [
            None,
            jnp.array([0.5, 0.5, 0.5, 0.5]),
            jnp.array([0.5, 1, 2, 0.5]),
        ]

        # Test d2_tweedie_score
        y_true = jnp.array([0.5, 1, 2.5, 7])
        y_pred = jnp.array([1, 1, 5, 3.5])
        for p in power_list:
            for weight in weight_list:
                sk_result = metrics.d2_tweedie_score(
                    y_true, y_pred, sample_weight=weight, power=p
                )
                
                copts = spu_pb2.CompilerOptions()
                
                d2_tweedie_score.__name__ = f"d2_tweedie_score_weight{weight}_power{p}"
                if p == -1:
                    if weight == None:
                        spu_fn = spsim.sim_jax(sim, d2_tweedie_score, static_argnums=(3,), copts=copts, pphlo_ref=(None, None)) #d2_tweedie_score_weightNone_power-1
                    elif np.array_equiv(weight, jnp.array([0.5, 0.5, 0.5, 0.5])):
                        spu_fn = spsim.sim_jax(sim, d2_tweedie_score, static_argnums=(3,), copts=copts, pphlo_ref=(None, None)) #d2_tweedie_score_weight[0.5 0.5 0.5 0.5]_power-1
                    elif np.array_equiv(weight, jnp.array([0.5, 1, 2, 0.5])): 
                        spu_fn = spsim.sim_jax(sim, d2_tweedie_score, static_argnums=(3,), copts=copts, pphlo_ref=(None, None)) #d2_tweedie_score_weight[0.5 1.  2.  0.5]_power-1
                elif p == 0:
                    if weight == None:
                        spu_fn = spsim.sim_jax(sim, d2_tweedie_score, static_argnums=(3,), copts=copts, pphlo_ref=(None, None)) #d2_tweedie_score_weightNone_power0
                    elif np.array_equiv(weight, jnp.array([0.5, 0.5, 0.5, 0.5])):
                        spu_fn = spsim.sim_jax(sim, d2_tweedie_score, static_argnums=(3,), copts=copts, pphlo_ref=(None, None)) #d2_tweedie_score_weight[0.5 0.5 0.5 0.5]_power0
                    elif np.array_equiv(weight, jnp.array([0.5, 1, 2, 0.5])):
                        spu_fn = spsim.sim_jax(sim, d2_tweedie_score, static_argnums=(3,), copts=copts, pphlo_ref=(None, None)) #d2_tweedie_score_weight[0.5 1.  2.  0.5]_power0
                elif p == 1:
                    if weight == None:
                        spu_fn = spsim.sim_jax(sim, d2_tweedie_score, static_argnums=(3,), copts=copts, pphlo_ref=(None, None)) #d2_tweedie_score_weightNone_power1
                    elif np.array_equiv(weight, jnp.array([0.5, 0.5, 0.5, 0.5])):
                        spu_fn = spsim.sim_jax(sim, d2_tweedie_score, static_argnums=(3,), copts=copts, pphlo_ref=(None, None)) #d2_tweedie_score_weight[0.5 0.5 0.5 0.5]_power1
                    elif np.array_equiv(weight, jnp.array([0.5, 1, 2, 0.5])):
                        spu_fn = spsim.sim_jax(sim, d2_tweedie_score, static_argnums=(3,), copts=copts, pphlo_ref=(None, None)) #d2_tweedie_score_weight[0.5 1.  2.  0.5]_power1
                elif p == 2:
                    if weight == None:
                        spu_fn = spsim.sim_jax(sim, d2_tweedie_score, static_argnums=(3,), copts=copts, pphlo_ref=(None, None)) #d2_tweedie_score_weightNone_power2
                    elif np.array_equiv(weight, jnp.array([0.5, 0.5, 0.5, 0.5])):
                        spu_fn = spsim.sim_jax(sim, d2_tweedie_score, static_argnums=(3,), copts=copts, pphlo_ref=(None, None)) #d2_tweedie_score_weight[0.5 0.5 0.5 0.5]_power2
                    elif np.array_equiv(weight, jnp.array([0.5, 1, 2, 0.5])):
                        spu_fn = spsim.sim_jax(sim, d2_tweedie_score, static_argnums=(3,), copts=copts, pphlo_ref=(None, None)) #d2_tweedie_score_weight[0.5 1.  2.  0.5]_power2
                elif p == 3:
                    if weight == None:
                        spu_fn = spsim.sim_jax(sim, d2_tweedie_score, static_argnums=(3,), copts=copts, pphlo_ref=(None, None)) #d2_tweedie_score_weightNone_power3
                    elif np.array_equiv(weight, jnp.array([0.5, 0.5, 0.5, 0.5])):
                        spu_fn = spsim.sim_jax(sim, d2_tweedie_score, static_argnums=(3,), copts=copts, pphlo_ref=(None, None)) #d2_tweedie_score_weight[0.5 0.5 0.5 0.5]_power3
                    elif np.array_equiv(weight, jnp.array([0.5, 1, 2, 0.5])):
                        spu_fn = spsim.sim_jax(sim, d2_tweedie_score, static_argnums=(3,), copts=copts, pphlo_ref=(None, None)) #d2_tweedie_score_weight[0.5 1.  2.  0.5]_power3
                             
                spu_result = spu_fn(
                    y_true, y_pred, weight, p
                )
                try:
                    if spu_result == "skipped":
                        continue
                except:
                    pass
                print(spu_fn.pphlo)
                np.testing.assert_allclose(sk_result, spu_result, rtol=0, atol=1e-4)

    def test_explained_variance_score(self):
        sim = spsim.Simulator.simple(
            2, spu_pb2.ProtocolKind.CHEETAH, spu_pb2.FieldType.FM128
        )

        weight_list = [
            None,
            jnp.array([0.5, 0.5, 0.5, 0.5]),
            jnp.array([0.5, 1, 2, 0.5]),
        ]

        # Test explained_variance_score
        y_true = jnp.array([3, -0.5, 2, 7])
        y_pred = jnp.array([2.5, 0.0, 2, 8])
        for weight in weight_list:
            sk_result = metrics.explained_variance_score(
                y_true,
                y_pred,
                sample_weight=weight,
                multioutput="variance_weighted",
                force_finite=True,
            )
            
            copts = spu_pb2.CompilerOptions()
            
            explained_variance_score.__name__ = f"explained_variance_score_weight{weight}"
            if weight == None:
                spu_fn = spsim.sim_jax(sim, explained_variance_score, static_argnums=(3,), copts=copts, pphlo_ref=(None, None)) #explained_variance_score_weightNone
            elif np.array_equiv(weight, jnp.array([0.5, 0.5, 0.5, 0.5])):
                spu_fn = spsim.sim_jax(sim, explained_variance_score, static_argnums=(3,), copts=copts, pphlo_ref=(None, None)) #explained_variance_score_weight[0.5 0.5 0.5 0.5]
            elif np.array_equiv(weight, jnp.array([0.5, 1, 2, 0.5])):
                spu_fn = spsim.sim_jax(sim, explained_variance_score, static_argnums=(3,), copts=copts, pphlo_ref=(None, None)) #explained_variance_score_weight[0.5 1.  2.  0.5]
                
            spu_result = spu_fn(
                y_true, y_pred, weight, "variance_weighted"
            )
            try:
                if spu_result == "skipped":
                    continue
            except:
                pass
            print(spu_fn.pphlo)
            np.testing.assert_allclose(sk_result, spu_result, rtol=0, atol=1e-4)

    def test_mean_squared_error(self):
        sim = spsim.Simulator.simple(
            2, spu_pb2.ProtocolKind.CHEETAH, spu_pb2.FieldType.FM128
        )

        weight_list = [
            None,
            jnp.array([0.5, 0.5, 0.5, 0.5]),
            jnp.array([0.5, 1, 2, 0.5]),
        ]

        # Test mean_squared_error
        y_true = jnp.array([3, -0.5, 2, 7])
        y_pred = jnp.array([2.5, 0.0, 2, 8])
        for weight in weight_list:
            sk_result = metrics.mean_squared_error(
                y_true, y_pred, sample_weight=weight, squared=False
            )
            
            copts = spu_pb2.CompilerOptions()
            
            mean_squared_error.__name__ = f"mean_squared_error_weight{weight}"
            if weight == None:
                spu_fn = spsim.sim_jax(sim, mean_squared_error, static_argnums=(3, 4), copts=copts, pphlo_ref=(None, None)) #mean_squared_error_weightNone
            elif np.array_equiv(weight, jnp.array([0.5, 0.5, 0.5, 0.5])):
                spu_fn = spsim.sim_jax(sim, mean_squared_error, static_argnums=(3, 4), copts=copts, pphlo_ref=(None, None)) #mean_squared_error_weight[0.5 0.5 0.5 0.5]
            elif np.array_equiv(weight, jnp.array([0.5, 1, 2, 0.5])):
                spu_fn = spsim.sim_jax(sim, mean_squared_error, static_argnums=(3, 4), copts=copts, pphlo_ref=(None, None)) #mean_squared_error_weight[0.5 1.  2.  0.5]

            spu_result = spu_fn(
                y_true, y_pred, weight, "uniform_average", False
            )
            try:
                if spu_result == "skipped":
                    continue
            except:
                pass
            print(spu_fn.pphlo)
            np.testing.assert_allclose(sk_result, spu_result, rtol=0, atol=1e-4)

    def test_mean_poisson_deviance(self):
        sim = spsim.Simulator.simple(
            2, spu_pb2.ProtocolKind.CHEETAH, spu_pb2.FieldType.FM128
        )

        weight_list = [
            None,
            jnp.array([0.5, 0.5, 0.5, 0.5]),
            jnp.array([0.5, 1, 2, 0.5]),
        ]

        # Test mean_poisson_deviance
        y_true = jnp.array([2, 0, 1, 4])
        y_pred = jnp.array([0.5, 0.5, 2.0, 2.0])
        for weight in weight_list:
            sk_result = metrics.mean_poisson_deviance(
                y_true, y_pred, sample_weight=weight
            )
            
            copts = spu_pb2.CompilerOptions()
            
            mean_poisson_deviance.__name__ = f"mean_poisson_deviance_weight{weight}"
            if weight == None:
                spu_fn = spsim.sim_jax(sim, mean_poisson_deviance, copts=copts, pphlo_ref=(None, None)) #mean_poisson_deviance_weightNone
            elif np.array_equiv(weight, jnp.array([0.5, 0.5, 0.5, 0.5])):
                spu_fn = spsim.sim_jax(sim, mean_poisson_deviance, copts=copts, pphlo_ref=(None, None)) #mean_poisson_deviance_weight[0.5 0.5 0.5 0.5]
            elif np.array_equiv(weight, jnp.array([0.5, 1, 2, 0.5])):
                spu_fn = spsim.sim_jax(sim, mean_poisson_deviance, copts=copts, pphlo_ref=(None, None)) #mean_poisson_deviance_weight[0.5 1.  2.  0.5]

            spu_result = spu_fn(
                y_true, y_pred, weight
            )
            try:
                if spu_result == "skipped":
                    continue
            except:
                pass
            print(spu_fn.pphlo)
            np.testing.assert_allclose(sk_result, spu_result, rtol=0, atol=1e-4)

    def test_mean_gamma_deviance(self):
        sim = spsim.Simulator.simple(
            2, spu_pb2.ProtocolKind.CHEETAH, spu_pb2.FieldType.FM128
        )

        weight_list = [
            None,
            jnp.array([0.5, 0.5, 0.5, 0.5]),
            jnp.array([0.5, 1, 2, 0.5]),
        ]

        # Test mean_gamma_deviance
        y_true = jnp.array([2, 0.5, 1, 4])
        y_pred = jnp.array([0.5, 0.5, 2.0, 2.0])
        for weight in weight_list:
            sk_result = metrics.mean_gamma_deviance(
                y_true, y_pred, sample_weight=weight
            )
            
            copts = spu_pb2.CompilerOptions()
            
            mean_gamma_deviance.__name__ = f"mean_gamma_deviance_weight{weight}"
            if weight == None:
                spu_fn = spsim.sim_jax(sim, mean_gamma_deviance, copts=copts, pphlo_ref=(None, None)) #mean_gamma_deviance_weightNone
            elif np.array_equiv(weight, jnp.array([0.5, 0.5, 0.5, 0.5])):
                spu_fn = spsim.sim_jax(sim, mean_gamma_deviance, copts=copts, pphlo_ref=(None, None)) #mean_gamma_deviance_weight[0.5 0.5 0.5 0.5]
            elif np.array_equiv(weight, jnp.array([0.5, 1, 2, 0.5])):
                spu_fn = spsim.sim_jax(sim, mean_gamma_deviance, copts=copts, pphlo_ref=(None, None)) #mean_gamma_deviance_weight[0.5 1.  2.  0.5]

            spu_result = spu_fn(y_true, y_pred, weight)
            try:
                if spu_result == "skipped":
                    continue
            except:
                pass
            print(spu_fn.pphlo)
            np.testing.assert_allclose(sk_result, spu_result, rtol=0, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
