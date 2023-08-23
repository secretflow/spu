import os
import sys
import unittest

from sklearn.datasets import load_iris
import jax
import jax.numpy as jnp

import spu.spu_pb2 as spu_pb2
import spu.utils.simulation as spsim

# Add the library directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))
from sml.gaussian_process._gpc import GaussianProcessClassifier


class UnitTests(unittest.TestCase):
    def test_gpc(self):
        sim = spsim.Simulator.simple(
            3, spu_pb2.ProtocolKind.ABY3, spu_pb2.FieldType.FM64
        )

        # Test GaussianProcessClassifier
        def proc(x, y):
            model = GaussianProcessClassifier(
                max_iter_predict=10
            )
            model.fit(x, y)

            pred = model.predict(x)
            return pred
        
        # Create dataset
        x, y = load_iris(return_X_y=True)
        x = x[45:55, :]
        y = y[45:55]

        # Run
        result = spsim.sim_jax(sim, proc)(x, y)
        print(result)
        print(y)
        print("Accuracy: ", jnp.sum(result == y)/len(y))


unittest.main()
