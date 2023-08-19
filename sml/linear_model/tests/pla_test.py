import unittest

import pandas as pd
import sklearn.linear_model as sk
from sklearn.datasets import load_iris

import jax.numpy as jnp
import spu.spu_pb2 as spu_pb2  # type: ignore
import spu.utils.simulation as spsim
from sml.perceptron.pla import Perceptron


class UnitTests(unittest.TestCase):
    def test_pla(self):
        sim = spsim.Simulator.simple(
            3, spu_pb2.ProtocolKind.ABY3, spu_pb2.FieldType.FM64
        )

        def proc(x, y):
            model = Perceptron(
                max_iter=100,
                eta0=1.0,
                penalty='elasticnet',
                alpha=0.0001,
                l1_ratio=0.15,
            )

            return model.fit(x, y).predict(x)

        def load_data():
            iris = load_iris()
            df = pd.DataFrame(
                iris.data,
                columns=['sepal length', 'sepal width', 'petal length', 'petal width'],
            )
            df['label'] = iris.target

            # only use sepal length and sepal width features
            # 100 samples
            data = jnp.array(df.iloc[0:100, [0, 1, -1]])
            x, y = data[:, :-1], data[:, -1]

            # y is -1 or 1
            y = jnp.sign(y)
            y = jnp.where(y == 0, -1, y)
            y = y.reshape((y.shape[0], 1))
            return x, y

        # load mock data
        x, y = load_data()
        n_samples = len(y)

        # compare with sklearn
        sk_pla = sk.Perceptron(
            max_iter=100, eta0=1.0, penalty='elasticnet', alpha=0.0001, l1_ratio=0.15
        )
        result_sk = sk_pla.fit(x, y).predict(x)
        result_sk = result_sk.reshape(result_sk.shape[0], 1)
        acc_sk = jnp.sum((result_sk == y)) / n_samples * 100

        # run with spu
        result = spsim.sim_jax(sim, proc)(x, y)
        result = result.reshape(result.shape[0], 1)
        acc_ = jnp.sum((result == y)) / n_samples * 100

        # print acc
        print(f"Accuracy in SKlearn: {acc_sk:.2f}%")
        print(f"Accuracy in SPU: {acc_:.2f}%")


if __name__ == "__main__":
    unittest.main()
