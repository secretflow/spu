'''
Author: Li Zhihang
Date: 2024-07-16 11:21:02
LastEditTime: 2024-07-17 15:26:04
FilePath: /klaus/spu/sml/adaboost/emulations/adaboost_emul.py
Description: 
'''
import time

import jax.numpy as jnp
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

import sml.utils.emulation as emulation
from sml.adaboost.adaboost import AdaBoostClassifier as sml_Adaboost

MAX_DEPTH = 3
CONFIG_FILE = emulation.CLUSTER_ABY3_3PC

def emul_ada(mode=emulation.Mode.MULTIPROCESS):
    def proc_wrapper(
        estimator = "dtc",
        n_estimators = 50,
        max_depth = MAX_DEPTH,
        learning_rate = 1.0,
        n_classes = 3,
    ):
        ada_custom = sml_Adaboost(
            estimator = "dtc",
            n_estimators = 50,
            max_depth = MAX_DEPTH,
            learning_rate = 1.0,
            n_classes = 3,
        )
        
        def proc(X, y):
            ada_custom_fit = ada_custom.fit(X, y, sample_weight=None)
            result = ada_custom_fit.predict(X)
            return result
        
        return proc
    
    def load_data():
        iris = load_iris()
        iris_data, iris_label = jnp.array(iris.data), jnp.array(iris.target)
        # sorted_features: n_samples * n_features_in
        n_samples, n_features_in = iris_data.shape
        n_labels = len(jnp.unique(iris_label))
        sorted_features = jnp.sort(iris_data, axis=0)
        new_threshold = (sorted_features[:-1, :] + sorted_features[1:, :]) / 2
        new_features = jnp.greater_equal(
            iris_data[:, :], new_threshold[:, jnp.newaxis, :]
        )
        new_features = new_features.transpose([1, 0, 2]).reshape(n_samples, -1)

        X, y = new_features[:, ::3], iris_label[:]
        return X, y
    
    try:
        # bandwidth and latency only work for docker mode
        emulator = emulation.Emulator(CONFIG_FILE, mode, bandwidth=300, latency=20)
        emulator.up()

        # load mock data
        X, y = load_data()
        n_labels = jnp.unique(y).shape[0]

        # compare with sklearn
        base_estimator = DecisionTreeClassifier(max_depth=3)  # 基分类器
        ada = AdaBoostClassifier(estimator=base_estimator, n_estimators=3, learning_rate=1.0, algorithm="SAMME")

        start = time.time()
        ada = ada.fit(X, y)
        score_plain = ada.score(X, y)
        end = time.time()
        print(f"Running time in SKlearn: {end - start:.2f}s")

        # mark these data to be protected in SPU
        X_spu, y_spu = emulator.seal(X, y)

        # run
        proc = proc_wrapper(
            estimator = "dtc",
            n_estimators = 3,
            max_depth = MAX_DEPTH,
            learning_rate = 1.0,
            n_classes = 3,
        )
        start = time.time()
        # 不可以使用bootstrap，否则在spu运行的正确率很低
        result = emulator.run(proc)(X_spu, y_spu)
        end = time.time()
        score_encrpted = jnp.mean((result == y))
        print(f"Running time in SPU: {end - start:.2f}s")

        # print acc
        print(f"Accuracy in SKlearn: {score_plain:.2f}")
        print(f"Accuracy in SPU: {score_encrpted:.2f}")

    finally:
        emulator.down()


if __name__ == "__main__":
    emul_ada(emulation.Mode.MULTIPROCESS)
