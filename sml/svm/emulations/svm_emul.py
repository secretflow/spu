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
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

import spu.spu_pb2 as spu_pb2  # type: ignore
from sml.svm.svm import SVM
import sml.utils.emulation as emulation

def emul_SVM(mode: emulation.Mode.MULTIPROCESS):
    def proc(x0, x1, y0, y1):
            rbf_svm = SVM()
            rbf_svm.fit(x0,y0)

            return rbf_svm.predict(x0, y0, x1)

    def load_data():
            breast_cancer = datasets.load_breast_cancer()
            data = breast_cancer.data
            data = data/(jnp.max(data)-jnp.min(data))
            target = breast_cancer.target
            X_train, X_test, y_train, y_test = train_test_split(
                data, target, test_size=0.2, random_state=1)

            party_split_num = len(X_train)//2
            
            y_train[y_train!=1] = -1
            X_train, X_test, y_train, y_test = jnp.array(X_train), jnp.array(X_test), jnp.array(y_train), jnp.array(y_test)
            
            return X_train, X_test, y_train, y_test

    try:
        # bandwidth and latency only work for docker mode
        emulator = emulation.Emulator(
            "examples/python/conf/3pc.json", mode, bandwidth=300, latency=20
        )
        emulator.up()

        # load data
        X_train, X_test, y_train, y_test = load_data()

        # mark these data to be protected in SPU
        X_train, X_test, y_train, y_test = emulator.seal(X_train, X_test, y_train, y_test)
        result = emulator.run(proc)(X_train, X_test, y_train, y_test)
        print("result\n", result)

        print("y_test\n", y_test)
    finally:
        emulator.down()


if __name__ == "__main__":
    emul_SVM(emulation.Mode.MULTIPROCESS)
