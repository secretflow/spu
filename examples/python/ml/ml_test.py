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


import json
import sys
import unittest

import multiprocess
import numpy.testing as npt
from sklearn import metrics

import examples.python.utils.dataset_utils as dsutil
import spu.utils.distributed as ppd

with open("examples/python/conf/3pc.json", 'r') as file:
    conf = json.load(file)


class UnitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workers = []
        for node_id in conf["nodes"].keys():
            worker = multiprocess.Process(
                target=ppd.RPC.serve, args=(node_id, conf["nodes"])
            )
            worker.start()
            cls.workers.append(worker)
        import time

        # wait for all process serving.
        time.sleep(0.05)

        ppd.init(conf["nodes"], conf["devices"])

    @classmethod
    def tearDownClass(cls):
        for worker in cls.workers:
            worker.kill()

    def test_flax_mlp(self):
        from examples.python.ml.flax_mlp import flax_mlp

        x_test, y_test = dsutil.breast_cancer(slice(None, None, None), False)
        params = flax_mlp.run_on_spu()
        params_r = ppd.get(params)
        y_predict = flax_mlp.predict(params_r, x_test)
        self.assertGreater(metrics.roc_auc_score(y_test, y_predict), 0.95)

    def test_flax_vae(self):
        from examples.python.ml.flax_vae import flax_vae

        flax_vae.args.num_epochs = 1
        flax_vae.args.num_steps = 1
        metrics = flax_vae.train()
        self.assertTrue(530 < metrics['loss'] < 540)
        self.assertTrue(530 < metrics['bce'] < 540)
        self.assertTrue(0.8 < metrics['kld'] < 1)

    def test_haiku_lstm(self):
        from examples.python.ml.haiku_lstm import haiku_lstm

        haiku_lstm.args.num_steps = 1
        _, train_loss = ppd.get(
            haiku_lstm.train_model(
                haiku_lstm.train_ds, haiku_lstm.valid_ds, run_on_spu=False
            )
        )
        self.assertTrue(0.4 < train_loss < 0.6)

    def test_jax_kmeans(self):
        from examples.python.ml.jax_kmeans import jax_kmeans

        cpu_centers, cpu_labels = jax_kmeans.run_on_cpu()
        spu_centers, spu_labels = jax_kmeans.run_on_spu()
        npt.assert_array_almost_equal(cpu_centers, spu_centers, decimal=4)
        npt.assert_array_equal(cpu_labels, spu_labels)

    def test_jax_lr(self):
        from examples.python.ml.jax_lr import jax_lr

        x_test, y_test = dsutil.breast_cancer(slice(None, None, None), False)
        w, b = jax_lr.run_on_spu()
        self.assertGreater(
            metrics.roc_auc_score(
                y_test, jax_lr.predict(x_test, ppd.get(w), ppd.get(b))
            ),
            0.95,
        )

    def test_jax_svm(self):
        from examples.python.ml.jax_svm import jax_svm

        jax_svm.args.n_epochs = 5
        auc_cpu = jax_svm.run_on_cpu()
        auc_spu = jax_svm.run_on_spu()
        self.assertGreater(auc_cpu, 0.6)
        self.assertGreater(auc_spu, 0.6)
        npt.assert_almost_equal(auc_cpu, auc_spu, 6)

    def test_jraph_gnn(self):
        from examples.python.ml.jraph_gnn import jraph_gnn

        accuracy = jraph_gnn.optimize_club(num_steps=30, run_on_spu=True)
        self.assertGreater(accuracy, 0.75)

    def test_ss_lr(self):
        from examples.python.ml.ss_lr import ss_lr

        ss_lr.args.epochs = 1
        ss_lr.dataset_config["n_samples"] = 50000
        score = ss_lr.train()
        self.assertGreater(score, 0.75)

    def test_ss_xgb(self):
        from examples.python.ml.ss_xgb import ss_xgb

        score = ss_xgb.main()
        self.assertGreater(score, 0.95)

    def test_stax_mnist_classifier(self):
        from examples.python.ml.stax_mnist_classifier import stax_mnist_classifier

        stax_mnist_classifier.num_batches = 10
        accuracy = stax_mnist_classifier.run_spu()
        self.assertGreater(accuracy, 0.7)

    def test_stax_nn(self):
        from examples.python.ml.stax_nn import stax_nn

        stax_nn.args.epoch = 1
        score = stax_nn.main()
        self.assertGreater(score, 0.9)

    def test_tf_experiment(self):
        from examples.python.ml.tf_experiment import tf_experiment

        score = tf_experiment.run_fit_manual_grad_spu()
        self.assertGreater(score, 0.9)

    def test_torch_experiment(self):
        from examples.python.ml.torch_experiment import torch_experiment

        model = torch_experiment.LinearRegression()
        torch_experiment.train(model)
        score = torch_experiment.run_inference_on_spu(model)
        self.assertGreater(score, 0.9)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(UnitTests('test_flax_mlp'))
    suite.addTest(UnitTests('test_flax_vae'))
    suite.addTest(UnitTests('test_haiku_lstm'))
    suite.addTest(UnitTests('test_jax_kmeans'))
    suite.addTest(UnitTests('test_jax_lr'))
    suite.addTest(UnitTests('test_jax_svm'))
    suite.addTest(UnitTests('test_jraph_gnn'))
    suite.addTest(UnitTests('test_ss_lr'))
    suite.addTest(UnitTests('test_ss_xgb'))
    suite.addTest(UnitTests('test_stax_mnist_classifier'))
    suite.addTest(UnitTests('test_stax_nn'))
    suite.addTest(UnitTests('test_tf_experiment'))
    suite.addTest(UnitTests('test_torch_experiment'))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner(failfast=True)
    ret = not runner.run(suite()).wasSuccessful()
    sys.exit(ret)
