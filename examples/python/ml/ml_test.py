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
import logging
import sys
import unittest
from time import perf_counter
import os
import warnings
from io import StringIO

import multiprocess
import numpy.testing as npt
import pandas as pd
import inspect

import spu.utils.distributed as ppd


with open("examples/python/conf/3pc.json", 'r') as file:
    conf = json.load(file)

logger = logging.getLogger(ppd.__name__)
logger.setLevel(level=logging.WARN)


FILENAME = "ml_test_run_data.csv"


def read_history_record():
    return pd.read_csv(StringIO(content), index_col=False)


_test_perf_table = read_history_record()


def compute_history_average():
    test_pints = _test_perf_table["name"].unique()

    avg_dict = {}
    for tp in test_pints:
        history = _test_perf_table[_test_perf_table["name"] == tp]
        avg = history["duration"].mean()
        avg_dict[tp] = avg

    return avg_dict


_perf_history_avg = compute_history_average()


def add_profile_data(name, duration):
    global _test_perf_table
    # Check data
    valid = True
    if name in _perf_history_avg:
        history_avg = _perf_history_avg[name]
        if duration > 1.05 * history_avg:
            warnings.warn(
                f"Testpoint {name} is more than 5% slower in this job, history average = {history_avg}, current = {duration}.\n"
            )
            valid = True

    if valid:
        # Save result to table
        new_row = pd.DataFrame({'name': name, 'duration': duration}, index=[0])
        _test_perf_table = pd.concat([_test_perf_table, new_row], ignore_index=True)


def profile_test_point(foo, *args, **kwargs):
    # Get function name
    test_fun_name = inspect.getmodule(foo).__name__

    # Run with perf counter
    start = perf_counter()
    result = foo(*args, **kwargs)
    end = perf_counter()

    # Save result to table
    add_profile_data(test_fun_name, end - start)
    return result


def save_perf_report():
    buf = _test_perf_table.to_csv(index=False)


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

        rt_config = conf["devices"]["SPU"]["config"]["runtime_config"]
        rt_config["enable_pphlo_profile"] = False
        rt_config["enable_hal_profile"] = False
        ppd.init(conf["nodes"], conf["devices"])

    @classmethod
    def tearDownClass(cls):
        for worker in cls.workers:
            worker.kill()

    def test_flax_mlp(self):
        from examples.python.ml.flax_mlp import flax_mlp

        param = profile_test_point(flax_mlp.run_on_spu)

        score = flax_mlp.compute_score(ppd.get(param), 'spu')
        self.assertGreater(score, 0.95)

    def test_flax_vae(self):
        from examples.python.ml.flax_vae import flax_vae

        flax_vae.args.num_epochs = 1
        flax_vae.args.num_steps = 10
        metrics = profile_test_point(flax_vae.train)

        self.assertTrue(260 < metrics['loss'] < 270)
        self.assertTrue(240 < metrics['bce'] < 247)
        self.assertTrue(20 < metrics['kld'] < 23)

    def test_flax_resnet(self):
        from examples.python.ml.flax_resnet import flax_resnet

        flax_resnet.args.num_epochs = 1
        flax_resnet.args.num_steps = 1
        metrics = profile_test_point(flax_resnet.train)

        self.assertTrue(0 < metrics['loss'] < 5)

    def test_haiku_lstm(self):
        from examples.python.ml.haiku_lstm import haiku_lstm

        haiku_lstm.args.num_steps = 1
        _, train_loss = ppd.get(
            profile_test_point(
                haiku_lstm.train_model,
                haiku_lstm.train_ds,
                haiku_lstm.valid_ds,
                run_on_spu=False,
            )
        )
        self.assertTrue(0.4 < train_loss < 0.6)

    def test_jax_kmeans(self):
        from examples.python.ml.jax_kmeans import jax_kmeans

        cpu_centers, cpu_labels = jax_kmeans.run_on_cpu()
        spu_centers, spu_labels = profile_test_point(jax_kmeans.run_on_spu)
        npt.assert_array_almost_equal(cpu_centers, spu_centers, decimal=4)
        npt.assert_array_equal(cpu_labels, spu_labels)

    def test_jax_lr(self):
        from examples.python.ml.jax_lr import jax_lr

        w, b = profile_test_point(jax_lr.run_on_spu)
        score = jax_lr.compute_score(w, b, 'spu')

        self.assertGreater(score, 0.95)

    def test_jax_svm(self):
        from examples.python.ml.jax_svm import jax_svm

        jax_svm.args.n_epochs = 5
        w, b = jax_svm.run_on_cpu()
        auc_cpu = jax_svm.compute_score(w, b, 'cpu')
        self.assertGreater(auc_cpu, 0.6)

        w, b = profile_test_point(jax_svm.run_on_spu)
        auc_spu = jax_svm.compute_score(w, b, 'spu')

        self.assertGreater(auc_spu, 0.6)
        npt.assert_almost_equal(auc_cpu, auc_spu, 6)

    def test_jraph_gnn(self):
        from examples.python.ml.jraph_gnn import jraph_gnn

        accuracy = profile_test_point(
            jraph_gnn.optimize_club, num_steps=30, run_on_spu=True
        )

        self.assertGreater(accuracy, 0.75)

    def test_ss_lr(self):
        from examples.python.ml.ss_lr import ss_lr

        ss_lr.args.epochs = 1
        ss_lr.dataset_config["n_samples"] = 50000
        score, train_time, predict_time = ss_lr.train()
        self.assertGreater(score, 0.75)

        add_profile_data("test_ss_lr_train", train_time)
        add_profile_data("test_ss_lr_predict", predict_time)

    def test_ss_xgb(self):
        from examples.python.ml.ss_xgb import ss_xgb

        score, train_time, predict_time = ss_xgb.main()
        self.assertGreater(score, 0.95)

        add_profile_data("test_ss_xgb_train", train_time)
        add_profile_data("test_ss_xgb_predict", predict_time)

    def test_stax_mnist_classifier(self):
        from examples.python.ml.stax_mnist_classifier import stax_mnist_classifier

        stax_mnist_classifier.num_batches = 10
        accuracy = profile_test_point(stax_mnist_classifier.run_spu)
        self.assertGreater(accuracy, 0.7)

    def test_stax_nn(self):
        from examples.python.ml.stax_nn import stax_nn

        stax_nn.args.epoch = 1
        score = profile_test_point(
            stax_nn.run_model, model_name="network_a", run_cpu=False
        )
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
    # suite.addTest(UnitTests('test_flax_resnet'))
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

    save_perf_report()

    sys.exit(ret)
